"""ExamForge AI: preprocessing, LoRA-only and SFT+LoRA experiments (CORRECTED version, no DPO).

Run from this folder. See CHANGES.md for the list of fixes (search for "FIX:" in this file).

Command line (long jobs: start them with nohup, see README / notebook):
  python examforge_experiments.py check-eduqg        (EduQG loader/filters only, no HuggingFace needed)
  python examforge_experiments.py preprocess [--test-per-source 30]
  python examforge_experiments.py check-masking
  python examforge_experiments.py train-lora-only --trial baseline [--resume]
  python examforge_experiments.py train-sft-lora  --trial baseline [--max-train 3000 --max-val 150 --epochs 1]
"""
from __future__ import annotations
import argparse, hashlib, json, random, re
from pathlib import Path
from collections import defaultdict
from typing import Any

SEED = 42
MODEL_NAME = "Qwen/Qwen3-1.7B"
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"
EDUQG_DIR = ROOT / "data" / "raw" / "eduqg"
MIN_WORDS, MAX_WORDS = 15, 450
RACE_TRAIN_LIMIT, MAX_PER_RACE_ARTICLE = 15000, 2
VAL_PER_SOURCE, TEST_PER_SOURCE = 300, 30
SOURCES = ("sciq", "race", "edugq")
MAX_LEN = 1024                       # prompt + answer tokens; longer examples are skipped, not truncated
PER_DEVICE_BATCH, GRAD_ACCUM = 1, 8  # unchanged from the first runs (effective batch = 8)
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Shared SFT baseline. Trials below change ONE factor at a time.
SFT_LORA_TRIALS = {
    "baseline": {"learning_rate": 2e-4, "r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "warmup_steps": 10, "num_train_epochs": 2},
    "trial1_lr": {"learning_rate": 1e-4, "r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "warmup_steps": 10, "num_train_epochs": 2},
    "trial2_dropout": {"learning_rate": 2e-4, "r": 16, "lora_alpha": 32, "lora_dropout": 0.10, "warmup_steps": 10, "num_train_epochs": 2},
    "trial3_capacity": {"learning_rate": 2e-4, "r": 32, "lora_alpha": 64, "lora_dropout": 0.05, "warmup_steps": 10, "num_train_epochs": 2},
}
# LoRA-only means continued causal-language-model training on passages (no question-generation
# instruction/answer pairs). This isolates the adapter method from the SFT objective.
LORA_ONLY_TRIALS = {
    "baseline": {"learning_rate": 2e-4, "r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "warmup_steps": 10, "num_train_epochs": 2},
    "trial1_lr": {"learning_rate": 1e-4, "r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "warmup_steps": 10, "num_train_epochs": 2},
    "trial2_dropout": {"learning_rate": 2e-4, "r": 16, "lora_alpha": 32, "lora_dropout": 0.10, "warmup_steps": 10, "num_train_epochs": 2},
    "trial3_capacity": {"learning_rate": 2e-4, "r": 32, "lora_alpha": 64, "lora_dropout": 0.05, "warmup_steps": 10, "num_train_epochs": 2},
}
# FIX: regex instead of substring matching. The old check `"a and b" in text` also rejected valid
# options such as "Africa and Brazil".
BAD_OPTION = re.compile(r"\b(all|none|both) of (the )?(above|them)\b|\b[ABCD] (and|or) [ABCD]\b", re.IGNORECASE)


def seed_everything(seed=SEED):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\u00a0", " ").replace("<hl>", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?)%])", r"\1", text)
    text = re.sub(r"([(])\s+", r"\1", text)
    return text.strip()


def digest(text: str) -> str:
    return hashlib.sha256(clean_text(text).lower().encode("utf-8")).hexdigest()[:20]


REJECTED = defaultdict(int)          # (source, reason) -> count, shown in preprocessing_report.json


def reject_reason(passage, question, options, answer_idx):
    """None if the example is usable, otherwise the reason it is removed."""
    passage, question = clean_text(passage), clean_text(question)
    if not passage: return "empty_passage"
    if not question: return "empty_question"
    if not isinstance(options, (list, tuple)) or len(options) != 4: return "not_exactly_four_options"
    words = len(passage.split())
    if words < MIN_WORDS: return "passage_too_short"
    if words > MAX_WORDS: return "passage_too_long"
    opts = [clean_text(x) for x in options]
    if any(not x for x in opts): return "empty_option"
    if len({x.lower() for x in opts}) != 4: return "duplicate_options"
    if isinstance(answer_idx, bool) or not isinstance(answer_idx, int) or not 0 <= answer_idx < 4: return "invalid_answer_index"
    if any(BAD_OPTION.search(x) for x in opts): return "position_dependent_option"
    return None


def valid_record(passage, question, options, answer_idx):
    return reject_reason(passage, question, options, answer_idx) is None


def make_record(source, passage, question, options, answer_idx, group_id, source_id=""):
    passage, question = clean_text(passage), clean_text(question)
    options = [clean_text(x) for x in options] if isinstance(options, (list, tuple)) else options
    reason = reject_reason(passage, question, options, answer_idx)
    if reason:
        REJECTED[(source, reason)] += 1
        return None
    answer = options[answer_idx]
    distractors = [x for i, x in enumerate(options) if i != answer_idx]
    pkey = digest(passage)
    group = str(group_id or pkey)
    # Prompt unchanged: adapters trained earlier used exactly this text.
    prompt = ("Generate one clear multiple-choice question using only the information in the passage. "
              "Return valid JSON with keys: question, answer, distractors. The answer must be supported "
              "by the passage and distractors must be plausible but incorrect.\n\nPASSAGE:\n" + passage)
    completion = json.dumps({"question": question, "answer": answer, "distractors": distractors}, ensure_ascii=False)
    return {"id": f"{source}-{pkey}-{digest(question)}", "source": source, "group_id": group,
            "passage_key": pkey, "passage": passage, "question": question, "answer": answer,
            "distractors": distractors, "prompt": prompt, "completion": completion,
            "source_id": str(source_id)}


def load_sciq():
    from datasets import load_dataset
    ds = load_dataset("allenai/sciq")
    result = {"train": [], "validation": [], "test": []}
    for split in result:
        if split not in ds:
            continue
        for i, row in enumerate(ds[split]):
            options = [row.get("correct_answer"), row.get("distractor1"), row.get("distractor2"), row.get("distractor3")]
            rec = make_record("sciq", row.get("support"), row.get("question"), options, 0,
                              digest(row.get("support", "")), f"{split}-{i}")
            if rec: result[split].append(rec)
    return result


def load_race():
    from datasets import load_dataset
    ds = load_dataset("ehovy/race", "all")
    result = {"train": [], "validation": [], "test": []}
    for split in result:
        if split not in ds: continue
        for row in ds[split]:
            options = row.get("options") or []
            letter = clean_text(row.get("answer", "")).upper()
            if letter not in ("A", "B", "C", "D"): continue      # FIX: "" used to pass `"" in "ABCD"` and became "A"
            article = clean_text(row.get("article", ""))
            rec = make_record("race", article, row.get("question"), list(options), "ABCD".index(letter),
                              digest(article), row.get("example_id", ""))
            if rec: result[split].append(rec)
    return result


def fetch_eduqg_file(name):
    """Download once into data/raw/eduqg/ (compute nodes without internet: put the 2 files there by hand)."""
    base = "https://raw.githubusercontent.com/hadifar/question-generation/main/raw_data/"
    path = EDUQG_DIR / name
    if not path.exists():
        import requests
        try:
            response = requests.get(base + name, timeout=120)
            response.raise_for_status()
        except Exception as exc:
            raise RuntimeError(f"Could not download {name} ({exc}).\nDownload it manually from {base + name} "
                               f"and put it in {EDUQG_DIR}") from exc
        EDUQG_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
    return json.loads(path.read_text(encoding="utf-8"))


def load_edugq():
    train_raw = fetch_eduqg_file("qg_train_v0.json")
    valid_raw = fetch_eduqg_file("qg_valid_v0.json")
    # Hold out whole chapters from the training file for validation.
    chapter_ids = sorted({f"{x.get('bname','')}-{x.get('chapter','')}" for x in train_raw})
    rng = random.Random(SEED); rng.shuffle(chapter_ids)
    heldout = set(chapter_ids[:max(1, int(len(chapter_ids) * 0.1))]) if chapter_ids else set()
    result = {"train": [], "validation": [], "test": []}

    # FIX: each row of the EduQG file is a CHAPTER; its questions are in row["questions"].
    # The old code read row["question"] directly, found nothing and silently returned 0 examples.
    def convert(chapters, target, heldout_filter=None):
        for ci, chapter in enumerate(chapters):
            cid = f"{chapter.get('bname','')}-{chapter.get('chapter','')}"
            if heldout_filter is not None and ((cid in heldout) != heldout_filter):
                continue
            for qi, item in enumerate(chapter.get("questions") or []):
                q = item.get("question") or {}; a = item.get("answer") or {}
                options = q.get("question_choices") or []
                try: ans = int(a.get("ans_choice"))            # use ans_choice (index), not ans_text
                except (TypeError, ValueError): continue
                rec = make_record("edugq", item.get("hl_context"), q.get("question_text"), options,
                                  ans, cid, f"{target}-{ci}-{qi}")
                if rec: result[target].append(rec)
    convert(train_raw, "train", False)
    convert(train_raw, "validation", True)
    convert(valid_raw, "test", None)
    return result


def check_eduqg():
    """Quick EduQG check: counts per split, why examples are removed, one example. No GPU, no HuggingFace."""
    REJECTED.clear()
    ed = load_edugq()
    print("EduQG examples kept:", {k: len(v) for k, v in ed.items()})
    print("EduQG examples removed (reason: count):")
    for (src, why), n in sorted(REJECTED.items()):
        print(f"   {why}: {n}")
    r = ed["train"][0]
    print("\nexample passage:", r["passage"][:150], "...\nquestion:", r["question"], "\nanswer:", r["answer"],
          "\ndistractors:", r["distractors"])
    assert all(len(v) > 0 for v in ed.values()), "EduQG produced an empty split"


def deduplicate(records):
    seen, out = set(), []
    for r in records:
        key = (r["passage_key"], clean_text(r["question"]).lower(),
               tuple(sorted([r["answer"].lower()] + [x.lower() for x in r["distractors"]])))
        if key not in seen:
            seen.add(key); out.append(r)
    return out


def sample_per_source(records, n, one_per_passage=False):
    rng = random.Random(SEED)
    by_source = defaultdict(list)
    for r in records: by_source[r["source"]].append(r)
    out = []
    for source, rows in by_source.items():
        rng.shuffle(rows); used, taken = set(), 0
        for r in rows:
            if one_per_passage and r["passage_key"] in used: continue
            out.append(r); used.add(r["passage_key"]); taken += 1
            if taken >= n: break
    return out


def subset_rows(rows, max_n=None, fraction=None, balanced=True, seed=SEED):
    """Deterministic small subset for fast experiments (same seed -> same subset for every trial).

    balanced=True : equal share per dataset (SciQ / RACE / EduQG); a dataset with fewer examples gives all it has
                    and the rest is shared by the others. Without it RACE (15k) would be ~60% of the data and
                    EduQG (2.2k, the closest to lecture material) almost absent.
    balanced=False: random subset that keeps the original proportions.
    """
    if max_n is None and fraction is None:
        return list(rows)
    n = max_n if max_n is not None else max(1, int(len(rows) * fraction))
    n = min(n, len(rows))
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in rows: by[r["source"]].append(r)
    for v in by.values(): rng.shuffle(v)
    if not balanced:
        pool = list(rows); rng.shuffle(pool)
        return pool[:n]
    take, remaining, active = {s: 0 for s in by}, n, sorted(by)
    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        for s in list(active):
            add = min(share, len(by[s]) - take[s], remaining)
            take[s] += add; remaining -= add
            if take[s] >= len(by[s]): active.remove(s)
            if remaining == 0: break
    out = [r for s in sorted(by) for r in by[s][:take[s]]]
    rng.shuffle(out)
    return out


def describe_subset(name, rows):
    counts = defaultdict(int)
    for r in rows: counts[r["source"]] += 1
    print(f"{name}: {len(rows)} examples " + str(dict(sorted(counts.items()))))
    return dict(sorted(counts.items()))


def save_run_config(output, trial, cfg, train_rows, val_rows):
    Path(output).mkdir(parents=True, exist_ok=True)
    info = {"trial": trial, "hyperparameters": cfg, "n_train": len(train_rows), "n_val": len(val_rows),
            "train_by_source": describe_subset("train", train_rows), "val_by_source": describe_subset("validation", val_rows)}
    (Path(output) / "run_config.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f: return [json.loads(line) for line in f if line.strip()]


def preprocess(test_per_source: int = TEST_PER_SOURCE):
    seed_everything()
    REJECTED.clear()
    print("Loading SciQ, RACE and EduQG...")
    sources = {"sciq": load_sciq(), "race": load_race(), "edugq": load_edugq()}
    # FIX: stop immediately if a dataset produced nothing (this is what hid the EduQG bug).
    for name, splits in sources.items():
        print(f"  {name}: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))
        if not splits["train"] or not splits["test"]:
            raise RuntimeError(f"Dataset '{name}' produced no train/test examples: check its loader.")
    # Official test records are never moved into train/validation.
    test = []
    for splits in sources.values(): test.extend(splits["test"])
    test = sample_per_source(deduplicate(test), test_per_source, one_per_passage=True)
    test_groups = {r["group_id"] for r in test}; test_passages = {r["passage_key"] for r in test}
    # Validation candidates from official validation and EduQG chapter holdout.
    val_candidates = []
    for splits in sources.values(): val_candidates.extend(splits["validation"])
    val_candidates = [r for r in deduplicate(val_candidates)
                      if r["group_id"] not in test_groups and r["passage_key"] not in test_passages]
    # Reserve every eligible validation group/passage before sampling, avoiding leakage.
    val_groups = {r["group_id"] for r in val_candidates}
    val_passages = {r["passage_key"] for r in val_candidates}
    val = sample_per_source(val_candidates, VAL_PER_SOURCE, one_per_passage=False)
    train = []
    for splits in sources.values(): train.extend(splits["train"])
    train = deduplicate([r for r in train if r["group_id"] not in test_groups | val_groups
                         and r["passage_key"] not in test_passages | val_passages])
    # Cap RACE train examples and max two per article.
    race, other = [r for r in train if r["source"] == "race"], [r for r in train if r["source"] != "race"]
    rng = random.Random(SEED); rng.shuffle(race)
    counts, capped = defaultdict(int), []
    for r in race:
        if counts[r["group_id"]] < MAX_PER_RACE_ARTICLE and len(capped) < RACE_TRAIN_LIMIT:
            capped.append(r); counts[r["group_id"]] += 1
    train = other + capped
    random.Random(SEED).shuffle(train)
    # Cross-check leakage by both passage and group.
    for left, right, label in [(train, val, "train/val"), (train, test, "train/test"), (val, test, "val/test")]:
        assert not ({x["group_id"] for x in left} & {x["group_id"] for x in right}), f"Group leakage: {label}"
        assert not ({x["passage_key"] for x in left} & {x["passage_key"] for x in right}), f"Passage leakage: {label}"
    for name, rows in [("train", train), ("validation", val), ("test", test)]:
        write_jsonl(DATA_DIR / f"{name}.jsonl", rows)
    by_source = {s: {k: sum(r["source"] == s for r in rows) for k, rows in [("train", train), ("validation", val), ("test", test)]} for s in SOURCES}
    for s, c in by_source.items():
        if not all(c.values()): raise RuntimeError(f"Source '{s}' is missing from a split: {c}")
    report = {"model": MODEL_NAME, "counts": {k: len(v) for k, v in {"train": train, "validation": val, "test": test}.items()},
              "by_source": by_source,
              "removed": {f"{src}::{why}": n for (src, why), n in sorted(REJECTED.items())},
              "notes": ["Test references are held out and are not used as training prompts.",
                        "Validation is used for checkpoint selection and hyperparameter comparison.",
                        "EduQG valid file is treated as held-out test data; chapter holdout from EduQG train is validation."]}
    (DATA_DIR / "preprocessing_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def require_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a CUDA GPU for this configuration. Use a GPU runtime and install a matching PyTorch build.")
    return torch


def load_base_model():
    """Load Qwen in fp16/bf16, NOT 4-bit. This is ordinary LoRA, not QLoRA."""
    import torch
    from transformers import AutoModelForCausalLM
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype, device_map="auto", trust_remote_code=True)
    except TypeError:                                   # older transformers: torch_dtype
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype, device_map="auto", trust_remote_code=True)
    model.config.use_cache = False
    # FIX: gradient checkpointing + frozen embeddings needs input grads, otherwise the LoRA layers may get no gradient.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    return model


def build_sft_text(row, tokenizer):
    messages = [{"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["completion"]}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)


class CompletionOnlyCollator:
    def __init__(self, tok, torch_module): self.tok, self.torch = tok, torch_module
    def __call__(self, features):
        labels = [f["labels"] for f in features]
        inputs = [{k: v for k, v in f.items() if k != "labels"} for f in features]
        batch = self.tok.pad(inputs, padding=True, return_tensors="pt")
        max_len = batch["input_ids"].shape[1]
        padded = [x + [-100] * (max_len - len(x)) for x in labels]
        batch["labels"] = self.torch.tensor(padded, dtype=self.torch.long)
        return batch


def encode_sft_rows(rows, tokenizer, Dataset, max_len: int = MAX_LEN):
    """prompt tokens (masked with -100) + completion tokens (trained on) + <|im_end|>.

    FIX: the prompt length used to be len(apply_chat_template(tokenize=True)). In recent transformers
    versions that call returns a dict (len = 2), so only 2 tokens were masked and the model was trained
    on the whole passage as well (loss ~1.4 instead of ~0.9). Here the prompt is rendered as TEXT and
    tokenised separately, so the mask is correct for every transformers version.
    """
    ids_list, labels_list, skipped = [], [], 0
    for r in rows:
        prompt_text = tokenizer.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                        tokenize=False, add_generation_prompt=True, enable_thinking=False)
        p = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        c = tokenizer(r["completion"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        if len(p) + len(c) > max_len:
            skipped += 1
            continue
        ids_list.append(p + c); labels_list.append([-100] * len(p) + c)
    print(f"encoded {len(ids_list)} examples, skipped {skipped} longer than {max_len} tokens")
    return Dataset.from_dict({"input_ids": ids_list, "attention_mask": [[1] * len(x) for x in ids_list],
                              "labels": labels_list})


def check_labels(ds, tokenizer, n: int = 2):
    """Print what the model is trained on and stop if the prompt is not masked."""
    for i in range(min(n, len(ds))):
        ids, labels = ds[i]["input_ids"], ds[i]["labels"]
        target = [t for t, l in zip(ids, labels) if l != -100]
        text = tokenizer.decode(target)
        print(f"example {i}: {len(ids)} tokens, loss on {len(target)} ({len(target) / len(ids):.0%}) -> {text[:120]!r}")
        assert len(target) < len(ids) - 15 and text.lstrip().startswith("{"), \
            "Label masking is wrong: the prompt is being trained on."


def check_masking(n_rows: int = 3):
    """Tokenizer-only check (no GPU): run it before any SFT training."""
    from transformers import AutoTokenizer
    from datasets import Dataset
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    ds = encode_sft_rows(read_jsonl(DATA_DIR / "train.jsonl")[:n_rows], tok, Dataset)
    check_labels(ds, tok, n_rows)
    print("masking OK")


def training_args(output, cfg, torch, eval_strategy="epoch"):
    from transformers import TrainingArguments
    return TrainingArguments(
        output_dir=str(output), num_train_epochs=cfg["num_train_epochs"],
        learning_rate=cfg["learning_rate"], per_device_train_batch_size=PER_DEVICE_BATCH,
        per_device_eval_batch_size=PER_DEVICE_BATCH, gradient_accumulation_steps=GRAD_ACCUM,
        warmup_steps=cfg.get("warmup_steps", 10), logging_steps=10,
        eval_strategy=eval_strategy, save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False, save_total_limit=2,
        fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_torch", report_to="none", gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )


def save_training_log(trainer, output):
    """Keeps the loss history next to the adapter (used by plot_training_curves)."""
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output) / "training_log.json").write_text(json.dumps(trainer.state.log_history, indent=1), encoding="utf-8")


def plot_training_curves(*trial_dirs):
    """plot_training_curves('outputs/sft_lora/baseline', 'outputs/sft_lora/trial2_dropout')"""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for d in trial_dirs:
        d = Path(d)
        log_file = d / "training_log.json"
        if log_file.exists():
            log = json.loads(log_file.read_text())
        else:                                             # fall back to the newest Trainer checkpoint
            cks = sorted(d.glob("checkpoint-*/trainer_state.json"), key=lambda p: int(p.parent.name.split("-")[1]))
            if not cks:
                print(f"no loss history in {d}"); continue
            log = json.loads(cks[-1].read_text())["log_history"]
        name = "/".join(d.parts[-2:])
        tr = [(x["step"], x["loss"]) for x in log if "loss" in x]
        ev = [(x["step"], x["eval_loss"]) for x in log if "eval_loss" in x]
        if tr: ax[0].plot(*zip(*tr), label=name)
        if ev: ax[1].plot(*zip(*ev), "o-", label=name)
    ax[0].set_title("training loss"); ax[1].set_title("validation loss")
    for a in ax: a.set_xlabel("step"); a.legend(); a.grid(alpha=.3)
    plt.tight_layout(); plt.show()


def _lora_config(cfg):
    from peft import LoraConfig
    return LoraConfig(r=cfg["r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                      bias="none", task_type="CAUSAL_LM", target_modules=LORA_TARGETS)


def run_sft_lora(trial: str, max_train=None, train_fraction=None, max_val=None, epochs=None, balanced=True):
    """max_train / train_fraction: use only a (balanced) part of train.jsonl; max_val: part of validation.jsonl;
    epochs: overrides the trial value (1 is enough for fast trials)."""
    torch = require_gpu()
    from transformers import AutoTokenizer, Trainer
    from datasets import Dataset
    from peft import get_peft_model
    cfg = dict(SFT_LORA_TRIALS[trial])
    if epochs: cfg["num_train_epochs"] = epochs
    train_rows = subset_rows(read_jsonl(DATA_DIR / "train.jsonl"), max_train, train_fraction, balanced)
    val_rows = subset_rows(read_jsonl(DATA_DIR / "validation.jsonl"), max_val, None, True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train_ds = encode_sft_rows(train_rows, tokenizer, Dataset)
    val_ds = encode_sft_rows(val_rows, tokenizer, Dataset)
    check_labels(train_ds, tokenizer)                                  # FIX: stops if the prompt is not masked
    model = get_peft_model(load_base_model(), _lora_config(cfg))
    model.print_trainable_parameters()
    output = OUT_DIR / "sft_lora" / trial
    save_run_config(output, trial, cfg, train_rows, val_rows)
    trainer = Trainer(model=model, args=training_args(output, cfg, torch),
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=CompletionOnlyCollator(tokenizer, torch))
    trainer.train()
    save_training_log(trainer, output)
    adapter_dir = output / "best_adapter"
    trainer.model.save_pretrained(adapter_dir); tokenizer.save_pretrained(adapter_dir)
    print(f"Saved full-precision SFT+LoRA adapter: {adapter_dir}")


def run_lora_only(trial: str, resume: bool = False, max_train=None, train_fraction=None, max_val=None,
                  epochs=None, balanced=True):
    """LoRA adapter trained with causal LM on passage text only; no MCQ SFT targets (same subset options as SFT)."""
    torch = require_gpu()
    from transformers import AutoTokenizer, Trainer
    from datasets import Dataset
    from peft import get_peft_model
    cfg = dict(LORA_ONLY_TRIALS[trial])
    if epochs: cfg["num_train_epochs"] = epochs
    train_rows = subset_rows(read_jsonl(DATA_DIR / "train.jsonl"), max_train, train_fraction, balanced)
    val_rows = subset_rows(read_jsonl(DATA_DIR / "validation.jsonl"), max_val, None, True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = get_peft_model(load_base_model(), _lora_config(cfg))
    def encode(rows):
        texts = [r["passage"] + tokenizer.eos_token for r in rows]
        toks = tokenizer(texts, truncation=True, max_length=MAX_LEN)
        toks["labels"] = [ids.copy() for ids in toks["input_ids"]]
        return Dataset.from_dict(toks)
    train_ds, val_ds = encode(train_rows), encode(val_rows)
    output = OUT_DIR / "lora_only" / trial
    save_run_config(output, trial, cfg, train_rows, val_rows)
    trainer = Trainer(model=model, args=training_args(output, cfg, torch),
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=CompletionOnlyCollator(tokenizer, torch))
    # FIX: the checkpoint path was hard-coded (checkpoint-3047) and crashed any new trial.
    trainer.train(resume_from_checkpoint=True if resume else None)
    save_training_log(trainer, output)
    adapter_dir = output / "best_adapter"
    trainer.model.save_pretrained(adapter_dir); tokenizer.save_pretrained(adapter_dir)
    print(f"Saved LoRA-only (continued LM) adapter: {adapter_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("preprocess"); pp.add_argument("--test-per-source", type=int, default=TEST_PER_SOURCE)
    sub.add_parser("check-masking")
    sub.add_parser("check-eduqg")
    def subset_args(sp):
        sp.add_argument("--max-train", type=int, help="use only N training examples (balanced over the datasets)")
        sp.add_argument("--train-fraction", type=float, help="or use this fraction of the training set, e.g. 0.1")
        sp.add_argument("--max-val", type=int, help="use only N validation examples (faster evaluation)")
        sp.add_argument("--epochs", type=int, help="override the number of epochs")
        sp.add_argument("--no-balance", action="store_true", help="keep the original dataset proportions")
    s = sub.add_parser("train-sft-lora"); s.add_argument("--trial", choices=SFT_LORA_TRIALS, default="baseline"); subset_args(s)
    lo = sub.add_parser("train-lora-only"); lo.add_argument("--trial", choices=LORA_ONLY_TRIALS, default="baseline"); lo.add_argument("--resume", action="store_true"); subset_args(lo)
    args = p.parse_args()
    if args.cmd == "preprocess": preprocess(args.test_per_source)
    elif args.cmd == "check-masking": check_masking()
    elif args.cmd == "check-eduqg": check_eduqg()
    elif args.cmd == "train-lora-only":
        run_lora_only(args.trial, args.resume, args.max_train, args.train_fraction, args.max_val, args.epochs, not args.no_balance)
    elif args.cmd == "train-sft-lora":
        run_sft_lora(args.trial, args.max_train, args.train_fraction, args.max_val, args.epochs, not args.no_balance)

if __name__ == "__main__": main()
