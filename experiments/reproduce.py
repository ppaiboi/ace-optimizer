"""Three-way reproduction on the ACE paper's finance benchmarks.

baseline vs GEPA vs ACE, on the paper's own data
(vendor/ace-upstream/eval/finance/data), reporting test accuracy + rollouts.

Tasks:
  * formula — numeric answer, exact match (binary reward)
  * finer   — comma-separated GAAP tags, fraction-correct (partial reward)

ACE here runs the fidelity-faithful config: a citing generator, embedding
dedup + size cap, and feedback-rich metric.

HONEST SCOPE: reproduces the paper's ranking/direction on real data. Absolute
numbers depend on the model. Results -> experiments/results/<task>_<model>.json.

    PYTHONPATH=src ./.venv/bin/python experiments/reproduce.py \
        --task finer --model bedrock/deepseek.v3-v1:0 \
        --train 500 --val 300 --test 441 --budget 800 --threads 8
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import dspy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# Prefer the fetch_data.py download dir; fall back to a local upstream clone.
_LOCAL = ROOT / "experiments/data"
_VENDOR = ROOT / "vendor/ace-upstream/eval/finance/data"
DATA = _LOCAL if _LOCAL.exists() else _VENDOR
RESULTS = ROOT / "experiments/results"

from dspy_ace import ACE  # noqa: E402

TASKS = {
    "formula": {
        "train": "formula_train_subset_500.jsonl",
        "val": "formula_val_subset_300.jsonl",
        "test": "formula_test.jsonl",
    },
    "finer": {
        "train": "finer_train_batched_1000_samples.jsonl",
        "val": "finer_val_batched_500_samples.jsonl",
        "test": "finer_test_subset_006_seed42.jsonl",
    },
}


def load_dotenv() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def load(name: str, limit: int) -> list[dspy.Example]:
    rows = []
    with open(DATA / name) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                rows.append(
                    dspy.Example(question=d["context"], answer=str(d["target"]).strip())
                    .with_inputs("question")
                )
            if len(rows) >= limit:
                break
    return rows


# ---- scorers -------------------------------------------------------------

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _num(s):
    m = _NUM.search(str(s).replace(",", ""))
    return round(float(m.group()), 2) if m else None


def formula_score(example, pred, *a) -> float:
    got, want = _num(getattr(pred, "answer", "")), _num(example.answer)
    return 1.0 if got is not None and got == want else 0.0


def finer_score(example, pred, *a) -> float:
    """Fraction of comma-aligned tag positions correct (paper's FiNER metric)."""
    p = [v.lower().strip() for v in str(getattr(pred, "answer", "")).split(",")]
    gold = [v.lower().strip() for v in example.answer.split(",")]
    if len(p) != len(gold):
        p = p[: len(gold)] if len(p) > len(gold) else p + [""] * (len(gold) - len(p))
    return (sum(a == b for a, b in zip(p, gold, strict=False)) / len(p)) if p else 0.0


def make_metrics(task: str):
    base = formula_score if task == "formula" else finer_score

    def feedback_metric(gold, pred, *a):
        s = base(gold, pred)
        exp = gold.answer if len(gold.answer) < 300 else gold.answer[:300] + "..."
        fb = "Correct." if s >= 0.999 else f"Score {s:.2f}. Expected: {exp}"
        return dspy.Prediction(score=s, feedback=fb)

    return base, feedback_metric


def evaluate(program, dataset, metric, threads: int) -> float:
    ev = dspy.Evaluate(
        devset=dataset, metric=metric, num_threads=threads,
        display_progress=False, display_table=0, provide_traceback=False,
    )
    return float(ev(program).score) / 100.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), default="finer")
    ap.add_argument("--model", default="bedrock/deepseek.v3-v1:0")
    ap.add_argument("--train", type=int, default=500)
    ap.add_argument("--val", type=int, default=300)
    ap.add_argument("--test", type=int, default=441)
    ap.add_argument("--budget", type=int, default=800)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-bullets", type=int, default=None)
    ap.add_argument("--methods", default="baseline,gepa,ace")
    # ACE loop knobs (the loop is always the paper-matching per-sample loop)
    ap.add_argument("--rounds", type=int, default=3, help="max reflection rounds")
    ap.add_argument("--curator-freq", type=int, default=1)
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--dedup", action="store_true", help="off by default, matching the paper")
    args = ap.parse_args()

    load_dotenv()
    files = TASKS[args.task]
    if not (DATA / files["test"]).exists():
        print(f"Benchmark data not found in {DATA}.\n"
              "Run this first:  python experiments/fetch_data.py")
        return
    if "bedrock" not in args.model and not any(
        k in os.environ for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "TOGETHERAI_API_KEY")
    ):
        print("No LM key found and not using bedrock.")
        return

    dspy.configure(lm=dspy.LM(args.model, max_tokens=4096))
    train = load(files["train"], args.train)
    val = load(files["val"], args.val)
    test = load(files["test"], args.test)
    base_metric, feedback_metric = make_metrics(args.task)
    methods = args.methods.split(",")
    program = dspy.Predict("question -> answer")

    # Embedding-based dedup for ACE (off by default, matching the paper).
    embed = None
    if args.dedup and "OPENAI_API_KEY" in os.environ:
        emb = dspy.Embedder("openai/text-embedding-3-small", batch_size=100)
        embed = lambda texts: [list(v) for v in emb(list(texts))]  # noqa: E731

    out = {
        "task": f"{args.task} (ACE paper / FinLoRA)", "model": args.model,
        "sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "budget": args.budget, "max_bullets": args.max_bullets,
        "dedup": embed is not None, "results": {},
    }
    print(f"{args.task} 3-way | {args.model} | train={len(train)} val={len(val)} "
          f"test={len(test)} budget={args.budget} dedup={embed is not None}")

    if "baseline" in methods:
        t = time.time()
        acc = evaluate(program, test, base_metric, args.threads)
        out["results"]["baseline"] = {"test_acc": acc, "secs": round(time.time() - t)}
        print(f"  baseline: {acc:.1%}")

    if "gepa" in methods:
        t = time.time()
        gepa = dspy.GEPA(
            metric=lambda g, p, trace=None, pn=None, pt=None: feedback_metric(g, p),
            max_metric_calls=args.budget,
            reflection_lm=dspy.LM(args.model, temperature=1.0, max_tokens=8000),
            num_threads=args.threads, track_stats=True,
        )
        gprog = gepa.compile(program, trainset=train, valset=val)
        acc = evaluate(gprog, test, base_metric, args.threads)
        calls = getattr(getattr(gprog, "detailed_results", None), "total_metric_calls", None)
        out["results"]["gepa"] = {"test_acc": acc, "metric_calls": calls,
                                  "secs": round(time.time() - t)}
        print(f"  GEPA:     {acc:.1%}  (rollouts={calls})")

    if "ace" in methods:
        t = time.time()
        ace = ACE(
            metric=feedback_metric,
            reflection_lm=dspy.LM(args.model, temperature=1.0, max_tokens=8000),
            num_threads=args.threads,
            embed=embed, max_bullets=args.max_bullets,
            max_num_rounds=args.rounds,
            curator_frequency=args.curator_freq, eval_steps=args.eval_steps,
        )
        aprog = ace.compile(program, trainset=train, valset=val)
        acc = evaluate(aprog, test, base_metric, args.threads)
        out["results"]["ace"] = {
            "test_acc": acc,
            "metric_calls": aprog.ace_result.total_metric_calls,
            "playbook_bullets": len(aprog.ace_playbook.bullets),
            "secs": round(time.time() - t),
        }
        out["ace_playbook"] = aprog.ace_playbook.render()
        print(f"  ACE:      {acc:.1%}  (rollouts={aprog.ace_result.total_metric_calls}, "
              f"{len(aprog.ace_playbook.bullets)} bullets)")

    RESULTS.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", args.model.lower())
    path = RESULTS / f"{args.task}_{slug}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
