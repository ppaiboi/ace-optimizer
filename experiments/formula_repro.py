"""Reproduce the ACE paper's Formula (finance) result with dspy.ACE.

Uses the paper's own bundled data (vendor/ace-upstream/eval/finance/data/).
Compares a baseline dspy program against the same program optimized by
dspy.ACE, on held-out test examples, and reports the accuracy delta — the
paper reports ~+18% on Formula.

Run (needs a real LM configured via env):
    export ANTHROPIC_API_KEY=...        # or OPENAI_API_KEY
    PYTHONPATH=src ./.venv/bin/python experiments/formula_repro.py \
        --model anthropic/claude-haiku-4-5-20251001 --train 40 --test 60

No key? It prints setup instructions and exits (no silent fake numbers).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import dspy

# make src importable without install
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dspy_ace import ACE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "vendor/ace-upstream/eval/finance/data"


def _load_dotenv() -> None:
    """Minimal .env loader (no dependency) so runs are self-contained."""
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def load(name: str, limit: int) -> list[dspy.Example]:
    rows = []
    with open(DATA / name) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            rows.append(
                dspy.Example(question=d["context"], answer=str(d["target"]).strip())
                .with_inputs("question")
            )
            if len(rows) >= limit:
                break
    return rows


_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def numeric_match(example, pred, *args) -> float:
    """Exact numeric match, tolerant of formatting (%, $, trailing text)."""

    def val(s):
        m = _NUM.search(str(s).replace(",", ""))
        return round(float(m.group()), 2) if m else None

    got = val(getattr(pred, "answer", ""))
    want = val(example.answer)
    return 1.0 if got is not None and got == want else 0.0


def evaluate(program, dataset) -> float:
    hits = 0
    for ex in dataset:
        try:
            hits += numeric_match(ex, program(**ex.inputs()))
        except Exception:
            pass
    return hits / len(dataset) if dataset else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("ACE_MODEL", "openai/gpt-4o-mini"))
    ap.add_argument("--train", type=int, default=40)
    ap.add_argument("--test", type=int, default=60)
    ap.add_argument("--minibatch", type=int, default=5)
    args = ap.parse_args()

    _load_dotenv()
    if not any(k in os.environ for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")):
        print(
            "No LM key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY, then:\n"
            "  PYTHONPATH=src ./.venv/bin/python experiments/formula_repro.py "
            "--model anthropic/claude-haiku-4-5-20251001"
        )
        return

    dspy.configure(lm=dspy.LM(args.model))

    train = load("formula_train_subset_500.jsonl", args.train)
    test = load("formula_test.jsonl", args.test)
    print(f"Formula reproduction | model={args.model} | train={len(train)} test={len(test)}")

    program = dspy.Predict("question -> answer")

    base_acc = evaluate(program, test)
    print(f"  baseline (no ACE):  {base_acc:.1%}")

    optimized = ACE(metric=numeric_match, minibatch_size=args.minibatch).compile(
        program, trainset=train
    )
    ace_acc = evaluate(optimized, test)
    print(f"  dspy.ACE:           {ace_acc:.1%}   (delta {ace_acc - base_acc:+.1%})")
    print(f"  learned playbook:   {len(optimized.ace_playbook.bullets)} bullets")
    print(optimized.ace_playbook.render())


if __name__ == "__main__":
    main()
