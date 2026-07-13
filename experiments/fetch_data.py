"""One-click download of the ACE paper's finance benchmark data.

Pulls the exact FiNER + Formula splits used by reproduce.py from the upstream
reference repo (ace-agent/ace) at a pinned commit, into experiments/data/.
Stdlib only — no pip install, no git, no API key.

    python experiments/fetch_data.py            # fetch missing files
    python experiments/fetch_data.py --force     # re-download everything

Then, for an exact paper reproduction, run against DeepSeek-V3.1 (the model the
ACE paper uses for all three roles) with the paper's FiNER config:

    PYTHONPATH=src python experiments/reproduce.py \
        --task finer --model bedrock/deepseek.v3-v1:0 \
        --train 1000 --val 500 --test 441 \
        --rounds 3 --curator-freq 1 --eval-steps 100 --threads 8

(bedrock/deepseek.v3-v1:0 is DeepSeek-V3.1 on AWS Bedrock, region us-west-2. Any
other model reproduces the *direction* but not the paper's absolute numbers.)

DATA FORMAT — to bring your own benchmark, drop JSONL files into experiments/data/
and point reproduce.py's TASKS entry at them. One JSON object per line, two keys:

    {"context": "<the full prompt/question>", "target": "<the gold answer>"}

reproduce.py maps context -> question (the model input) and target -> answer (the
gold label). Concrete examples:

  Formula (numeric, exact-match reward):
    {"context": "Use formula Operating Margin ... Answer with 2 decimal places ...",
     "target": "15.0"}

  FiNER (comma-separated GAAP tags, fraction-correct reward):
    {"context": "You are XBRL expert. Here is a list of US GAAP tags options: ...",
     "target": "InterestExpense,GoodwillImpairmentLoss,SaleOfStockPricePerShare"}

The scorer is chosen by --task (formula = exact float match; finer = fraction of
comma-aligned tags correct), so match your target format to the task you pick.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

# Pinned so the data never drifts out from under the reproduction.
COMMIT = "bcb7cea0504afad6f55fec4845dd4864c9f9eee7"
BASE = f"https://raw.githubusercontent.com/ace-agent/ace/{COMMIT}/eval/finance/data"
DEST = Path(__file__).resolve().parent / "data"

# file -> expected byte size (a cheap integrity check against a truncated download)
FILES = {
    "finer_train_batched_1000_samples.jsonl": 8028162,
    "finer_val_batched_500_samples.jsonl": 4022980,
    "finer_test_subset_006_seed42.jsonl": 3572008,
    "formula_train_subset_500.jsonl": 266931,
    "formula_val_subset_300.jsonl": 159618,
    "formula_test.jsonl": 106566,
}


def fetch(name: str, size: int, force: bool) -> None:
    out = DEST / name
    if out.exists() and out.stat().st_size == size and not force:
        print(f"  ok   {name} (cached)")
        return
    url = f"{BASE}/{name}"
    print(f"  get  {name} ...", end="", flush=True)
    tmp = out.with_suffix(out.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 (pinned raw.githubusercontent URL)
    got = tmp.stat().st_size
    if got != size:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"\n  size mismatch for {name}: got {got}, want {size}")
    tmp.replace(out)
    print(f" {got:,} bytes")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    args = ap.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    print(f"ACE benchmark data -> {DEST}")
    print(f"source: ace-agent/ace @ {COMMIT[:12]}")
    for name, size in FILES.items():
        fetch(name, size, args.force)
    total = sum(f.stat().st_size for f in DEST.glob("*.jsonl"))
    print(f"done — {total / 1e6:.1f} MB in {DEST}")


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError as e:  # pragma: no cover - network failure path
        print(f"\ndownload failed: {e}\n(check your connection; the URL is a pinned "
              "raw.githubusercontent.com path)", file=sys.stderr)
        raise SystemExit(1) from e
