# ace-optimizer

**Agentic Context Engineering (ACE)** for [DSPy](https://github.com/stanfordnlp/dspy) —
an optimizer that evolves a *playbook* of reusable strategies from execution
feedback, instead of rewriting a single instruction.

Based on Zhang et al., *"Agentic Context Engineering: Evolving Contexts for
Self-Improving Language Models"* ([arXiv:2510.04618](https://arxiv.org/abs/2510.04618),
ICLR 2026). Complementary to GEPA, not a replacement.

## Two layers

| Package | What it is | Deps |
|---|---|---|
| `ace` | Standalone optimizer engine + deterministic playbook core (delta-merge, grow-and-refine). Pure Python. | none |
| `dspy_ace` | The `dspy.ACE` teleprompter (intended for upstreaming into `dspy/teleprompt/ace/`). | `dspy` |

The split mirrors GEPA's packaging: a standalone engine (`gepa`) + a thin native
teleprompter inside dspy.

## Usage

```python
import dspy
from dspy_ace import ACE

optimized = ACE(metric=my_metric, reflection_lm=dspy.LM("openai/gpt-4.1")).compile(
    program, trainset=trainset, valset=valset,
)
# optimized.ace_playbook -> the learned Playbook
```

For a paper-faithful run (per-sample stepping, multi-round reflection, periodic
validation), pass `faithful=True`.

## Playbook: the artifact

Line-structured and interop-compatible with the reference implementation:

```
## FORMULAS_AND_CALCULATIONS
[calc-00042] helpful=3 harmful=1 :: Convert percentages to decimals before applying the formula.
```

Each bullet carries an id and helpful/harmful counters. The deterministic core
(`ace.merge`) applies ADD / BUMP / EDIT / DELETE / MERGE deltas and does
embedding-based grow-and-refine — all pure functions, fully unit-tested.

## Reproduction

`experiments/reproduce.py` runs baseline vs GEPA vs ACE on the paper's own
finance benchmarks (FiNER, Formula). See `docs/rfc/` for the design proposal.

**One-click:**

```bash
# 1. fetch the paper's data (16 MB, stdlib only, pinned upstream commit)
python experiments/fetch_data.py

# 2. exact paper reproduction — DeepSeek-V3.1 (the paper's model, all roles)
PYTHONPATH=src python experiments/reproduce.py \
    --task finer --model bedrock/deepseek.v3-v1:0 \
    --train 1000 --val 500 --test 441 \
    --rounds 3 --curator-freq 1 --eval-steps 100 --threads 8
```

For an **exact** reproduction use `bedrock/deepseek.v3-v1:0` (DeepSeek-V3.1 on AWS
Bedrock, region `us-west-2`) — the model the ACE paper uses for the Generator,
Reflector, and Curator. Any other model (e.g. `openai/gpt-4o-mini`) reproduces the
paper's *direction* (ACE ≥ GEPA ≥ baseline) but not its absolute numbers.

**Bring your own data:** drop JSONL files into `experiments/data/` — one
`{"context": "...", "target": "..."}` object per line — and point `TASKS` in
`reproduce.py` at them. See `experiments/fetch_data.py` for the format and examples.

## Status

Early prototype. Design under discussion (see `docs/rfc/0001-dspy-ace-optimizer.md`).

## License

Apache-2.0. Portions derived from the Apache-2.0 ACE reference implementation —
see `NOTICE`.
