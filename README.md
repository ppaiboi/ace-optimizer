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

## Status

Early prototype. Design under discussion (see `docs/rfc/0001-dspy-ace-optimizer.md`).

## License

Apache-2.0. Portions derived from the Apache-2.0 ACE reference implementation —
see `NOTICE`.
