# ace-optimizer

**Agentic Context Engineering (ACE)** — a framework-agnostic optimizer that
evolves a *playbook* of reusable strategies from execution feedback, instead of
rewriting a single instruction. Works with any LLM app; ships with a DSPy
teleprompter and an Anthropic client wrapper.

Based on Zhang et al., *"Agentic Context Engineering: Evolving Contexts for
Self-Improving Language Models"* ([arXiv:2510.04618](https://arxiv.org/abs/2510.04618),
ICLR 2026). Complementary to GEPA, not a replacement.

## The core idea: split the hot path from the learning path

The failure mode of every "self-improving prompt" library is learning
*synchronously inside the request*. ACE doesn't.

```
HOT PATH (µs, no LLM)            LEARNING PATH (async, batched, gated)
────────────────────            ─────────────────────────────────────
ace.augment(system) ─► prompt   observe() ─► Reflector ─► Curator (Δ: ADD/EDIT/…)
your model call    ─► response        │                        │
                                  gate: validate Δ on holdout ──┘
                                        │ pass → playbook v(n+1)  (versioned)
                                        │ fail → quarantine + reason
```

The inference path does one thing — string concatenation. If the learning
subsystem is down, your app doesn't notice.

### Does it need gold labels?

**No.** The Reflector reasons over a *feedback signal*, and labels are just one
kind. Batteries included: `GroundTruth` (you have labels), `LLMJudge` (you
don't), `ImplicitUser` (product telemetry: accepted/edited/rejected),
`ExecutionResult` (tests passed / API 200'd).

## Three integration levels

```python
# Level 1 — drop-in client wrapper (Anthropic): playbook injected, same interface.
from ace import ACE, SQLitePlaybookStore, LLMReflector, LLMCurator
from ace.integrations.anthropic import wrap, completion_fn
import anthropic

client = anthropic.Anthropic()
llm = completion_fn(client, model="claude-haiku-4-5")
ace = ACE(store=SQLitePlaybookStore("playbook.db"),
          reflector=LLMReflector(llm), curator=LLMCurator(llm))
client = wrap(client, ace)              # messages.create now injects the playbook
```

```python
# Level 2 — explicit control (any stack).
system = ace.augment(base_system_prompt)          # hot path: pure string op
result = my_pipeline(system, user_input)
ace.observe(input=user_input, output=result, signal=ImplicitUser())  # queue a trace
ace.learn(holdout=holdout, evaluate=eval_fn)      # offline: reflect→curate→gate→promote
```

```python
# Level 3 — DSPy teleprompter (batch optimization over a dataset).
import dspy
from ace.integrations.dspy import ACE as DspyACE   # (from dspy_ace import ACE still works)

optimized = DspyACE(metric=my_metric, reflection_lm=dspy.LM("openai/gpt-4.1")).compile(
    program, trainset=trainset, valset=valset,
)
```

## What's built

| Module | Role |
|---|---|
| `ace.playbook` / `ace.merge` | Pure, deterministic delta core (ADD/BUMP/EDIT/DELETE/MERGE + grow-and-refine). No I/O, no LLM. |
| `ace.store` | **Event-sourced** delta log (in-memory / JSON / SQLite): playbook@v*n* = fold of the first *n* commits → audit, rollback, time-travel. |
| `ace.signals` | `Signal` protocol + the four label-free-friendly defaults. |
| `ace.reflect` / `ace.curate` | Reflector / Curator protocols + LLM-backed defaults (model injected as a `str→str` callable). |
| `ace.gate` | Promotion policy: validate a candidate on a holdout before it goes live; quarantine rejects. *CD for prompts.* |
| `ace.facade.ACE` | The hot-path/learning-path facade above. |
| `ace.integrations.{anthropic,dspy}` | Client wrapper + DSPy teleprompter. |

Everything that touches an LLM sits behind a protocol, so the whole suite runs
with fakes and zero API calls.

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
