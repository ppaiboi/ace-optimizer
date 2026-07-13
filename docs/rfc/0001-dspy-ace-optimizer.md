# RFC: `dspy.ACE` — an Agentic Context Engineering optimizer for DSPy

- **Status:** Draft (for maintainer triage)
- **Author:** Pierre Rappolt
- **Target:** `stanfordnlp/dspy` — new optimizer under `dspy/teleprompt/`, backed by a standalone `ace` package
- **Date:** 2026-07-13

---

## 1. The ask

Add a new optimizer, `dspy.ACE`, implementing **Agentic Context Engineering** (Zhang et al.,
[arXiv:2510.04618](https://arxiv.org/abs/2510.04618), ICLR 2026). It would ship the same way `dspy.GEPA`
did: a **thin native teleprompter** in `dspy/teleprompt/ace/` that wraps a **standalone `ace` package**
(the optimizer engine), with `ace` as an optional dependency.

Proposed surface — identical ergonomics to every other DSPy optimizer:

```python
optimized = dspy.ACE(metric=my_metric, reflection_lm=dspy.LM("openai/gpt-4.1")).compile(
    program, trainset=trainset, valset=valset,
)
```

I have the engine and the DSPy wrapper scoped against the GEPA precedent (§4–5) and am ready to build.
**This RFC is to confirm interest, agree on the API surface, and avoid duplicating in-flight work before
I open the PR.**

## 2. Motivation

**What ACE does.** Instead of optimizing a single concise instruction (GEPA, MIPROv2), ACE evolves a
**playbook**: a growing, itemized collection of strategy bullets, each with metadata (a stable id and
helpful/harmful counters). It updates the playbook through a **Generator → Reflector → Curator** loop
that emits small **delta** entries and merges them **deterministically** (non-LLM), then runs a
**grow-and-refine** dedup/prune pass. This deliberately fights the two failure modes the paper names in
prior context optimizers: **brevity bias** (summarization drops domain detail) and **context collapse**
(iterative monolithic rewriting erodes accumulated knowledge).

**Reported results** (per the paper — numbers to be re-verified against the PDF tables before posting):
+10.6% avg on agent tasks (AppWorld), +8.6% on domain tasks (FiNER, Formula), and notably **82.3% lower
adaptation latency / 75.1% fewer rollouts than GEPA**. It adapts from natural execution feedback and does
not require labeled supervision. The authors explicitly position ACE as **complementary to GEPA**, not a
replacement ("ACE does not conflict with methods like GEPA, and in fact can be used jointly") — e.g. a
GEPA-optimized base prompt plus an ACE-grown playbook.

**Why now.** SambaNova [publicly committed](https://sambanova.ai/blog/ace-open-sourced-on-github)
(2025-11-19) that "we are working on ACE integration into DSPy, and we will update the community on that
in a few weeks." Eight months later, **nothing has landed**: as of 2026-07-13 there is no `ACE` in
`dspy/teleprompt/`, and no ACE/playbook issue or PR in `stanfordnlp/dspy`. There's an open,
well-scoped gap and a paper with a reference implementation ([ace-agent/ace](https://github.com/ace-agent/ace))
to validate against. (Coordination note: worth pinging the paper authors / SambaNova before merge so a
parallel effort isn't wasted.)

## 3. Non-goals

- Not replacing GEPA or MIPROv2. ACE is a distinct, complementary optimizer.
- Not (initially) the paper's **online / inference-time** playbook adaptation (agent memory updated
  live). v1 targets the **offline compile-time** setting that matches the `Teleprompter.compile`
  contract; online mode is a documented future extension (§7).
- No new hard dependency in DSPy core. `ace` stays optional, imported lazily inside `compile` — exactly
  as `dspy.GEPA` lazily imports `gepa`.

## 4. Precedent: how GEPA is packaged (and how ACE mirrors it)

`dspy.GEPA` is the template. Its structure, verified against current `main`:

- **Standalone engine (`gepa` package).** Public surface is one function,
  `optimize(seed_candidate, trainset, valset, adapter, ...) -> GEPAResult`. Pure wiring: builds strategy
  objects, constructs a `GEPAEngine`, runs it, returns an immutable `GEPAResult` from a mutable
  `GEPAState`. **The engine treats scores as opaque floats and traces as fully opaque** — it never
  imports dspy.
- **Adapter contract** (`gepa.core.adapter.GEPAAdapter`, a structural `Protocol`): two required methods —
  `evaluate(batch, candidate, capture_traces) -> EvaluationBatch` and
  `make_reflective_dataset(candidate, eval_batch, components_to_update)` — plus an optional
  `propose_new_texts` and optional `get_adapter_state`/`set_adapter_state` persistence hooks.
- **Thin DSPy wrapper** (`dspy/teleprompt/gepa/gepa.py`): `GEPA(Teleprompter)` validates `metric` via
  `inspect.signature(metric).bind(...)`, lazily imports `gepa.optimize` inside `compile`, builds a
  `DspyAdapter` that (a) applies a candidate by deep-copying the student and rewriting
  `pred.signature = pred.signature.with_instructions(candidate[name])`, (b) evaluates via
  `bootstrap_trace_data` / `Evaluate`, and (c) maps the winning candidate back to a `Module`.

The base class is trivial and imposes no friction:

```python
class Teleprompter:
    def compile(self, student, *, trainset, teacher=None, valset=None, **kwargs) -> Module: ...
```

So `dspy.ACE(metric=...).compile(program, trainset=...)` fits with `metric` on `__init__` and the program
as `student` — no base-class change needed.

## 5. Proposed architecture

Two layers, mirroring GEPA one-to-one where the concepts coincide and diverging only where ACE's method
genuinely differs.

### 5.1 `ace` package (standalone engine)

```
ace/
  api.py            # optimize(seed_playbook, trainset, valset, adapter, ...) -> ACEResult
  core/
    adapter.py      # ACEAdapter Protocol + EvaluationBatch (mirrors gepa.core.adapter)
    playbook.py     # Playbook / Bullet dataclasses — the serializable artifact
    merge.py        # delta-merge + grow-and-refine (dedup/prune)  <-- deterministic core
    curator.py      # applies Reflector deltas to a Playbook via merge.py
    engine.py       # the Generator->Reflector->Curator loop + acceptance
    state.py        # ACEState: playbook history, counters, checkpoint
    result.py       # ACEResult (immutable, JSON-serializable)
  strategies/
    reflection.py   # default Reflector signature/prompt (overridable)
```

`optimize(...)` stays pure wiring returning an immutable `ACEResult` built from a mutable `ACEState`, and
the engine treats adapter outputs (scores, traces) as opaque — copied straight from GEPA's separation of
concerns.

### 5.2 `ACEAdapter` contract

Same shape as `GEPAAdapter`, so the DSPy adapter is nearly a re-skin of `DspyAdapter`:

| Method | Role |
|---|---|
| `evaluate(batch, playbook, capture_traces) -> EvaluationBatch` | **Generator** side: run the program (with the playbook injected as context) over the batch; return outputs + scores, and traces when asked. |
| `make_reflective_dataset(playbook, eval_batch, ...)` | Turn traces + metric feedback into the Reflector's input (what worked / what failed per example). |
| `propose_deltas(playbook, reflective_dataset) -> list[Delta]` | **Reflector** side: distill compact candidate bullets (deltas). ACE-specific; the analog of GEPA's `propose_new_texts`. |
| `get/set_adapter_state` (optional) | Checkpoint/resume. |

`EvaluationBatch` carries `outputs / scores / trajectories`, identical to GEPA.

### 5.3 The DSPy wrapper (`dspy/teleprompt/ace/ace.py`)

`ACE(Teleprompter)` — `metric` validated via `inspect.signature(...).bind(...)`; `compile` lazily
`from ace import optimize`. The DSPy adapter's **playbook-injection** step is the one real design choice:
the playbook is rendered into each predictor's context (prepended to the instruction, or supplied via a
dedicated input field). `build_program(playbook)` deep-copies the student and applies that injection —
structurally the same move as GEPA's `with_instructions`, just with a rendered playbook instead of a raw
instruction string.

### 5.4 Where ACE diverges from GEPA (the net-new work)

This is the heart of the library and the part that is **pure, deterministic, unit-testable Python** —
no LLM in the loop:

- **Artifact.** GEPA candidate = `dict[str, str]`. ACE artifact = `Playbook` (ordered `Bullet`s, each
  `{id, content, helpful, harmful}`). Serializable to/from JSON.
- **State evolution.** GEPA selects whole candidates on a Pareto frontier. ACE **merges deltas into one
  evolving playbook**: new bullets append, existing bullets update in place (counter increments),
  low-value bullets are pruned.
- **`merge.py` (the testable core).** Two deterministic operations, each with a crisp spec and table-
  driven tests:
  - *delta-merge:* apply a `list[Delta]` (add / update-counter / edit) to a `Playbook`, idempotently and
    order-independently where the paper requires it.
  - *grow-and-refine:* dedup near-identical bullets by embedding similarity above a threshold; prune by
    helpful/harmful ratio and/or size cap; support **proactive** (every delta) vs **lazy** (only past a
    size limit) refinement.

Everything above is deterministic given a fixed embedding function, so the merge/dedup/prune logic is
fully unit-testable without any model calls — the embedding function is injected and stubbed in tests.

## 6. Dogfooding

Generator / Reflector / Curator are themselves implemented as `dspy.Module`s (the LLM-driven parts:
generation, reflection, and delta-distillation), while `merge.py` stays plain Python. This gives the
"DSPy optimizing a DSPy program" story and lets users swap the Reflector/Curator prompts like any other
module.

## 7. Open questions for maintainers

1. **Playbook injection mechanism** — prepend to `signature.instructions`, or a first-class input field
   on the signature? Affects how ACE composes with GEPA (GEPA optimizes the instruction; ACE owns the
   playbook slot). *Leaning: dedicated field, so the two are orthogonal and jointly usable.*
2. **Online mode** — in scope for v1, or explicitly deferred? (I propose deferred; offline-only first.)
3. **Embedding dependency** — reuse `dspy.Embedder` for the dedup step to avoid a new dep. OK?
4. **Package ownership** — standalone `ace` under a neutral org (like `gepa-ai/gepa`), or fold the engine
   into an existing home? Mirroring GEPA's separate-package model is my default.
5. **Coordination** — is anyone already in contact with the ACE authors / SambaNova about their promised
   integration? I'd rather join that than fork it.

## 8. Rollout plan

1. RFC (this doc) → DSPy issue + Discord (#optimizers) for triage.
2. Standalone `ace` package: playbook + `merge.py` with full unit tests first (deterministic core),
   then engine, then `ACEAdapter`.
3. `dspy/teleprompt/ace/` wrapper + adapter (re-skin of `DspyAdapter`), lazy optional import.
4. Reproduce one paper benchmark (AppWorld or FiNER) against `ace-agent/ace` as ground truth.
5. Docs + tutorial paralleling `docs/.../gepa-*`.

## References

- ACE paper — https://arxiv.org/abs/2510.04618 (ICLR 2026)
- ACE reference implementation — https://github.com/ace-agent/ace
- SambaNova DSPy-integration commitment — https://sambanova.ai/blog/ace-open-sourced-on-github
- GEPA optimizer (packaging precedent) — https://github.com/gepa-ai/gepa · https://dspy.ai/api/optimizers/GEPA/overview/
