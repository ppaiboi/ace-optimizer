"""Immutable result of an ACE optimization run — plus a comprehensive,
replayable trace of *everything* the run did.

The trace is intentionally exhaustive so any UI can be built on top of it
without re-running: every LM call (full rendered prompt + raw completion), every
generation attempt and its score, every reflection's lessons/reasoning, every
curation's new bullets, and a playbook snapshot at each validation checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

from ace.playbook import Bullet, Playbook


@dataclass(frozen=True)
class IterationRecord:
    iteration: int
    num_deltas: int
    candidate_score: float
    accepted_best: bool
    playbook_size: int
    metric_calls: int


@dataclass(frozen=True)
class LMCall:
    """One raw LM interaction: the full rendered prompt and the completion.

    ``messages`` is the exact chat payload sent to the model (system + user,
    including the injected playbook and the real task input), so you can show
    how the *whole* prompt — not just the instruction — evolves."""

    role: str  # "generator" | "reflector" | "curator"
    messages: tuple[dict, ...]
    completion: str


@dataclass(frozen=True)
class GenAttempt:
    """One Generator pass on a sample (initial, or a post-reflection retry)."""

    round: int
    reflection_in: str
    prediction: str
    score: float
    feedback: str
    cited: tuple[str, ...]
    calls: tuple[LMCall, ...]


@dataclass(frozen=True)
class ReflectRecord:
    """One Reflector pass: the lessons it drew and the counters it bumped."""

    lessons: tuple[str, ...]
    reflection_text: str
    tags: tuple[str, ...]  # e.g. "calc-00001:helpful"
    calls: tuple[LMCall, ...]


@dataclass(frozen=True)
class BulletEdit:
    """A bullet whose content and/or counters changed in place this step."""

    before: Bullet
    after: Bullet


@dataclass(frozen=True)
class StepRecord:
    """Everything that happened on a single training sample.

    ``added`` / ``removed`` / ``edited`` are the full playbook diff over the
    step — ACE both *grows* (Add) and *refines* (Delete, dedup-Merge, cap-prune,
    Edit), so all three are tracked, not just additions."""

    step: int
    sample_index: int
    inputs: str
    gold: str
    attempts: tuple[GenAttempt, ...]
    reflections: tuple[ReflectRecord, ...]
    curated_lessons: tuple[str, ...]
    added: tuple[Bullet, ...]
    removed: tuple[Bullet, ...]
    edited: tuple[BulletEdit, ...]
    curate_calls: tuple[LMCall, ...]
    playbook_size: int


@dataclass(frozen=True)
class TraceCheckpoint:
    """A replayable snapshot at a validation checkpoint: the running score and
    the exact playbook at that point."""

    step: int
    val_score: float
    metric_calls: int
    playbook: Playbook


@dataclass(frozen=True)
class ACEResult:
    """What ``optimize`` returns.

    ``best_playbook`` is the highest-scoring playbook on the valset;
    ``final_playbook`` is the last accumulated one (they differ if a late
    iteration regressed).
    """

    best_playbook: Playbook
    final_playbook: Playbook
    best_score: float
    seed_score: float
    history: tuple[IterationRecord, ...] = ()
    trace: tuple[TraceCheckpoint, ...] = ()
    steps: tuple[StepRecord, ...] = ()
    total_metric_calls: int = 0

    @property
    def improved(self) -> bool:
        return self.best_score > self.seed_score
