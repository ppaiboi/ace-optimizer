"""Immutable result of an ACE optimization run."""

from __future__ import annotations

from dataclasses import dataclass

from ace.playbook import Playbook


@dataclass(frozen=True)
class IterationRecord:
    iteration: int
    num_deltas: int
    candidate_score: float
    accepted_best: bool
    playbook_size: int
    metric_calls: int


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
    total_metric_calls: int = 0

    @property
    def improved(self) -> bool:
        return self.best_score > self.seed_score
