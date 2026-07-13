"""Promotion gate — CD for prompts.

Every naive "self-improving prompt" merges the Curator's deltas straight into
the live playbook. That's how prompts silently rot. Instead, a candidate
playbook (current + proposed deltas) is **validated against a held-out set**
before it goes live, under an explicit policy. Rejected deltas are quarantined
with a reason, not dropped silently — so a human can see what was tried.

``evaluate`` is injected (``playbook, holdout -> mean score in [0, 1]``), so the
gate is pure decision logic and unit-testable without any model calls.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ace.merge import Delta, apply_deltas
from ace.playbook import Playbook
from ace.store import PlaybookStore


@dataclass(frozen=True)
class PromotionPolicy:
    """When may a candidate replace the live playbook?

    * ``min_delta``      — required improvement over current to promote.
    * ``max_regression`` — a candidate that *drops* more than this vs current is
      rejected even if some later check would pass (belt and suspenders).
    """

    min_delta: float = 0.0
    max_regression: float = 0.0


_DEFAULT_POLICY = PromotionPolicy()


@dataclass(frozen=True)
class Decision:
    promoted: bool
    reason: str
    current_score: float
    candidate_score: float

    @property
    def delta(self) -> float:
        return self.candidate_score - self.current_score


def evaluate_candidate(
    current: Playbook,
    deltas: Sequence[Delta],
    holdout: Sequence[Any],
    evaluate: Callable[[Playbook, Sequence[Any]], float],
    policy: PromotionPolicy = _DEFAULT_POLICY,
) -> tuple[Decision, Playbook]:
    """Decide whether ``current + deltas`` should be promoted. Returns the
    decision and the candidate playbook (whether or not it passed)."""
    candidate = apply_deltas(current, list(deltas))
    cur = float(evaluate(current, holdout))
    cand = float(evaluate(candidate, holdout))
    improvement = cand - cur

    if improvement < -abs(policy.max_regression):
        reason = f"regression {improvement:+.3f} exceeds max_regression"
        return Decision(False, reason, cur, cand), candidate
    if improvement < policy.min_delta:
        reason = f"improvement {improvement:+.3f} below min_delta {policy.min_delta:+.3f}"
        return Decision(False, reason, cur, cand), candidate
    return Decision(True, f"promoted ({improvement:+.3f})", cur, cand), candidate


def promote(
    store: PlaybookStore,
    deltas: Sequence[Delta],
    holdout: Sequence[Any],
    evaluate: Callable[[Playbook, Sequence[Any]], float],
    *,
    policy: PromotionPolicy = _DEFAULT_POLICY,
    meta: dict | None = None,
) -> Decision:
    """Gate a batch of deltas against ``store``'s live playbook and, if it
    passes, commit a new version. On failure, quarantine the deltas with the
    reason. The commit's metadata records the before/after scores for audit."""
    decision, _ = evaluate_candidate(store.head(), deltas, holdout, evaluate, policy)
    if decision.promoted:
        store.append(
            list(deltas),
            meta={
                **(meta or {}),
                "reason": decision.reason,
                "current_score": decision.current_score,
                "candidate_score": decision.candidate_score,
            },
        )
    else:
        store.quarantine(list(deltas), decision.reason)
    return decision
