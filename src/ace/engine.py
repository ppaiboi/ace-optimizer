"""The ACE optimization loop (offline / compile-time).

Pure orchestration: the engine only sees floats (scores) and opaque objects
(deltas, traces). It drives the Generator/Reflector/Curator cycle through the
adapter and evolves the playbook with the deterministic core in ``ace.merge``.

The same primitives power *online* adaptation — see ``ace.online``.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from ace.core.adapter import ACEAdapter
from ace.merge import EmbedFn, apply_deltas, grow_and_refine
from ace.playbook import Playbook
from ace.result import ACEResult, IterationRecord


def _cap(playbook: Playbook, max_bullets: int) -> Playbook:
    """Keep the highest-value bullets when over the cap.

    Rank by net score (helpful - harmful), breaking ties toward more recently
    added bullets (higher id number), so proven strategies survive and stale
    unused ones are dropped first.
    """
    ranked = sorted(
        playbook.bullets,
        key=lambda b: (b.helpful - b.harmful, b.num),
        reverse=True,
    )
    kept_ids = {b.id for b in ranked[:max_bullets]}
    # preserve original order among survivors
    kept = tuple(b for b in playbook.bullets if b.id in kept_ids)
    return Playbook(bullets=kept, section_order=playbook.section_order)


def _minibatches(
    data: Sequence[Any], size: int, epochs: int, rng: random.Random
) -> list[list[Any]]:
    batches: list[list[Any]] = []
    for _ in range(epochs):
        idx = list(range(len(data)))
        rng.shuffle(idx)
        for i in range(0, len(idx), size):
            batches.append([data[j] for j in idx[i : i + size]])
    return batches


def optimize(
    seed_playbook: Playbook,
    trainset: Sequence[Any],
    adapter: ACEAdapter,
    *,
    valset: Sequence[Any] | None = None,
    epochs: int = 1,
    minibatch_size: int = 4,
    embed: EmbedFn | None = None,
    refine_threshold: float = 0.90,
    refine_every: int = 1,
    max_bullets: int | None = None,
    max_metric_calls: int | None = None,
    seed: int = 0,
) -> ACEResult:
    """Grow a playbook from ``seed_playbook`` over ``trainset``.

    ACE *accumulates*: each accepted delta batch is folded into the running
    playbook (it does not revert like a Pareto search). ``best_playbook`` still
    tracks the highest valset score so a late regression can't lose ground.
    """
    if not trainset:
        raise ValueError("trainset must be non-empty")
    valset = valset if valset else trainset
    rng = random.Random(seed)

    def full_score(pb: Playbook) -> float:
        return adapter.evaluate(valset, pb, capture_traces=False).mean_score

    playbook = seed_playbook
    seed_score = full_score(seed_playbook)
    best_playbook, best_score = seed_playbook, seed_score
    metric_calls = len(valset)
    history: list[IterationRecord] = []

    for it, batch in enumerate(_minibatches(trainset, minibatch_size, epochs, rng), 1):
        eval_batch = adapter.evaluate(batch, playbook, capture_traces=True)
        metric_calls += len(batch)

        reflective = adapter.make_reflective_dataset(playbook, eval_batch)
        deltas = adapter.propose_deltas(playbook, reflective)

        candidate = apply_deltas(playbook, deltas)
        if embed is not None and it % refine_every == 0:
            candidate = grow_and_refine(candidate, embed, threshold=refine_threshold)
        if max_bullets is not None and len(candidate.bullets) > max_bullets:
            candidate = _cap(candidate, max_bullets)

        candidate_score = full_score(candidate)
        metric_calls += len(valset)

        accepted_best = candidate_score >= best_score
        if accepted_best:
            best_playbook, best_score = candidate, candidate_score
        playbook = candidate  # accumulate regardless

        history.append(
            IterationRecord(
                iteration=it,
                num_deltas=len(deltas),
                candidate_score=candidate_score,
                accepted_best=accepted_best,
                playbook_size=len(candidate.bullets),
                metric_calls=metric_calls,
            )
        )
        if max_metric_calls is not None and metric_calls >= max_metric_calls:
            break

    return ACEResult(
        best_playbook=best_playbook,
        final_playbook=playbook,
        best_score=best_score,
        seed_score=seed_score,
        history=tuple(history),
        total_metric_calls=metric_calls,
    )
