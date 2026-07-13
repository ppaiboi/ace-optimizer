"""The ACE optimization loop.

One loop, mirroring the reference `_train_single_sample`:
  * per-sample stepping
  * multi-round reflect -> regenerate on wrong answers (``max_num_rounds``;
    set to 1 for a cheap single-shot variant)
  * counter updates applied immediately after each reflection
  * Curator runs every ``curator_frequency`` steps from accumulated lessons
  * validation runs only every ``eval_steps`` to pick the best playbook

Pure orchestration: the engine only sees floats (scores) and opaque objects
(deltas). It drives Generator/Reflector/Curator through the adapter primitives
(``generate_one`` / ``reflect_one`` / ``curate`` / ``evaluate``) and evolves the
playbook with the deterministic core in ``ace.merge``. The same primitives power
*online* adaptation — see ``ace.online``.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

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
    kept = tuple(b for b in playbook.bullets if b.id in kept_ids)  # preserve order
    return Playbook(bullets=kept, section_order=playbook.section_order)


def _reflect_and_bump(adapter: Any, sample: Any, gen: dict, playbook: Playbook):
    """Reflect on one generation; apply any helpful/harmful counter bumps."""
    refl = adapter.reflect_one(sample, gen, playbook)
    if refl["tags"]:
        playbook = apply_deltas(playbook, refl["tags"])
    return playbook, refl


def optimize(
    seed_playbook: Playbook,
    trainset: Sequence[Any],
    adapter: Any,
    *,
    valset: Sequence[Any] | None = None,
    epochs: int = 1,
    max_num_rounds: int = 3,
    curator_frequency: int = 1,
    eval_steps: int = 100,
    perfect_score: float = 1.0,
    embed: EmbedFn | None = None,
    refine_threshold: float = 0.90,
    max_bullets: int | None = None,
    seed: int = 0,
) -> ACEResult:
    """Grow a playbook from ``seed_playbook`` over ``trainset``.

    ACE *accumulates*: each sample's lessons are folded into the running
    playbook. ``best_playbook`` tracks the highest valset score (checked every
    ``eval_steps``) so a late regression can't lose ground.
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

    lessons_buffer: list[str] = []
    step = 0
    for _ in range(epochs):
        order = list(range(len(trainset)))
        rng.shuffle(order)
        for idx in order:
            step += 1
            sample = trainset[idx]

            # -- generate (+ multi-round reflect/regenerate on failure) -------
            gen = adapter.generate_one(sample, playbook, reflection="(empty)")
            metric_calls += 1

            if gen["score"] < perfect_score:
                for _r in range(max_num_rounds):
                    playbook, refl = _reflect_and_bump(adapter, sample, gen, playbook)
                    lessons_buffer.extend(refl["lessons"])
                    gen = adapter.generate_one(
                        sample, playbook, reflection=refl["reflection_text"]
                    )
                    metric_calls += 1
                    if gen["score"] >= perfect_score:
                        break
            else:
                playbook, refl = _reflect_and_bump(adapter, sample, gen, playbook)
                lessons_buffer.extend(refl["lessons"])

            # -- curate every curator_frequency steps -------------------------
            n_deltas = 0
            if step % curator_frequency == 0 and lessons_buffer:
                adds = adapter.curate(playbook, lessons_buffer)
                n_deltas = len(adds)
                playbook = apply_deltas(playbook, adds)
                if embed is not None:
                    playbook = grow_and_refine(playbook, embed, threshold=refine_threshold)
                if max_bullets is not None and len(playbook.bullets) > max_bullets:
                    playbook = _cap(playbook, max_bullets)
                lessons_buffer = []

            # -- periodic validation to track the best playbook ---------------
            if step % eval_steps == 0:
                s = full_score(playbook)
                metric_calls += len(valset)
                if s >= best_score:
                    best_score, best_playbook = s, playbook
                history.append(
                    IterationRecord(
                        iteration=step, num_deltas=n_deltas, candidate_score=s,
                        accepted_best=(s >= best_score),
                        playbook_size=len(playbook.bullets), metric_calls=metric_calls,
                    )
                )

    # final validation
    s = full_score(playbook)
    metric_calls += len(valset)
    if s >= best_score:
        best_score, best_playbook = s, playbook

    return ACEResult(
        best_playbook=best_playbook,
        final_playbook=playbook,
        best_score=best_score,
        seed_score=seed_score,
        history=tuple(history),
        total_metric_calls=metric_calls,
    )
