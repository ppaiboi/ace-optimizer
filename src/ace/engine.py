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

from ace.checkpoint import Checkpoint, config_hash, load_checkpoint, save_checkpoint
from ace.merge import EmbedFn, apply_deltas, grow_and_refine
from ace.playbook import Playbook
from ace.result import (
    ACEResult,
    BulletEdit,
    GenAttempt,
    IterationRecord,
    LMCall,
    ReflectRecord,
    StepRecord,
    TraceCheckpoint,
)


def _training_order(n: int, epochs: int, seed: int) -> list[int]:
    """The exact sequence of sample indices the loop visits — a deterministic
    function of ``seed`` (so resume can skip the first ``step`` of them)."""
    rng = random.Random(seed)
    sequence: list[int] = []
    for _ in range(epochs):
        order = list(range(n))
        rng.shuffle(order)
        sequence.extend(order)
    return sequence


def _calls(d: dict) -> tuple[LMCall, ...]:
    """Lift any captured raw LM calls out of an adapter's return dict."""
    return tuple(
        LMCall(role=c.get("role", ""), messages=tuple(c.get("messages", ())),
               completion=c.get("completion", ""))
        for c in d.get("calls", ())
    )


def _gen_attempt(round_: int, reflection_in: str, gen: dict) -> GenAttempt:
    return GenAttempt(
        round=round_,
        reflection_in=reflection_in,
        prediction=str(gen.get("pred", "") if gen.get("pred") is not None else ""),
        score=gen["score"],
        feedback=gen.get("feedback", ""),
        cited=tuple(gen.get("cited", ())),
        calls=_calls(gen),
    )


def _reflect_record(refl: dict) -> ReflectRecord:
    return ReflectRecord(
        lessons=tuple(refl.get("lessons", ())),
        reflection_text=refl.get("reflection_text", ""),
        tags=tuple(str(t) for t in refl.get("tags", ())),
        calls=_calls(refl),
    )


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
    interim_valset: Sequence[Any] | None = None,
    epochs: int = 1,
    max_num_rounds: int = 3,
    curator_frequency: int = 1,
    eval_steps: int = 100,
    perfect_score: float = 1.0,
    embed: EmbedFn | None = None,
    refine_threshold: float = 0.90,
    max_bullets: int | None = None,
    seed: int = 0,
    checkpoint_path: str | None = None,
    checkpoint_every: int = 50,
    resume: bool = True,
    progress: bool = False,
) -> ACEResult:
    """Grow a playbook from ``seed_playbook`` over ``trainset``.

    ACE *accumulates*: each sample's lessons are folded into the running
    playbook. ``best_playbook`` tracks the highest valset score (checked every
    ``eval_steps``) so a late regression can't lose ground.

    Cost/robustness knobs:
      * ``interim_valset`` — a (smaller) set for the periodic best-playbook
        checks; the full ``valset`` is used only for the final score. Cuts the
        dominant validation cost without changing the reported result.
      * ``checkpoint_path`` / ``checkpoint_every`` / ``resume`` — persist
        resumable state every N steps so a killed run picks up where it left off
        (see ``ace.checkpoint``).
      * ``progress`` — print a ``step/total`` line at each checkpoint.
    """
    if not trainset:
        raise ValueError("trainset must be non-empty")
    valset = valset if valset else trainset
    interim = interim_valset if interim_valset else valset

    def score_on(dataset: Sequence[Any], pb: Playbook) -> float:
        return adapter.evaluate(dataset, pb, capture_traces=False).mean_score

    sequence = _training_order(len(trainset), epochs, seed)
    total = len(sequence)
    chash = config_hash(
        seed=seed, epochs=epochs, n_train=len(trainset), n_val=len(valset),
        max_num_rounds=max_num_rounds, curator_frequency=curator_frequency,
        eval_steps=eval_steps, max_bullets=max_bullets,
    )

    # -- restore from checkpoint, or start fresh -----------------------------
    ckpt = load_checkpoint(checkpoint_path, chash) if (checkpoint_path and resume) else None
    if ckpt is not None:
        playbook = ckpt.playbook
        best_playbook, best_score = ckpt.best_playbook, ckpt.best_score
        seed_score = ckpt.seed_score
        metric_calls = ckpt.metric_calls
        lessons_buffer = list(ckpt.lessons_buffer)
        history = list(ckpt.history)
        start_step = ckpt.step
        trace = [TraceCheckpoint(start_step, best_score, metric_calls, playbook)]
        if progress:
            print(f"[ace] resumed from {checkpoint_path} at step {start_step}/{total}")
    else:
        playbook = seed_playbook
        seed_score = score_on(valset, seed_playbook)
        best_playbook, best_score = seed_playbook, seed_score
        metric_calls = len(valset)
        history = []
        lessons_buffer = []
        start_step = 0
        trace = [TraceCheckpoint(0, seed_score, metric_calls, seed_playbook)]

    steps: list[StepRecord] = []

    def _save(step: int) -> None:
        if checkpoint_path:
            save_checkpoint(checkpoint_path, Checkpoint(
                config_hash=chash, step=step, seed_score=seed_score,
                best_score=best_score, metric_calls=metric_calls, playbook=playbook,
                best_playbook=best_playbook, lessons_buffer=lessons_buffer,
                history=history,
            ))

    # -- main loop (resumes at start_step in the deterministic order) --------
    for step_i in range(start_step, total):
        step = step_i + 1
        idx = sequence[step_i]
        sample = trainset[idx]
        pb_start = {b.id: b for b in playbook.bullets}  # for the step diff
        attempts: list[GenAttempt] = []
        reflections: list[ReflectRecord] = []

        # -- generate (+ multi-round reflect/regenerate on failure) -------
        gen = adapter.generate_one(sample, playbook, reflection="(empty)")
        metric_calls += 1
        attempts.append(_gen_attempt(0, "(empty)", gen))

        if gen["score"] < perfect_score:
            for _r in range(max_num_rounds):
                playbook, refl = _reflect_and_bump(adapter, sample, gen, playbook)
                reflections.append(_reflect_record(refl))
                lessons_buffer.extend(refl["lessons"])
                gen = adapter.generate_one(
                    sample, playbook, reflection=refl["reflection_text"]
                )
                metric_calls += 1
                attempts.append(_gen_attempt(_r + 1, refl["reflection_text"], gen))
                if gen["score"] >= perfect_score:
                    break
        else:
            playbook, refl = _reflect_and_bump(adapter, sample, gen, playbook)
            reflections.append(_reflect_record(refl))
            lessons_buffer.extend(refl["lessons"])

        # -- curate every curator_frequency steps -------------------------
        n_deltas = 0
        curated_lessons: tuple[str, ...] = ()
        curate_calls: tuple[LMCall, ...] = ()
        if step % curator_frequency == 0 and lessons_buffer:
            curated_lessons = tuple(lessons_buffer)
            adds = adapter.curate(playbook, lessons_buffer)
            n_deltas = len(adds)
            curate_calls = _calls(getattr(adapter, "last_curate", {}) or {})
            playbook = apply_deltas(playbook, adds)
            if embed is not None:
                playbook = grow_and_refine(playbook, embed, threshold=refine_threshold)
            if max_bullets is not None and len(playbook.bullets) > max_bullets:
                playbook = _cap(playbook, max_bullets)
            lessons_buffer = []

        # full step diff: ACE both grows (Add) and refines (Delete/merge/cap/Edit)
        pb_end = {b.id: b for b in playbook.bullets}
        added = tuple(b for bid, b in pb_end.items() if bid not in pb_start)
        removed = tuple(b for bid, b in pb_start.items() if bid not in pb_end)
        edited = tuple(
            BulletEdit(before=pb_start[bid], after=pb_end[bid])
            for bid in pb_end.keys() & pb_start.keys()
            if pb_end[bid] != pb_start[bid]
        )

        steps.append(StepRecord(
            step=step, sample_index=idx,
            inputs=gen.get("inputs", ""), gold=gen.get("gold", ""),
            attempts=tuple(attempts), reflections=tuple(reflections),
            curated_lessons=curated_lessons,
            added=added, removed=removed, edited=edited,
            curate_calls=curate_calls, playbook_size=len(playbook.bullets),
        ))

        # -- periodic validation (on the cheaper interim set) -------------
        if step % eval_steps == 0:
            s = score_on(interim, playbook)
            metric_calls += len(interim)
            if s >= best_score:
                best_score, best_playbook = s, playbook
            history.append(IterationRecord(
                iteration=step, num_deltas=n_deltas, candidate_score=s,
                accepted_best=(s >= best_score),
                playbook_size=len(playbook.bullets), metric_calls=metric_calls,
            ))
            trace.append(TraceCheckpoint(step, s, metric_calls, playbook))
            _save(step)
            if progress:
                print(f"[ace] step {step}/{total}  interim_val={s:.3f}  "
                      f"bullets={len(playbook.bullets)}  calls={metric_calls}")
        elif checkpoint_path and step % checkpoint_every == 0:
            _save(step)
            if progress:
                print(f"[ace] step {step}/{total}  bullets={len(playbook.bullets)}")

    # -- final validation on the FULL valset ---------------------------------
    s = score_on(valset, playbook)
    metric_calls += len(valset)
    if s >= best_score:
        best_score, best_playbook = s, playbook
    if not trace or trace[-1].step != total:
        trace.append(TraceCheckpoint(total, s, metric_calls, playbook))
    _save(total)

    return ACEResult(
        best_playbook=best_playbook,
        final_playbook=playbook,
        best_score=best_score,
        seed_score=seed_score,
        history=tuple(history),
        trace=tuple(trace),
        steps=tuple(steps),
        total_metric_calls=metric_calls,
    )
