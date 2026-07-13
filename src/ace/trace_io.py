"""Serialize an ``ACEResult`` (and its exhaustive trace) to a JSON-able dict.

Everything the run captured is preserved so any UI can be built on top without
re-running. Because the full raw prompts can be large (the FiNER context is
repeated in every message), ``include_prompts=False`` drops the raw ``messages``
while keeping all structured events — handy for a browser-loadable demo trace.
"""

from __future__ import annotations

from typing import Any

from ace.playbook import Bullet, Playbook
from ace.result import ACEResult, BulletEdit, GenAttempt, LMCall, ReflectRecord, StepRecord


def bullet_to_dict(b: Bullet) -> dict:
    return {
        "id": b.id, "section": b.section, "content": b.content,
        "helpful": b.helpful, "harmful": b.harmful, "render": b.render(),
    }


def playbook_to_dict(pb: Playbook) -> dict:
    return {"render": pb.render(), "bullets": [bullet_to_dict(b) for b in pb.bullets]}


def _call_to_dict(c: LMCall, include_prompts: bool) -> dict:
    d: dict[str, Any] = {"role": c.role, "completion": c.completion}
    if include_prompts:
        d["messages"] = list(c.messages)
    return d


def _attempt_to_dict(a: GenAttempt, include_prompts: bool) -> dict:
    return {
        "round": a.round, "reflection_in": a.reflection_in,
        "prediction": a.prediction, "score": a.score, "feedback": a.feedback,
        "cited": list(a.cited),
        "calls": [_call_to_dict(c, include_prompts) for c in a.calls],
    }


def _reflect_to_dict(r: ReflectRecord, include_prompts: bool) -> dict:
    return {
        "lessons": list(r.lessons), "reflection_text": r.reflection_text,
        "tags": list(r.tags),
        "calls": [_call_to_dict(c, include_prompts) for c in r.calls],
    }


def _edit_to_dict(e: BulletEdit) -> dict:
    return {"before": bullet_to_dict(e.before), "after": bullet_to_dict(e.after)}


def _step_to_dict(s: StepRecord, include_prompts: bool) -> dict:
    return {
        "step": s.step, "sample_index": s.sample_index,
        "inputs": s.inputs, "gold": s.gold,
        "attempts": [_attempt_to_dict(a, include_prompts) for a in s.attempts],
        "reflections": [_reflect_to_dict(r, include_prompts) for r in s.reflections],
        "curated_lessons": list(s.curated_lessons),
        "added": [bullet_to_dict(b) for b in s.added],
        "removed": [bullet_to_dict(b) for b in s.removed],
        "edited": [_edit_to_dict(e) for e in s.edited],
        "curate_calls": [_call_to_dict(c, include_prompts) for c in s.curate_calls],
        "playbook_size": s.playbook_size,
    }


def result_to_dict(result: ACEResult, *, include_prompts: bool = True) -> dict:
    """Full ACEResult -> JSON-able dict (playbooks, checkpoints, per-step firehose)."""
    return {
        "seed_score": result.seed_score,
        "best_score": result.best_score,
        "improved": result.improved,
        "total_metric_calls": result.total_metric_calls,
        "final_playbook": playbook_to_dict(result.final_playbook),
        "best_playbook": playbook_to_dict(result.best_playbook),
        "checkpoints": [
            {
                "step": c.step, "val_score": c.val_score,
                "metric_calls": c.metric_calls,
                "playbook": playbook_to_dict(c.playbook),
            }
            for c in result.trace
        ],
        "history": [
            {
                "iteration": h.iteration, "candidate_score": h.candidate_score,
                "num_deltas": h.num_deltas, "accepted_best": h.accepted_best,
                "playbook_size": h.playbook_size, "metric_calls": h.metric_calls,
            }
            for h in result.history
        ],
        "steps": [_step_to_dict(s, include_prompts) for s in result.steps],
    }
