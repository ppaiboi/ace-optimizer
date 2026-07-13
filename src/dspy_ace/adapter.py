"""DSPy adapter: bridges a citing dspy Generator to the ACE engine.

Implements ``ace.ACEAdapter``. Faithful to the reference: the Generator
receives the playbook as an input and cites the ``bullet_ids`` it used; only
those cited bullets can be tagged helpful/harmful by the Reflector (real
attribution). The Reflector and Curator are ``dspy.Module``s (dogfooding).

Requires the student to be a single-signature program (exactly one predictor),
which the citing generator augments. Multi-predictor programs are a future
extension.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

import dspy

from ace.core.adapter import EvaluationBatch
from ace.merge import Add, Bump
from ace.playbook import Playbook
from dspy_ace.generator import ACEGenerator, build_ace_predictor
from dspy_ace.signatures import Curate, Reflect

_ID_RE = re.compile(r"[a-z]{3,}-\d{5}")


def _extract_ids(text: str) -> list[str]:
    """Pull well-formed bullet ids out of a free-form string (robust to prose)."""
    return _ID_RE.findall(text or "")


def _lines(text: str) -> list[str]:
    """Non-empty, stripped lines of a string output."""
    return [x.strip() for x in (text or "").splitlines() if x.strip()]


def _as_score_feedback(result: Any) -> tuple[float, str]:
    """Normalize a metric return into (score, feedback), GEPA-style."""
    if hasattr(result, "score"):
        score = float(result["score"] if hasattr(result, "__getitem__") else result.score)
        fb = getattr(result, "feedback", None) or f"score={score}"
        return score, fb
    score = float(result)
    return score, f"score={score}"


def _cited_bullets_text(playbook: Playbook, cited_ids: Sequence[str]) -> tuple[str, list[str]]:
    """Render only the bullets the generator cited; return (text, valid_ids)."""
    known = {b.id: b for b in playbook.bullets}
    valid = [i for i in cited_ids if i in known]
    if not valid:
        return "(no bullets cited)", []
    return "\n".join(known[i].render() for i in valid), valid


class DspyAdapter:
    """Adapts a single-signature dspy program for ACE optimization."""

    def __init__(
        self,
        student: dspy.Module,
        metric: Callable,
        *,
        reflection_lm: dspy.LM | None = None,
        failure_score: float = 0.0,
        num_threads: int = 1,
    ) -> None:
        preds = list(student.named_predictors())
        if len(preds) != 1:
            raise ValueError(
                "dspy.ACE currently requires a single-predictor program "
                f"(got {len(preds)}); citing multi-module programs is future work."
            )
        self.metric = metric
        self.reflection_lm = reflection_lm
        self.failure_score = failure_score
        self.num_threads = max(1, num_threads)
        # Build the citing generator from the student's task signature.
        self._base_signature = preds[0][1].signature
        self._predictor = build_ace_predictor(self._base_signature)
        self._reflect = dspy.Predict(Reflect)
        self._curate = dspy.Predict(Curate)

    # -- candidate -> program ------------------------------------------------

    def build_program(self, playbook: Playbook) -> dspy.Module:
        return ACEGenerator(self._predictor.deepcopy(), playbook.render().strip())

    # -- evaluation ----------------------------------------------------------

    def evaluate(
        self,
        batch: Sequence[Any],
        playbook: Playbook,
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        prog = self.build_program(playbook)

        def run_one(ex):
            inputs = ex.inputs() if hasattr(ex, "inputs") else ex
            try:
                pred = prog(**inputs)
                score, feedback = _as_score_feedback(self.metric(ex, pred))
                cited = _extract_ids(str(getattr(pred, "bullet_ids", "") or ""))
            except Exception as e:  # never raise on a single example
                pred, score, feedback, cited = None, self.failure_score, f"error: {e}", []
            return pred, score, {
                "example": ex, "pred": pred, "feedback": feedback, "cited": cited,
            }

        if self.num_threads > 1 and len(batch) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=self.num_threads) as pool:
                results = list(pool.map(run_one, batch))
        else:
            results = [run_one(ex) for ex in batch]

        outputs = [r[0] for r in results]
        scores = [r[1] for r in results]
        trajectories = [r[2] for r in results] if capture_traces else None
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    # -- per-sample Generator / Reflector / Curator primitives ---------------

    def generate_one(self, sample: Any, playbook: Playbook, reflection: str = "(empty)") -> dict:
        """One generation on a single sample, with optional reflection retry."""
        inputs = sample.inputs() if hasattr(sample, "inputs") else sample
        prog = self.build_program(playbook)
        try:
            with dspy.context(lm=dspy.settings.lm):
                pred = prog(reflection=reflection, **inputs)
            score, feedback = _as_score_feedback(self.metric(sample, pred))
            cited = _extract_ids(str(getattr(pred, "bullet_ids", "") or ""))
        except Exception as e:
            pred, score, feedback, cited = None, self.failure_score, f"error: {e}", []
        return {"pred": pred, "score": score, "feedback": feedback, "cited": cited}

    def reflect_one(self, sample: Any, gen: dict, playbook: Playbook) -> dict:
        """One reflection: tag cited bullets + extract lessons + a retry hint."""
        inputs = sample.inputs() if hasattr(sample, "inputs") else sample
        cited_text, cited_ids = _cited_bullets_text(playbook, gen.get("cited", []))
        try:
            with dspy.context(lm=self.reflection_lm or dspy.settings.lm):
                r = self._reflect(
                    task=str(dict(inputs)),
                    generated_output="" if gen["pred"] is None else str(gen["pred"]),
                    feedback=gen["feedback"],
                    bullets_in_play=cited_text,
                )
        except Exception:  # a bad LM parse -> no tags/lessons this round
            return {"tags": [], "lessons": [], "reflection_text": gen["feedback"]}
        allowed = set(cited_ids)
        tags = [d for t in _lines(r.bullet_tags) if (d := _parse_tag(t, allowed))]
        lessons = _lines(r.lessons)
        retry_hint = gen["feedback"] + ("\n" + "\n".join(lessons) if lessons else "")
        return {"tags": tags, "lessons": lessons, "reflection_text": retry_hint}

    def curate(self, playbook: Playbook, lessons: Sequence[str]) -> list[Add]:
        """Author new bullets from accumulated lessons (Curator role)."""
        if not lessons:
            return []
        try:
            with dspy.context(lm=self.reflection_lm or dspy.settings.lm):
                c = self._curate(
                    existing_playbook=playbook.render().strip() or "(empty)",
                    lessons="\n".join(f"- {ln}" for ln in lessons),
                )
        except Exception:  # a bad LM parse -> add nothing this step
            return []
        return [d for a in _lines(c.additions) if (d := _parse_addition(a))]


def _parse_tag(tag: str, allowed_ids: set[str]) -> Bump | None:
    if ":" not in tag:
        return None
    bid, _, kind = tag.partition(":")
    bid, kind = bid.strip(), kind.strip().lower()
    if bid in allowed_ids and kind in ("helpful", "harmful"):
        return Bump(id=bid, tag=kind)
    return None


def _parse_addition(add: str) -> Add | None:
    if "::" in add:
        section, _, content = add.partition("::")
        section, content = section.strip(), content.strip()
    else:
        section, content = "general", add.strip()
    return Add(content=content, section=section) if content else None
