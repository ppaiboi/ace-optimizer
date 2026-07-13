"""``dspy.ACE`` — the ACE optimizer as a DSPy teleprompter.

Mirrors ``dspy.GEPA``'s shape: subclass ``Teleprompter``, validate ``metric``,
lazily use the standalone ``ace`` engine inside ``compile``, and return a
compiled ``Module`` with the learned playbook injected. Intended to live at
``dspy/teleprompt/ace/ace.py`` when contributed upstream.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import Any

import dspy
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.utils.annotation import experimental

from ace.merge import EmbedFn
from ace.playbook import Playbook
from dspy_ace.adapter import DspyAdapter


@experimental(version="3.2.0")
class ACE(Teleprompter):
    """Agentic Context Engineering optimizer.

    Grows a *playbook* of reusable strategies from execution feedback and
    injects it into the program's instructions. Complementary to GEPA.
    """

    def __init__(
        self,
        metric: Callable,
        *,
        reflection_lm: dspy.LM | None = None,
        seed_playbook: Playbook | None = None,
        epochs: int = 1,
        max_num_rounds: int = 3,
        curator_frequency: int = 1,
        eval_steps: int = 100,
        embed: EmbedFn | None = None,
        refine_threshold: float = 0.90,
        max_bullets: int | None = None,
        failure_score: float = 0.0,
        num_threads: int = 1,
        seed: int = 0,
    ) -> None:
        super().__init__()
        try:
            inspect.signature(metric).bind(None, None)
        except TypeError as e:
            raise TypeError(
                "ACE metric must accept at least (gold, pred)."
            ) from e
        self.metric = metric
        self.reflection_lm = reflection_lm
        self.seed_playbook = seed_playbook or Playbook()
        self.epochs = epochs
        self.max_num_rounds = max_num_rounds
        self.curator_frequency = curator_frequency
        self.eval_steps = eval_steps
        self.embed = embed
        self.refine_threshold = refine_threshold
        self.max_bullets = max_bullets
        self.failure_score = failure_score
        self.num_threads = num_threads
        self.seed = seed

    def compile(
        self,
        student: dspy.Module,
        *,
        trainset: Sequence[Any],
        teacher: dspy.Module | None = None,
        valset: Sequence[Any] | None = None,
        **kwargs,
    ) -> dspy.Module:
        assert trainset, "trainset must be non-empty"
        assert teacher is None, "ACE does not use a teacher program"

        # Lazy import keeps `ace` an optional dependency of dspy.
        from ace.engine import optimize

        adapter = DspyAdapter(
            student=student,
            metric=self.metric,
            reflection_lm=self.reflection_lm,
            failure_score=self.failure_score,
            num_threads=self.num_threads,
        )
        result = optimize(
            self.seed_playbook, trainset, adapter,
            valset=valset, epochs=self.epochs,
            max_num_rounds=self.max_num_rounds,
            curator_frequency=self.curator_frequency,
            eval_steps=self.eval_steps,
            embed=self.embed, refine_threshold=self.refine_threshold,
            max_bullets=self.max_bullets, seed=self.seed,
        )

        compiled = adapter.build_program(result.best_playbook)
        compiled.ace_playbook = result.best_playbook
        compiled.ace_result = result
        return compiled
