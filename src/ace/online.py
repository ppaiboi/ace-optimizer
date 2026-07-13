"""Online / inference-time playbook adaptation.

The paper's "online" mode: instead of a fixed trainset optimized once, the
playbook is *agent memory* that keeps evolving as the agent runs. Each episode
(one task the agent handles live) contributes deltas back into the playbook,
so later tasks benefit from lessons learned on earlier ones — with no labels,
using only natural execution feedback.

This is the *same* Generator->Reflector->Curator cycle as the offline engine,
minus the valset/acceptance bookkeeping: one batch of size 1 (the episode you
just ran), applied immediately and persisted.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ace.core.adapter import ACEAdapter
from ace.merge import EmbedFn, apply_deltas, grow_and_refine
from ace.playbook import Playbook


class OnlinePlaybook:
    """A living playbook an agent updates as it works.

    Usage per handled task::

        mem = OnlinePlaybook(adapter, playbook=Playbook.from_dict(load()))
        prog = mem.program()          # student with current playbook injected
        pred = prog(**task_inputs)    # the agent acts (the Generator)
        mem.learn([episode])          # reflect+curate+merge from what happened
        save(mem.playbook.to_dict())  # persist for the next task
    """

    def __init__(
        self,
        adapter: ACEAdapter,
        *,
        playbook: Playbook | None = None,
        embed: EmbedFn | None = None,
        refine_threshold: float = 0.90,
        refine_every: int = 1,
    ) -> None:
        self.adapter = adapter
        self.playbook = playbook if playbook is not None else Playbook()
        self._embed = embed
        self._refine_threshold = refine_threshold
        self._refine_every = max(1, refine_every)
        self._episodes = 0

    def program(self):
        """The student program with the current playbook injected (build_program)."""
        # build_program is adapter-specific; expose it when present (DSPy adapter).
        build = getattr(self.adapter, "build_program", None)
        if build is None:
            raise AttributeError("adapter does not expose build_program()")
        return build(self.playbook)

    def learn(self, episodes: Sequence[Any]) -> list:
        """Fold lessons from just-run episodes into the playbook. Returns deltas."""
        eval_batch = self.adapter.evaluate(episodes, self.playbook, capture_traces=True)
        reflective = self.adapter.make_reflective_dataset(self.playbook, eval_batch)
        deltas = self.adapter.propose_deltas(self.playbook, reflective)
        self.playbook = apply_deltas(self.playbook, deltas)
        self._episodes += 1
        if self._embed is not None and self._episodes % self._refine_every == 0:
            self.playbook = grow_and_refine(
                self.playbook, self._embed, threshold=self._refine_threshold
            )
        return deltas
