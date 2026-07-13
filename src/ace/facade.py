"""``ACE`` — the framework-agnostic facade.

The whole design is the split between two paths:

* **Hot path** (``augment``): a pure string op — read the cached playbook, inject
  it into the system prompt. No LLM call, no I/O beyond a cached read, no new
  failure mode. If the learning subsystem is down, inference doesn't notice.
* **Learning path** (``observe`` → ``learn``): async and batched — queue traces,
  then diagnose (Reflector) → propose deltas (Curator) → validate against a
  holdout (gate) → promote a new, versioned, rollback-able playbook.

    ace = ACE(store=SQLitePlaybookStore("pb.db"), reflector=r, curator=c)
    system = ace.augment(base_system_prompt)      # hot path
    ...call your model...
    ace.observe(input=user_msg, output=reply, signal=my_signal, gold=label)
    ace.learn(holdout=holdout, evaluate=eval_fn)  # learning path (offline)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

# a Curator is anything with .curate(playbook, lessons) -> [Delta]
from ace.curate import Curator  # noqa: E402  (grouped with the role imports)
from ace.gate import Decision, PromotionPolicy, promote
from ace.playbook import Playbook
from ace.reflect import Reflector
from ace.signals import Feedback, Interaction, Signal
from ace.store import PlaybookStore


@dataclass
class _Pending:
    interaction: Interaction
    feedback: Feedback


@dataclass
class ACE:
    """Continuous playbook learning around any LLM app.

    ``reflector`` and ``curator`` are the only pieces that touch a model; supply
    your own (or the LLM-backed defaults). ``signal`` scores traces when you call
    ``observe`` without an explicit feedback.
    """

    store: PlaybookStore
    reflector: Reflector | None = None
    curator: Curator | None = None
    signal: Signal | None = None
    policy: PromotionPolicy = field(default_factory=PromotionPolicy)
    _queue: list[_Pending] = field(default_factory=list, init=False)

    # -- hot path ------------------------------------------------------------
    def playbook(self) -> Playbook:
        return self.store.head()

    def augment(self, system_prompt: str = "") -> str:
        """Inject the current playbook into a system prompt. Pure + fast."""
        rendered = self.store.head().render().strip()
        if not rendered:
            return system_prompt
        block = (
            "# Playbook — learned strategies. Apply the relevant ones.\n" + rendered
        )
        return f"{system_prompt}\n\n{block}".strip() if system_prompt else block

    # -- learning path (capture) --------------------------------------------
    def observe(
        self,
        *,
        input: Any,
        output: Any,
        signal: Signal | None = None,
        feedback: Feedback | None = None,
        gold: Any = None,
        telemetry: dict | None = None,
        meta: dict | None = None,
    ) -> None:
        """Queue a trace for later learning. Provide either an explicit
        ``feedback``, or a ``signal`` (falls back to the instance ``signal``)."""
        interaction = Interaction(
            input=input, output=output, gold=gold,
            telemetry=telemetry or {}, meta=meta or {},
        )
        fb = feedback
        if fb is None:
            sig = signal or self.signal
            if sig is None:
                raise ValueError("observe() needs a feedback= or a signal= (or ACE(signal=...))")
            fb = sig.score(interaction)
        self._queue.append(_Pending(interaction, fb))

    @property
    def pending(self) -> int:
        return len(self._queue)

    # -- learning path (offline consume) ------------------------------------
    def learn(
        self,
        *,
        holdout: Sequence[Any] | None = None,
        evaluate: Callable[[Playbook, Sequence[Any]], float] | None = None,
    ) -> Decision | None:
        """Diagnose queued traces, propose deltas, and (if a holdout+evaluate is
        given) gate them before promoting. Without a holdout, deltas are promoted
        unconditionally (dev mode). Returns the gate ``Decision`` (or ``None`` if
        there was nothing to learn)."""
        if self.reflector is None or self.curator is None:
            raise ValueError("learn() requires a reflector and a curator")
        if not self._queue:
            return None

        pb = self.store.head()
        lessons: list[str] = []
        for p in self._queue:
            refl = self.reflector.reflect(p.interaction, p.feedback, pb)
            lessons.extend(refl.lessons)
        self._queue.clear()

        deltas = self.curator.curate(pb, lessons)
        if not deltas:
            return None

        if holdout is not None and evaluate is not None:
            return promote(self.store, deltas, holdout, evaluate, policy=self.policy,
                           meta={"lessons": len(lessons)})
        # dev mode: no gate
        self.store.append(deltas, meta={"lessons": len(lessons), "gated": False})
        return None
