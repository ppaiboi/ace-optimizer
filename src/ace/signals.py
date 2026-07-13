"""Feedback signals — how a trace becomes something to learn from.

The premise of ACE is *you have traces, not datasets*, so most signals are
label-free. A ``Signal`` turns one ``Interaction`` into ``Feedback`` (a score in
[0, 1] plus a human-readable note the Reflector reasons over). Anything that
touches an LLM or your product telemetry is injected as a callable, so the whole
module is hermetic and unit-testable with zero API calls.

Batteries included:
  * ``GroundTruth``    — you have labels (a metric over gold vs output)
  * ``LLMJudge``       — you don't (an injected judge callable / rubric)
  * ``ImplicitUser``   — product telemetry (accepted / edited / rejected)
  * ``ExecutionResult``— the world answered (tests passed, API returned 200, …)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Interaction:
    """One thing that happened: an input, the model's output, and optional
    context (gold label, telemetry, execution result) a signal can score."""

    input: Any
    output: Any
    gold: Any = None
    telemetry: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Feedback:
    """A score in [0, 1] and a note explaining it (what the Reflector reads)."""

    score: float
    text: str


@runtime_checkable
class Signal(Protocol):
    def score(self, interaction: Interaction) -> Feedback: ...


@dataclass
class GroundTruth:
    """You have labels. ``metric(gold, output) -> float`` in [0, 1]."""

    metric: Callable[[Any, Any], float]

    def score(self, interaction: Interaction) -> Feedback:
        s = float(self.metric(interaction.gold, interaction.output))
        note = "Correct." if s >= 0.999 else f"Score {s:.2f}. Expected: {interaction.gold}"
        return Feedback(score=s, text=note)


@dataclass
class LLMJudge:
    """You don't have labels. Delegate scoring to an injected judge callable
    ``judge(interaction, rubric) -> (score, rationale)`` — wrap your own model
    call. Kept dependency-free so tests pass a fake judge."""

    judge: Callable[[Interaction, str], tuple[float, str]]
    rubric: str = "Rate how well the output satisfies the request, 0.0-1.0."

    def score(self, interaction: Interaction) -> Feedback:
        s, rationale = self.judge(interaction, self.rubric)
        return Feedback(score=float(s), text=rationale)


@dataclass
class ImplicitUser:
    """Product telemetry as reward. Reads ``interaction.telemetry`` flags
    (accepted / edited / rejected). Accepted-unedited is best; rejected is worst;
    accepted-but-edited is partial credit."""

    edit_penalty: float = 0.5

    def score(self, interaction: Interaction) -> Feedback:
        t = interaction.telemetry
        if t.get("rejected"):
            return Feedback(0.0, "User rejected the output.")
        if t.get("accepted"):
            if t.get("edited"):
                return Feedback(self.edit_penalty, "User accepted after editing.")
            return Feedback(1.0, "User accepted the output unchanged.")
        return Feedback(0.0, "No positive user signal.")


@dataclass
class ExecutionResult:
    """The world answered. Reads ``telemetry`` for ``passed`` (bool) or an
    ``exit_code`` / ``status`` — code ran, tests passed, API returned 200."""

    def score(self, interaction: Interaction) -> Feedback:
        t = interaction.telemetry
        if "passed" in t:
            ok = bool(t["passed"])
        elif "exit_code" in t:
            ok = t["exit_code"] == 0
        elif "status" in t:
            ok = 200 <= int(t["status"]) < 300
        else:
            return Feedback(0.0, "No execution signal present.")
        return Feedback(1.0 if ok else 0.0, "Execution succeeded." if ok else "Execution failed.")
