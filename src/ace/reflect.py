"""The Reflector: diagnose an interaction into reusable lessons.

Protocol + a default LLM-backed implementation. The model is an injected
``llm(prompt) -> str`` callable, so this stays framework-agnostic and testable
with a fake (no provider SDK, no network).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ace.playbook import Playbook
from ace.signals import Feedback, Interaction


@dataclass(frozen=True)
class Reflection:
    """What the Reflector concluded: transferable lessons for the Curator."""

    lessons: tuple[str, ...] = ()
    notes: str = ""


@runtime_checkable
class Reflector(Protocol):
    def reflect(
        self, interaction: Interaction, feedback: Feedback, playbook: Playbook
    ) -> Reflection: ...


_PROMPT = """You are diagnosing an AI system's behavior to extract reusable lessons.

CURRENT PLAYBOOK (strategies already known):
{playbook}

WHAT HAPPENED
- Input: {input}
- Output: {output}
- Evaluation ({score:.2f}): {feedback}

List concrete, transferable lessons that would improve future outputs on
*similar* inputs. One lesson per line, imperative voice, no numbering. If the
output was already good, say what made it work. Do not restate the input."""


@dataclass
class LLMReflector:
    """Default Reflector. ``llm`` is any ``str -> str`` completion callable."""

    llm: Callable[[str], str]
    max_chars: int = 1500

    def reflect(
        self, interaction: Interaction, feedback: Feedback, playbook: Playbook
    ) -> Reflection:
        prompt = _PROMPT.format(
            playbook=playbook.render().strip() or "(empty)",
            input=str(interaction.input)[: self.max_chars],
            output=str(interaction.output)[: self.max_chars],
            score=feedback.score,
            feedback=feedback.text,
        )
        try:
            raw = self.llm(prompt)
        except Exception:  # a flaky model call must not break the learning path
            return Reflection()
        lessons = tuple(
            ln.strip(" -*\t") for ln in (raw or "").splitlines() if ln.strip(" -*\t")
        )
        return Reflection(lessons=lessons, notes=raw or "")


@dataclass
class NoopReflector:
    """Turns a single free-text lesson straight through (for tests/manual use)."""

    lessons_field: str = "lesson"

    def reflect(
        self, interaction: Interaction, feedback: Feedback, playbook: Playbook
    ) -> Reflection:  # noqa: ARG002
        lesson = interaction.meta.get(self.lessons_field)
        return Reflection(lessons=(lesson,) if lesson else ())
