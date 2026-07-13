"""The Curator: turn accumulated lessons into playbook deltas.

Crucially it emits *deltas* (ADD/EDIT/…), never a full rewrite — that's what
prevents context collapse and keeps the merge deterministic. Default impl uses
an injected ``llm(prompt) -> str`` callable; parsing is line-based and lenient.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ace.merge import Add, Delta
from ace.playbook import Playbook


@runtime_checkable
class Curator(Protocol):
    def curate(self, playbook: Playbook, lessons: Sequence[str]) -> list[Delta]: ...


_PROMPT = """You maintain a playbook of reusable strategies as a bulleted list.

EXISTING PLAYBOOK:
{playbook}

NEW LESSONS to fold in:
{lessons}

Propose bullets to ADD. Only add genuinely new, non-duplicate strategies. Format
each on its own line as:
    SECTION :: the strategy, stated imperatively
Use a short UPPER_SNAKE section name that groups related strategies. Output only
the lines, nothing else. If nothing new is worth adding, output nothing."""


def _parse_addition(line: str) -> Add | None:
    line = line.strip(" -*\t")
    if not line:
        return None
    if "::" in line:
        section, _, content = line.partition("::")
        section, content = section.strip(), content.strip()
    else:
        section, content = "general", line
    return Add(content=content, section=section) if content else None


@dataclass
class LLMCurator:
    """Default Curator. ``llm`` is any ``str -> str`` completion callable."""

    llm: Callable[[str], str]

    def curate(self, playbook: Playbook, lessons: Sequence[str]) -> list[Delta]:
        lessons = [ln for ln in lessons if ln]
        if not lessons:
            return []
        prompt = _PROMPT.format(
            playbook=playbook.render().strip() or "(empty)",
            lessons="\n".join(f"- {ln}" for ln in lessons),
        )
        try:
            raw = self.llm(prompt)
        except Exception:  # a flaky model call adds nothing this round
            return []
        deltas: list[Delta] = []
        for line in (raw or "").splitlines():
            d = _parse_addition(line)
            if d:
                deltas.append(d)
        return deltas


@dataclass
class DirectCurator:
    """Adds one bullet per lesson verbatim (no LLM) — for tests/manual use."""

    section: str = "general"

    def curate(self, playbook: Playbook, lessons: Sequence[str]) -> list[Delta]:  # noqa: ARG002
        return [Add(content=ln, section=self.section) for ln in lessons if ln]
