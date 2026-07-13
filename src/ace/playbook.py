"""Playbook data model + (de)serialization.

The on-disk / on-the-wire representation is line-structured, interop-compatible
with the reference ACE implementation (github.com/ace-agent/ace, Apache-2.0):

    ## SECTION NAME
    [section-00042] helpful=3 harmful=1 :: <strategy content>

Internally we keep structured ``Bullet`` / ``Playbook`` dataclasses so the
delta operations in ``ace.merge`` are pure and testable, and only touch strings
at the parse/render boundary.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

# [id] helpful=X harmful=Y :: content
_LINE_RE = re.compile(r"\[([^\]]+)\]\s*helpful=(\d+)\s*harmful=(\d+)\s*::\s*(.*)")
_SECTION_RE = re.compile(r"^##\s*(.+?)\s*$")
_ID_SUFFIX_RE = re.compile(r"-(\d+)$")


def section_slug(section: str) -> str:
    """Normalize a section name (used for section identity / headers)."""
    return section.strip().lower().replace(" ", "_").replace("&", "and")


# Fixed abbreviations from the reference (ace-upstream utils.get_section_slug).
_ID_SLUG_MAP = {
    "financial_strategies_and_insights": "fin",
    "formulas_and_calculations": "calc",
    "code_snippets_and_templates": "code",
    "common_mistakes_to_avoid": "err",
    "problem_solving_heuristics": "prob",
    "context_clues_and_indicators": "ctx",
    "others": "misc",
    "meta_strategies": "meta",
}


def id_slug(section: str) -> str:
    """Abbreviated, letters-only slug used as the *id prefix* for a section.

    Mirrors the reference so ids look like ``calc-00042`` and satisfy the
    generator's citation regex ``[a-z]{3,}-\\d{5}``. Distinct from
    ``section_slug``, which keeps the full name for section identity.
    """
    clean = section_slug(section)
    if clean in _ID_SLUG_MAP:
        return _ID_SLUG_MAP[clean]
    words = [w for w in clean.split("_") if w]
    base = words[0][:4] if len(words) == 1 else "".join(w[:1] for w in words[:5])
    base = re.sub(r"[^a-z]", "", base)
    return base or "misc"


@dataclass(frozen=True)
class Bullet:
    """A single playbook entry. Immutable; edits produce a new Bullet."""

    id: str
    content: str
    section: str = "general"
    helpful: int = 0
    harmful: int = 0

    def render(self) -> str:
        return f"[{self.id}] helpful={self.helpful} harmful={self.harmful} :: {self.content}"

    @property
    def num(self) -> int:
        """Numeric suffix of the id (0 if none), used for id allocation."""
        m = _ID_SUFFIX_RE.search(self.id)
        return int(m.group(1)) if m else 0


@dataclass(frozen=True)
class Playbook:
    """An ordered collection of bullets grouped into sections.

    Section order is preserved; within a section, bullet order is preserved.
    """

    bullets: tuple[Bullet, ...] = ()
    # Declared section order (slugs). Sections seen in bullets are appended.
    section_order: tuple[str, ...] = ()

    # ---- construction / parsing -------------------------------------------------

    @classmethod
    def parse(cls, text: str) -> Playbook:
        bullets: list[Bullet] = []
        order: list[str] = []
        current = "general"
        for raw in text.strip().splitlines():
            line = raw.strip()
            if not line:
                continue
            sec = _SECTION_RE.match(line)
            if sec:
                current = section_slug(sec.group(1))
                if current not in order:
                    order.append(current)
                continue
            m = _LINE_RE.match(line)
            if m:
                if current not in order:
                    order.append(current)
                bullets.append(
                    Bullet(
                        id=m.group(1),
                        helpful=int(m.group(2)),
                        harmful=int(m.group(3)),
                        content=m.group(4),
                        section=current,
                    )
                )
        return cls(bullets=tuple(bullets), section_order=tuple(order))

    # ---- rendering --------------------------------------------------------------

    def render(self) -> str:
        out: list[str] = []
        for sec in self._ordered_sections():
            out.append(f"## {sec.upper()}")
            for b in self.bullets:
                if b.section == sec:
                    out.append(b.render())
            out.append("")
        return "\n".join(out).rstrip() + "\n" if out else ""

    def _ordered_sections(self) -> list[str]:
        seen = list(self.section_order)
        for b in self.bullets:
            if b.section not in seen:
                seen.append(b.section)
        return seen

    # ---- lookups ----------------------------------------------------------------

    def by_id(self, bullet_id: str) -> Bullet | None:
        for b in self.bullets:
            if b.id == bullet_id:
                return b
        return None

    def next_num(self) -> int:
        """Next numeric id suffix (max existing + 1)."""
        return (max((b.num for b in self.bullets), default=0)) + 1

    # ---- JSON -------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "section_order": list(self.section_order),
            "bullets": [
                {
                    "id": b.id,
                    "content": b.content,
                    "section": b.section,
                    "helpful": b.helpful,
                    "harmful": b.harmful,
                }
                for b in self.bullets
            ],
        }

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), **kw)

    @classmethod
    def from_dict(cls, d: dict) -> Playbook:
        bullets = tuple(
            Bullet(
                id=b["id"],
                content=b["content"],
                section=b.get("section", "general"),
                helpful=int(b.get("helpful", 0)),
                harmful=int(b.get("harmful", 0)),
            )
            for b in d.get("bullets", [])
        )
        return cls(bullets=bullets, section_order=tuple(d.get("section_order", ())))
