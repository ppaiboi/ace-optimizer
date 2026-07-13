"""Deterministic playbook evolution: delta-merge + grow-and-refine.

This is the pure, LLM-free core of ACE. Given a ``Playbook`` and a list of
``Delta`` operations (produced upstream by the Reflector/Curator), these
functions produce a new ``Playbook`` deterministically — no model calls, no
global state, no I/O — so they are exhaustively unit-testable.

Operations:
  * ADD           — append a new bullet to a section (reference-implemented)
  * BUMP          — increment helpful/harmful counter of an existing bullet
  * EDIT          — rewrite the content of an existing bullet in place
  * DELETE        — remove a bullet
  * MERGE         — collapse several bullets into one (counters summed)

The reference implementation (Apache-2.0) only implemented ADD + counter bumps
and left UPDATE/MERGE/DELETE as TODOs; those are built out here.

``grow_and_refine`` performs embedding-based dedup + counter-based pruning. The
embedding function is *injected*, so the whole step is deterministic and can be
tested with a stub embedder (no sentence-transformers/faiss needed).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from ace.playbook import Bullet, Playbook, id_slug, section_slug

# --------------------------------------------------------------------------- #
# Delta types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Add:
    content: str
    section: str = "general"


@dataclass(frozen=True)
class Bump:
    id: str
    tag: str  # "helpful" | "harmful" | "neutral"


@dataclass(frozen=True)
class Edit:
    id: str
    content: str


@dataclass(frozen=True)
class Delete:
    id: str


@dataclass(frozen=True)
class Merge:
    ids: tuple[str, ...]
    content: str
    section: str | None = None  # defaults to the section of the first id


Delta = Add | Bump | Edit | Delete | Merge


# --------------------------------------------------------------------------- #
# Delta application
# --------------------------------------------------------------------------- #


def apply_deltas(playbook: Playbook, deltas: Sequence[Delta]) -> Playbook:
    """Apply deltas in order, returning a new Playbook. Never mutates input.

    Id allocation is monotonic across the batch so two Adds never collide.
    """
    bullets: list[Bullet] = list(playbook.bullets)
    order: list[str] = list(playbook.section_order)
    next_num = playbook.next_num()

    def index_of(bid: str) -> int | None:
        for i, b in enumerate(bullets):
            if b.id == bid:
                return i
        return None

    def ensure_section(sec: str) -> str:
        sec = section_slug(sec)
        if sec not in order:
            order.append(sec)
        return sec

    for d in deltas:
        if isinstance(d, Add):
            sec = ensure_section(d.section)
            new_id = f"{id_slug(sec)}-{next_num:05d}"
            next_num += 1
            bullets.append(Bullet(id=new_id, content=d.content, section=sec))

        elif isinstance(d, Bump):
            i = index_of(d.id)
            if i is None:
                continue  # tolerate stale references, like the reference impl
            b = bullets[i]
            if d.tag == "helpful":
                bullets[i] = replace(b, helpful=b.helpful + 1)
            elif d.tag == "harmful":
                bullets[i] = replace(b, harmful=b.harmful + 1)
            # neutral: no change

        elif isinstance(d, Edit):
            i = index_of(d.id)
            if i is not None:
                bullets[i] = replace(bullets[i], content=d.content)

        elif isinstance(d, Delete):
            i = index_of(d.id)
            if i is not None:
                bullets.pop(i)

        elif isinstance(d, Merge):
            idxs = [i for i in (index_of(x) for x in d.ids) if i is not None]
            if not idxs:
                continue
            members = [bullets[i] for i in idxs]
            sec = ensure_section(d.section or members[0].section)
            merged = Bullet(
                id=members[0].id,  # keep the earliest id as canonical
                content=d.content,
                section=sec,
                helpful=sum(m.helpful for m in members),
                harmful=sum(m.harmful for m in members),
            )
            # Replace the first member in place; drop the rest.
            first = idxs[0]
            bullets[first] = merged
            for i in sorted(idxs[1:], reverse=True):
                bullets.pop(i)

        else:  # pragma: no cover - exhaustiveness guard
            raise TypeError(f"unknown delta: {d!r}")

    return Playbook(bullets=tuple(bullets), section_order=tuple(order))


# --------------------------------------------------------------------------- #
# Grow-and-refine (dedup + prune)
# --------------------------------------------------------------------------- #

EmbedFn = Callable[[Sequence[str]], Sequence[Sequence[float]]]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def grow_and_refine(
    playbook: Playbook,
    embed: EmbedFn,
    *,
    threshold: float = 0.90,
) -> Playbook:
    """Deduplicate near-identical bullets (the reference's grow-and-refine).

    Deterministic given ``embed``. Bullets are scanned in order; each one that
    is within cosine ``threshold`` of an already-kept bullet is folded into it
    (counters SUMMED, earliest id + content kept). This matches the reference's
    ``merge=True`` counter behavior without the reference's LLM content rewrite.

    Pruning is intentionally NOT done here: the reference performs no
    counter-based pruning. Use :func:`prune` explicitly if you want it.
    """
    bullets = list(playbook.bullets)
    if len(bullets) <= 1:
        return playbook

    vecs = list(embed([b.content for b in bullets]))
    keep: list[Bullet] = []
    kept_vecs: list[Sequence[float]] = []
    for b, v in zip(bullets, vecs, strict=False):
        dup_at = None
        for i, kv in enumerate(kept_vecs):
            if _cosine(v, kv) >= threshold:
                dup_at = i
                break
        if dup_at is None:
            keep.append(b)
            kept_vecs.append(v)
        else:
            k = keep[dup_at]
            keep[dup_at] = replace(
                k, helpful=k.helpful + b.helpful, harmful=k.harmful + b.harmful
            )
    return Playbook(bullets=tuple(keep), section_order=playbook.section_order)


def prune(playbook: Playbook, *, min_observations: int = 3) -> Playbook:
    """Drop bullets that have proven net-harmful over enough observations.

    NOT part of the reference implementation — an optional, conservative extra.
    A bullet is dropped only if ``harmful > helpful`` AND it has been exercised
    at least ``min_observations`` times, so fresh or lightly-used bullets are
    never removed on a single bad outcome.
    """
    return Playbook(
        bullets=tuple(
            b
            for b in playbook.bullets
            if not (b.harmful > b.helpful and (b.helpful + b.harmful) >= min_observations)
        ),
        section_order=playbook.section_order,
    )
