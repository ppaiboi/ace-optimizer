"""Deterministic-core tests: parsing, delta application, grow-and-refine."""

from __future__ import annotations

import pytest

from ace import (
    Add,
    Bullet,
    Bump,
    Delete,
    Edit,
    Merge,
    Playbook,
    apply_deltas,
    grow_and_refine,
    id_slug,
    prune,
)

# --------------------------------------------------------------------------- #
# Round-trip parse/render
# --------------------------------------------------------------------------- #

SAMPLE = """
## STRATEGIES
[strategies-00001] helpful=3 harmful=1 :: Always validate inputs first.
[strategies-00002] helpful=0 harmful=0 :: Prefer explicit over implicit.

## OTHERS
[others-00005] helpful=2 harmful=0 :: Log failures with context.
"""


def test_parse_extracts_bullets_and_sections():
    pb = Playbook.parse(SAMPLE)
    assert len(pb.bullets) == 3
    b = pb.by_id("strategies-00001")
    assert b is not None
    assert b.helpful == 3 and b.harmful == 1
    assert b.section == "strategies"
    assert pb.section_order == ("strategies", "others")


def test_render_roundtrip_is_stable():
    pb = Playbook.parse(SAMPLE)
    reparsed = Playbook.parse(pb.render())
    assert reparsed.to_dict() == pb.to_dict()


def test_json_roundtrip():
    pb = Playbook.parse(SAMPLE)
    assert Playbook.from_dict(pb.to_dict()).to_dict() == pb.to_dict()


def test_next_num_is_max_plus_one():
    assert Playbook.parse(SAMPLE).next_num() == 6


# --------------------------------------------------------------------------- #
# ADD
# --------------------------------------------------------------------------- #


def test_add_allocates_monotonic_ids_without_collision():
    pb = Playbook.parse(SAMPLE)
    out = apply_deltas(pb, [Add("New A", "strategies"), Add("New B", "strategies")])
    new_ids = [b.id for b in out.bullets if b.content in ("New A", "New B")]
    # id prefix is the abbreviated slug (reference: get_section_slug), not full name
    assert new_ids == ["stra-00006", "stra-00007"]


def test_id_slug_matches_reference_abbreviation():
    assert id_slug("formulas_and_calculations") == "calc"  # mapped
    assert id_slug("Common Mistakes to Avoid") == "err"  # mapped, normalized
    assert id_slug("geography") == "geog"  # single word -> first 4
    assert id_slug("Error Handling Notes") == "ehn"  # multi word -> initials
    import re as _re

    assert _re.fullmatch(r"[a-z]+", id_slug("strategies"))  # letters only


def test_add_creates_new_section_when_absent():
    pb = Playbook.parse(SAMPLE)
    out = apply_deltas(pb, [Add("Edge case note", "Error Handling")])
    added = out.bullets[-1]
    assert added.section == "error_handling"
    assert "error_handling" in out.section_order


def test_apply_does_not_mutate_input():
    pb = Playbook.parse(SAMPLE)
    before = pb.to_dict()
    apply_deltas(pb, [Add("x"), Delete("strategies-00001")])
    assert pb.to_dict() == before


# --------------------------------------------------------------------------- #
# BUMP / EDIT / DELETE
# --------------------------------------------------------------------------- #


def test_bump_counters():
    pb = Playbook.parse(SAMPLE)
    out = apply_deltas(
        pb,
        [
            Bump("strategies-00001", "helpful"),
            Bump("strategies-00001", "harmful"),
            Bump("strategies-00002", "neutral"),
        ],
    )
    b1 = out.by_id("strategies-00001")
    b2 = out.by_id("strategies-00002")
    assert (b1.helpful, b1.harmful) == (4, 2)
    assert (b2.helpful, b2.harmful) == (0, 0)


def test_bump_unknown_id_is_ignored():
    pb = Playbook.parse(SAMPLE)
    out = apply_deltas(pb, [Bump("does-not-exist", "helpful")])
    assert out.to_dict() == pb.to_dict()


def test_edit_replaces_content_only():
    pb = Playbook.parse(SAMPLE)
    out = apply_deltas(pb, [Edit("strategies-00002", "Rewritten.")])
    b = out.by_id("strategies-00002")
    assert b.content == "Rewritten." and b.helpful == 0


def test_delete_removes_bullet():
    pb = Playbook.parse(SAMPLE)
    out = apply_deltas(pb, [Delete("others-00005")])
    assert out.by_id("others-00005") is None
    assert len(out.bullets) == 2


# --------------------------------------------------------------------------- #
# MERGE
# --------------------------------------------------------------------------- #


def test_merge_sums_counters_and_keeps_first_id():
    pb = Playbook.parse(SAMPLE)
    out = apply_deltas(
        pb,
        [Merge(("strategies-00001", "others-00005"), "Consolidated rule.")],
    )
    assert out.by_id("others-00005") is None
    m = out.by_id("strategies-00001")
    assert m.content == "Consolidated rule."
    assert (m.helpful, m.harmful) == (5, 1)  # 3+2, 1+0
    assert len(out.bullets) == 2


# --------------------------------------------------------------------------- #
# grow_and_refine
# --------------------------------------------------------------------------- #


def _stub_embedder(vectors: dict[str, list[float]]):
    """Return an embed fn mapping exact content -> vector (default orthogonal)."""

    def embed(contents):
        out = []
        for i, c in enumerate(contents):
            out.append(vectors.get(c, [0.0] * i + [1.0] + [0.0] * 8))
        return out

    return embed


def test_dedup_folds_duplicate_into_earlier_and_sums_counters():
    pb = Playbook(
        bullets=(
            Bullet("s-1", "validate inputs", "s", helpful=2, harmful=0),
            Bullet("s-2", "validate inputs", "s", helpful=1, harmful=1),
            Bullet("s-3", "log errors", "s", helpful=0, harmful=0),
        )
    )
    embed = _stub_embedder(
        {"validate inputs": [1.0, 0.0], "log errors": [0.0, 1.0]}
    )
    out = grow_and_refine(pb, embed, threshold=0.9)
    assert [b.id for b in out.bullets] == ["s-1", "s-3"]
    folded = out.by_id("s-1")
    assert (folded.helpful, folded.harmful) == (3, 1)


def test_grow_and_refine_does_not_prune():
    # faithful to reference: dedup never drops a net-harmful bullet
    pb = Playbook(bullets=(Bullet("s-2", "bad", "s", helpful=0, harmful=3),))
    embed = _stub_embedder({"bad": [1.0, 0.0]})
    assert grow_and_refine(pb, embed).to_dict() == pb.to_dict()


def test_prune_drops_net_harmful_only_after_min_observations():
    pb = Playbook(
        bullets=(
            Bullet("s-1", "good", "s", helpful=5, harmful=1),  # net-helpful: keep
            Bullet("s-2", "bad", "s", helpful=0, harmful=3),  # net-harmful, obs=3: drop
            Bullet("s-3", "fresh", "s", helpful=0, harmful=1),  # net-harmful but obs=1: keep
        )
    )
    out = prune(pb, min_observations=3)
    assert {b.id for b in out.bullets} == {"s-1", "s-3"}


def test_grow_and_refine_is_deterministic():
    pb = Playbook.parse(SAMPLE)
    embed = _stub_embedder({})
    a = grow_and_refine(pb, embed).to_dict()
    b = grow_and_refine(pb, embed).to_dict()
    assert a == b


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
