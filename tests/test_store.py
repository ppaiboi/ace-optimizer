"""Event-sourced store: fold, versioning, rollback, quarantine, persistence."""

from __future__ import annotations

from ace import (
    Add,
    Bump,
    Delete,
    InMemoryPlaybookStore,
    JSONPlaybookStore,
    SQLitePlaybookStore,
    delta_from_dict,
    delta_to_dict,
)
from ace.merge import Merge


def test_delta_serde_roundtrip():
    for d in [Add("x", "sec"), Bump("sec-00001", "helpful"), Delete("sec-00002"),
              Merge(ids=("a-00001", "a-00002"), content="merged", section="s")]:
        assert delta_from_dict(delta_to_dict(d)) == d


def test_playbook_is_fold_over_commits():
    s = InMemoryPlaybookStore()
    assert s.version() == 0
    assert len(s.head().bullets) == 0

    v1 = s.append([Add("first strategy", "general")])
    v2 = s.append([Add("second strategy", "general")])
    assert (v1, v2) == (1, 2)
    assert len(s.head().bullets) == 2
    # time-travel: version 1 has only the first bullet
    assert len(s.playbook_at(1).bullets) == 1
    assert "first" in s.playbook_at(1).bullets[0].content


def test_rollback_moves_head_and_is_reversible():
    s = InMemoryPlaybookStore()
    s.append([Add("a", "general")])
    s.append([Add("b", "general")])
    assert len(s.head().bullets) == 2
    s.rollback(1)
    assert s.version() == 1
    assert len(s.head().bullets) == 1
    # appending after a rollback forks from the rolled-back head
    s.append([Add("c", "general")])
    contents = [b.content for b in s.head().bullets]
    assert contents == ["a", "c"]  # 'b' was abandoned


def test_quarantine_records_rejected_deltas():
    s = InMemoryPlaybookStore()
    s.quarantine([Add("bad", "general")], reason="regressed on holdout")
    q = s.quarantined()
    assert len(q) == 1 and q[0]["reason"] == "regressed on holdout"
    assert s.version() == 0  # nothing promoted


def test_json_store_persists(tmp_path):
    p = str(tmp_path / "pb.json")
    s = JSONPlaybookStore(p)
    s.append([Add("persisted", "general")])
    s.append([Bump(s.head().bullets[0].id, "helpful")])
    # reopen from disk
    s2 = JSONPlaybookStore(p)
    assert len(s2.head().bullets) == 1
    assert s2.head().bullets[0].helpful == 1
    assert s2.version() == 2


def test_sqlite_store_persists_and_namespaces(tmp_path):
    db = str(tmp_path / "pb.db")
    a = SQLitePlaybookStore(db, namespace="team-a")
    b = SQLitePlaybookStore(db, namespace="team-b")
    a.append([Add("a-strategy", "general")])
    b.append([Add("b-strategy", "general")])
    a.close()
    b.close()
    # reopen: each namespace is isolated and durable
    a2 = SQLitePlaybookStore(db, namespace="team-a")
    assert [x.content for x in a2.head().bullets] == ["a-strategy"]
    assert a2.version() == 1
    a2.close()
