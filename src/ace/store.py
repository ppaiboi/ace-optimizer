"""Event-sourced playbook storage.

The **delta log is the source of truth** — the playbook at version *n* is a
deterministic fold over the first *n* commits. That buys audit trail, rollback,
and time-travel debugging ("why is this rule in my prompt?" → the commit that
added it) essentially for free, and makes concurrency safe (append-only).

A ``PlaybookStore`` keeps, per ``namespace``:
  * an append-only list of **commits** (each a batch of deltas + metadata),
  * a movable ``head`` version pointer (so rollback is O(1) and itself audited),
  * a **quarantine** of rejected delta batches (see ``ace.gate``).

``head()`` returns the live ``Playbook``. Three implementations share one
protocol: ``InMemoryPlaybookStore`` (tests), ``JSONPlaybookStore`` (a file),
and ``SQLitePlaybookStore`` (the production default).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ace.merge import Delta, apply_deltas, delta_from_dict, delta_to_dict
from ace.playbook import Playbook


@dataclass(frozen=True)
class Commit:
    """One promoted batch of deltas — the unit of versioning."""

    version: int
    deltas: tuple[Delta, ...]
    meta: dict = field(default_factory=dict)  # e.g. reason, trace_ids, val_score, ts

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "deltas": [delta_to_dict(d) for d in self.deltas],
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Commit:
        return cls(
            version=d["version"],
            deltas=tuple(delta_from_dict(x) for x in d["deltas"]),
            meta=d.get("meta", {}),
        )


@runtime_checkable
class PlaybookStore(Protocol):
    """Persistence seam for the event-sourced playbook."""

    def head(self) -> Playbook: ...
    def version(self) -> int: ...
    def append(self, deltas: list[Delta], meta: dict | None = None) -> int: ...
    def playbook_at(self, version: int) -> Playbook: ...
    def commits(self) -> list[Commit]: ...
    def rollback(self, to_version: int, *, reason: str = "") -> int: ...
    def quarantine(self, deltas: list[Delta], reason: str) -> None: ...
    def quarantined(self) -> list[dict]: ...


class _BaseStore:
    """Shared fold/rollback logic; subclasses provide persistence hooks."""

    _commits: list[Commit]
    _head: int
    _quarantine: list[dict]

    # -- reads ---------------------------------------------------------------
    def version(self) -> int:
        return self._head

    def commits(self) -> list[Commit]:
        return list(self._commits)

    def quarantined(self) -> list[dict]:
        return list(self._quarantine)

    def playbook_at(self, version: int) -> Playbook:
        if version < 0:
            raise ValueError("version must be >= 0")
        pb = Playbook()
        for c in self._commits:
            if c.version > version:
                break
            pb = apply_deltas(pb, list(c.deltas))
        return pb

    def head(self) -> Playbook:
        return self.playbook_at(self._head)

    # -- writes --------------------------------------------------------------
    def append(self, deltas: list[Delta], meta: dict | None = None) -> int:
        if not deltas:
            return self._head
        # Appending only makes sense from the tip; a rolled-back head that then
        # appends forks history — we drop the abandoned tail (kept in the log is
        # cleaner, but O(1) truncation keeps versions contiguous).
        if self._head < len(self._commits):
            self._commits = self._commits[: self._head]
        version = self._head + 1
        self._commits.append(Commit(version=version, deltas=tuple(deltas), meta=meta or {}))
        self._head = version
        self._persist()
        return version

    def rollback(self, to_version: int, *, reason: str = "") -> int:
        if not (0 <= to_version <= self._head):
            raise ValueError(f"cannot roll back to {to_version} (head={self._head})")
        self._head = to_version
        self._persist()
        return self._head

    def quarantine(self, deltas: list[Delta], reason: str) -> None:
        self._quarantine.append(
            {"deltas": [delta_to_dict(d) for d in deltas], "reason": reason}
        )
        self._persist()

    # -- persistence hook ----------------------------------------------------
    def _persist(self) -> None:  # overridden by durable stores
        pass


class InMemoryPlaybookStore(_BaseStore):
    """Ephemeral store — ideal for tests and single-process experiments."""

    def __init__(self) -> None:
        self._commits = []
        self._head = 0
        self._quarantine = []


class JSONPlaybookStore(_BaseStore):
    """A single JSON file. Human-inspectable; fine for small/dev deployments."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._commits = []
        self._head = 0
        self._quarantine = []
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path) as f:
                d = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return
        self._commits = [Commit.from_dict(c) for c in d.get("commits", [])]
        self._head = d.get("head", len(self._commits))
        self._quarantine = d.get("quarantine", [])

    def _persist(self) -> None:
        tmp = f"{self.path}.tmp"
        with open(tmp, "w") as f:
            json.dump(
                {
                    "commits": [c.to_dict() for c in self._commits],
                    "head": self._head,
                    "quarantine": self._quarantine,
                },
                f,
                indent=2,
            )
        import os

        os.replace(tmp, self.path)


class SQLitePlaybookStore(_BaseStore):
    """The production default: durable, concurrent-safe append-only log.

    One row per commit; a tiny ``meta`` table holds the head pointer. Namespaced
    so many playbooks can share one database file.
    """

    def __init__(self, path: str, *, namespace: str = "default") -> None:
        self.path = path
        self.namespace = namespace
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS commits ("
            " namespace TEXT, version INTEGER, deltas TEXT, meta TEXT,"
            " PRIMARY KEY (namespace, version))"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS state ("
            " namespace TEXT PRIMARY KEY, head INTEGER)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS quarantine ("
            " namespace TEXT, deltas TEXT, reason TEXT)"
        )
        self._conn.commit()
        self._commits = []
        self._head = 0
        self._quarantine = []
        self._load()

    def _load(self) -> None:
        rows = self._conn.execute(
            "SELECT version, deltas, meta FROM commits WHERE namespace=? ORDER BY version",
            (self.namespace,),
        ).fetchall()
        self._commits = [
            Commit(version=v, deltas=tuple(delta_from_dict(x) for x in json.loads(ds)),
                   meta=json.loads(ms))
            for v, ds, ms in rows
        ]
        r = self._conn.execute(
            "SELECT head FROM state WHERE namespace=?", (self.namespace,)
        ).fetchone()
        self._head = r[0] if r else len(self._commits)
        self._quarantine = [
            {"deltas": json.loads(ds), "reason": rs}
            for ds, rs in self._conn.execute(
                "SELECT deltas, reason FROM quarantine WHERE namespace=?", (self.namespace,)
            ).fetchall()
        ]

    def _persist(self) -> None:
        # Rewrite this namespace's rows to match in-memory state (small tables).
        c = self._conn
        c.execute("DELETE FROM commits WHERE namespace=?", (self.namespace,))
        c.executemany(
            "INSERT INTO commits VALUES (?,?,?,?)",
            [(self.namespace, cm.version, json.dumps([delta_to_dict(d) for d in cm.deltas]),
              json.dumps(cm.meta)) for cm in self._commits],
        )
        c.execute(
            "INSERT INTO state VALUES (?,?) ON CONFLICT(namespace) DO UPDATE SET head=?",
            (self.namespace, self._head, self._head),
        )
        c.execute("DELETE FROM quarantine WHERE namespace=?", (self.namespace,))
        c.executemany(
            "INSERT INTO quarantine VALUES (?,?,?)",
            [(self.namespace, json.dumps(q["deltas"]), q["reason"]) for q in self._quarantine],
        )
        c.commit()

    def close(self) -> None:
        self._conn.close()
