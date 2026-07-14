"""Checkpointing for the optimization loop — pause and resume.

A checkpoint is the minimal *resumable* state: the running playbook, the best
playbook so far, the counters, and how many training steps have completed. The
trainset order is a deterministic function of ``seed``, so resuming just means
skipping the first ``step`` items of that order — no need to persist the order
itself.

The heavyweight per-step trace/firehose is **not** persisted (it would bloat the
checkpoint); on resume, ``trace``/``steps`` in the returned result cover only the
post-resume portion. For a complete single trace (e.g. the demo), run to
completion without interrupting.

A ``config_hash`` guards against resuming into an incompatible run.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

from ace.playbook import Playbook
from ace.result import IterationRecord


def config_hash(**parts: object) -> str:
    """Stable hash of the run config; resume only if it matches."""
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass
class Checkpoint:
    config_hash: str
    step: int
    seed_score: float
    best_score: float
    metric_calls: int
    playbook: Playbook
    best_playbook: Playbook
    lessons_buffer: list[str]
    history: list[IterationRecord]

    def to_dict(self) -> dict:
        return {
            "config_hash": self.config_hash,
            "step": self.step,
            "seed_score": self.seed_score,
            "best_score": self.best_score,
            "metric_calls": self.metric_calls,
            "playbook": self.playbook.to_dict(),
            "best_playbook": self.best_playbook.to_dict(),
            "lessons_buffer": self.lessons_buffer,
            "history": [h.__dict__ for h in self.history],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Checkpoint:
        return cls(
            config_hash=d["config_hash"],
            step=d["step"],
            seed_score=d["seed_score"],
            best_score=d["best_score"],
            metric_calls=d["metric_calls"],
            playbook=Playbook.from_dict(d["playbook"]),
            best_playbook=Playbook.from_dict(d["best_playbook"]),
            lessons_buffer=list(d["lessons_buffer"]),
            history=[IterationRecord(**h) for h in d["history"]],
        )


def save_checkpoint(path: str, ckpt: Checkpoint) -> None:
    """Atomically write a checkpoint (write-tmp-then-rename)."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(ckpt.to_dict(), f, indent=2)
    os.replace(tmp, path)


def load_checkpoint(path: str, expected_hash: str | None = None) -> Checkpoint | None:
    """Load a checkpoint, or ``None`` if missing / config-mismatched / corrupt."""
    try:
        with open(path) as f:
            d = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if expected_hash is not None and d.get("config_hash") != expected_hash:
        return None
    return Checkpoint.from_dict(d)
