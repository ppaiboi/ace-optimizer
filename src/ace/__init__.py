"""ace — Agentic Context Engineering optimizer (standalone engine).

Public surface (engine core). The DSPy teleprompter wrapper lives separately.
"""

from ace.core.adapter import ACEAdapter, EvaluationBatch, ReflectiveDataset
from ace.engine import optimize
from ace.merge import (
    Add,
    Bump,
    Delete,
    Delta,
    Edit,
    Merge,
    apply_deltas,
    grow_and_refine,
    prune,
)
from ace.online import OnlinePlaybook
from ace.playbook import Bullet, Playbook, id_slug
from ace.result import ACEResult, IterationRecord

__all__ = [
    "Bullet",
    "Playbook",
    "Add",
    "Bump",
    "Edit",
    "Delete",
    "Merge",
    "Delta",
    "apply_deltas",
    "grow_and_refine",
    "prune",
    "id_slug",
    "ACEAdapter",
    "EvaluationBatch",
    "ReflectiveDataset",
    "optimize",
    "ACEResult",
    "IterationRecord",
    "OnlinePlaybook",
]
