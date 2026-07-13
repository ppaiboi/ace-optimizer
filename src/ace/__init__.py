"""ace — Agentic Context Engineering optimizer (standalone engine).

Public surface (engine core). The DSPy teleprompter wrapper lives separately.
"""

from ace.core.adapter import ACEAdapter, EvaluationBatch, ReflectiveDataset
from ace.curate import Curator, DirectCurator, LLMCurator
from ace.engine import optimize
from ace.facade import ACE
from ace.gate import Decision, PromotionPolicy, evaluate_candidate, promote
from ace.merge import (
    Add,
    Bump,
    Delete,
    Delta,
    Edit,
    Merge,
    apply_deltas,
    delta_from_dict,
    delta_to_dict,
    grow_and_refine,
    prune,
)
from ace.online import OnlinePlaybook
from ace.playbook import Bullet, Playbook, id_slug
from ace.reflect import LLMReflector, NoopReflector, Reflection, Reflector
from ace.result import (
    ACEResult,
    BulletEdit,
    GenAttempt,
    IterationRecord,
    LMCall,
    ReflectRecord,
    StepRecord,
    TraceCheckpoint,
)
from ace.signals import (
    ExecutionResult,
    Feedback,
    GroundTruth,
    ImplicitUser,
    Interaction,
    LLMJudge,
    Signal,
)
from ace.store import (
    Commit,
    InMemoryPlaybookStore,
    JSONPlaybookStore,
    PlaybookStore,
    SQLitePlaybookStore,
)

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
    "TraceCheckpoint",
    "StepRecord",
    "GenAttempt",
    "ReflectRecord",
    "LMCall",
    "BulletEdit",
    "OnlinePlaybook",
    # framework-agnostic library surface
    "ACE",
    "delta_to_dict",
    "delta_from_dict",
    "PlaybookStore",
    "InMemoryPlaybookStore",
    "JSONPlaybookStore",
    "SQLitePlaybookStore",
    "Commit",
    "Signal",
    "Feedback",
    "Interaction",
    "GroundTruth",
    "LLMJudge",
    "ImplicitUser",
    "ExecutionResult",
    "Reflector",
    "Reflection",
    "LLMReflector",
    "NoopReflector",
    "Curator",
    "LLMCurator",
    "DirectCurator",
    "PromotionPolicy",
    "Decision",
    "promote",
    "evaluate_candidate",
]
