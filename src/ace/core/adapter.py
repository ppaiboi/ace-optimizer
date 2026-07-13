"""The adapter contract between the ACE engine and a concrete system (e.g. DSPy).

Mirrors GEPA's separation of concerns: the engine treats scores as opaque
floats and traces as fully opaque, and only ever touches the target system
through this Protocol. The DSPy integration lives in ``dspy_ace`` and never
leaks into the engine.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ace.merge import Delta
from ace.playbook import Playbook


@dataclass
class EvaluationBatch:
    """Result of running a candidate playbook over a batch of inputs."""

    outputs: list[Any]
    scores: list[float]
    # Opaque per-example traces; only the adapter's make_reflective_dataset reads them.
    trajectories: list[Any] | None = None

    def __post_init__(self) -> None:
        if len(self.outputs) != len(self.scores):
            raise ValueError("outputs and scores must be the same length")

    @property
    def mean_score(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0


# A per-component reflective dataset: component name -> list of JSON-ish records.
ReflectiveDataset = Mapping[str, Sequence[Mapping[str, Any]]]


@runtime_checkable
class ACEAdapter(Protocol):
    """What a target system must implement for the ACE engine to optimize it.

    ``evaluate`` scores a whole batch (used for validation); ``generate_one`` /
    ``reflect_one`` / ``curate`` are the per-sample Generator / Reflector /
    Curator primitives the optimization loop drives.
    """

    def evaluate(
        self,
        batch: Sequence[Any],
        playbook: Playbook,
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        """Run the system (with ``playbook`` injected) over ``batch``.

        Returns per-example outputs + scores (higher is better). Must never
        raise on a single-example failure — return a fallback score instead.
        """
        ...

    def generate_one(
        self, sample: Any, playbook: Playbook, reflection: str = "(empty)"
    ) -> dict:
        """Generator: one generation on a single sample, optionally retrying
        with a ``reflection`` hint. Returns
        ``{pred, score, feedback, cited}``."""
        ...

    def reflect_one(self, sample: Any, gen: dict, playbook: Playbook) -> dict:
        """Reflector: tag cited bullets + extract lessons + a retry hint.
        Returns ``{tags: list[Delta], lessons: list[str], reflection_text: str}``."""
        ...

    def curate(self, playbook: Playbook, lessons: Sequence[str]) -> list[Delta]:
        """Curator: author new-bullet deltas from accumulated lessons."""
        ...
