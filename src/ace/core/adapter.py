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
    """What a target system must implement for the ACE engine to optimize it."""

    def evaluate(
        self,
        batch: Sequence[Any],
        playbook: Playbook,
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        """Run the system (with ``playbook`` injected as context) over ``batch``.

        Returns per-example outputs + scores (higher is better), plus traces
        when ``capture_traces`` is True. Must never raise on a single-example
        failure — return a fallback score instead.
        """
        ...

    def make_reflective_dataset(
        self, playbook: Playbook, eval_batch: EvaluationBatch
    ) -> ReflectiveDataset:
        """Turn captured traces + feedback into the Reflector's input."""
        ...

    def propose_deltas(
        self, playbook: Playbook, reflective_dataset: ReflectiveDataset
    ) -> list[Delta]:
        """Reflector + Curator: distill reflective data into playbook deltas."""
        ...
