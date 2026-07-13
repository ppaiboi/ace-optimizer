"""The ACE Generator: a citing predictor.

Faithful to the reference implementation, the Generator receives the playbook
as an explicit input and must cite the ``bullet_ids`` it actually relied on.
Those citations are what make counter attribution meaningful — only cited
bullets can be tagged helpful/harmful by the Reflector.

We build the generator by augmenting the student's own task signature: prepend
a ``playbook`` input and append a ``bullet_ids`` output. This keeps the task
(its input/output fields) intact while adding the ACE machinery around it.
"""

from __future__ import annotations

import dspy

_PLAYBOOK_DESC = (
    "Accumulated strategy playbook (bullets with ids like 'calc-00003'). "
    "Use the relevant strategies; ignore irrelevant ones. May be empty."
)
_CITE_DESC = (
    "List of the playbook bullet ids you actually relied on to answer "
    "(e.g. ['calc-00003']). Empty list if the playbook did not help."
)
_REFLECTION_DESC = (
    "Feedback from a previous failed attempt at this task, if any. Use it to "
    "correct your approach. '(empty)' on the first attempt."
)


def build_ace_predictor(base_signature) -> dspy.Predict:
    """Augment a task signature with playbook + reflection inputs and a
    bullet_ids output (matching the reference generator's interface)."""
    sig = base_signature.prepend(
        "reflection", dspy.InputField(desc=_REFLECTION_DESC), type_=str
    )
    sig = sig.prepend(
        "playbook", dspy.InputField(desc=_PLAYBOOK_DESC), type_=str
    )
    sig = sig.append(
        "bullet_ids", dspy.OutputField(desc=_CITE_DESC), type_=list[str]
    )
    return dspy.Predict(sig)


class ACEGenerator(dspy.Module):
    """A task program with a bound playbook, exposing bullet citations.

    ``forward(**task_inputs)`` injects the current playbook (and optionally a
    ``reflection`` string for a retry) and returns the task prediction plus
    ``bullet_ids``.
    """

    def __init__(self, predictor: dspy.Predict, playbook_text: str) -> None:
        super().__init__()
        self.predictor = predictor
        self.playbook_text = playbook_text or "(empty)"

    def forward(self, reflection: str = "(empty)", **task_inputs):
        return self.predictor(
            playbook=self.playbook_text, reflection=reflection, **task_inputs
        )
