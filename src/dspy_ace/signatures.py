"""DSPy signatures for the Reflector and Curator steps.

Kept deliberately small and typed with ``list[str]`` outputs so they are easy
to drive with a real LM or a ``DummyLM`` in tests, and easy to parse into
``ace`` deltas.
"""

from __future__ import annotations

import dspy


class Reflect(dspy.Signature):
    """Analyze one execution trace and extract reusable lessons.

    Judge which playbook bullets helped or hurt, and propose concise new
    strategies or pitfalls worth remembering. Use only the feedback provided;
    do not invent labels.
    """

    task: str = dspy.InputField(desc="the input the system was given")
    generated_output: str = dspy.InputField(desc="what the system produced")
    feedback: str = dspy.InputField(desc="score and any error/feedback signal")
    bullets_in_play: str = dspy.InputField(
        desc="the playbook bullets the system had access to"
    )

    bullet_tags: list[str] = dspy.OutputField(
        desc="for bullets that clearly helped or hurt, '<bullet_id>: helpful' "
        "or '<bullet_id>: harmful'; empty list if unclear"
    )
    lessons: list[str] = dspy.OutputField(
        desc="new reusable strategies or pitfalls, one concise item each; "
        "empty list if nothing new was learned"
    )


class Curate(dspy.Signature):
    """Turn reflection lessons into new playbook bullets.

    Only add genuinely new, reusable insights. Do NOT restate bullets already
    present. Assign each to a short section name.
    """

    existing_playbook: str = dspy.InputField(desc="current playbook, may be empty")
    lessons: str = dspy.InputField(desc="candidate lessons from reflection")

    additions: list[str] = dspy.OutputField(
        desc="new bullets, each formatted '<section> :: <content>'; "
        "empty list if nothing should be added"
    )
