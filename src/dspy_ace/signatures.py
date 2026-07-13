"""DSPy signatures for the Reflector and Curator steps.

Outputs are plain strings (not ``list[str]``) with a documented line format,
then parsed leniently in the adapter. Typed-list outputs proved fragile across
models (some LMs format lists in ways dspy's adapter rejects, crashing the
run); string outputs + our own parsing are robust.
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

    bullet_tags: str = dspy.OutputField(
        desc="one per line, 'BULLET_ID: helpful' or 'BULLET_ID: harmful', for "
        "bullets that clearly helped or hurt; empty if unclear"
    )
    lessons: str = dspy.OutputField(
        desc="new reusable strategies or pitfalls, ONE PER LINE, concise; "
        "empty if nothing new was learned"
    )


class Curate(dspy.Signature):
    """Turn reflection lessons into new playbook bullets.

    Only add genuinely new, reusable insights. Do NOT restate bullets already
    present. Assign each to a short section name.
    """

    existing_playbook: str = dspy.InputField(desc="current playbook, may be empty")
    lessons: str = dspy.InputField(desc="candidate lessons from reflection")

    additions: str = dspy.OutputField(
        desc="new bullets, ONE PER LINE, each formatted 'section :: content'; "
        "empty if nothing should be added"
    )
