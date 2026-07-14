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
    """Maintain the playbook by proposing edit operations.

    You are the *curator* of a living playbook: fold in new lessons, but also
    keep it lean and non-redundant. Do not only add — actively prune. If a
    lesson duplicates an existing bullet, DELETE or MERGE the duplicates rather
    than adding another. If an existing bullet is wrong, obsolete, or superseded,
    DELETE it. If it is close but imprecise, UPDATE it. Reference existing
    bullets by their exact id (e.g. calc-00042) shown in the playbook.
    """

    existing_playbook: str = dspy.InputField(
        desc="current playbook with bullet ids and helpful/harmful counters"
    )
    lessons: str = dspy.InputField(desc="candidate lessons from reflection")

    operations: str = dspy.OutputField(
        desc=(
            "playbook edit operations, ONE PER LINE, each exactly one of:\n"
            "  ADD SECTION :: content        (a genuinely new strategy)\n"
            "  UPDATE <id> :: new content    (refine an existing bullet)\n"
            "  DELETE <id>                   (remove a redundant/obsolete bullet)\n"
            "  MERGE <id1>,<id2>,... :: content   (collapse duplicates into one)\n"
            "Prefer UPDATE/DELETE/MERGE over piling on ADDs. Empty if no change."
        )
    )
