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
    """Master curator of knowledge (ported from the ACE reference curator prompt).

    Your job is to identify what new insights should be added to an existing
    playbook based on a reflection from a previous attempt. The playbook will be
    used to help answer similar questions; the reflection was made with ground
    truth that will NOT be available at use time, so add content that helps the
    playbook user produce answers that align with the ground truth.

    Instructions (verbatim from the paper):
      * Review the existing playbook and the reflection from the previous attempt.
      * Identify ONLY the NEW insights, strategies, or mistakes that are MISSING
        from the current playbook.
      * Avoid redundancy — if similar advice already exists, only add new content
        that is a perfect complement to the existing playbook.
      * Do NOT regenerate the entire playbook — only provide the additions needed.
      * Focus on quality over quantity — a focused, well-organized playbook is
        better than an exhaustive one.
      * If there is no new content to add, return nothing.
      * Be concise and specific — each addition should be actionable.
      * Respect the token budget shown in the stats; as it fills, be more selective.
    """

    existing_playbook: str = dspy.InputField(
        desc="current playbook with bullet ids and helpful/harmful counters"
    )
    playbook_stats: str = dspy.InputField(
        desc="size of the playbook so far vs. the token budget (stay within it)"
    )
    lessons: str = dspy.InputField(desc="candidate lessons (the recent reflection)")

    operations: str = dspy.OutputField(
        desc=(
            "the additions to make, ONE PER LINE as 'SECTION :: content' — new, "
            "missing, non-redundant strategies only. Empty if nothing new to add. "
            "(You may also consolidate a true duplicate with 'MERGE <id1>,<id2> :: "
            "content' or drop a clearly-wrong bullet with 'DELETE <id>', but ADD "
            "is the norm.)"
        )
    )
