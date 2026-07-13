"""Signals, promotion gate, and the ACE facade — all hermetic (fake LLMs)."""

from __future__ import annotations

import pytest

from ace import (
    ACE,
    Add,
    DirectCurator,
    ExecutionResult,
    Feedback,
    GroundTruth,
    ImplicitUser,
    InMemoryPlaybookStore,
    Interaction,
    LLMJudge,
    NoopReflector,
    Playbook,
    PromotionPolicy,
    promote,
)

# -- signals ---------------------------------------------------------------

def test_ground_truth_signal():
    sig = GroundTruth(metric=lambda gold, out: 1.0 if gold == out else 0.0)
    assert sig.score(Interaction(input="q", output="a", gold="a")).score == 1.0
    assert sig.score(Interaction(input="q", output="b", gold="a")).score == 0.0


def test_llm_judge_signal_uses_injected_judge():
    sig = LLMJudge(judge=lambda inter, rubric: (0.75, "mostly right"))
    fb = sig.score(Interaction(input="q", output="a"))
    assert fb.score == 0.75 and "mostly" in fb.text


def test_implicit_user_signal():
    sig = ImplicitUser(edit_penalty=0.5)
    assert sig.score(Interaction("q", "a", telemetry={"accepted": True})).score == 1.0
    assert sig.score(Interaction("q", "a", telemetry={"accepted": True, "edited": True})).score == 0.5
    assert sig.score(Interaction("q", "a", telemetry={"rejected": True})).score == 0.0


def test_execution_result_signal():
    sig = ExecutionResult()
    assert sig.score(Interaction("q", "a", telemetry={"passed": True})).score == 1.0
    assert sig.score(Interaction("q", "a", telemetry={"status": 500})).score == 0.0
    assert sig.score(Interaction("q", "a", telemetry={"exit_code": 0})).score == 1.0


# -- gate ------------------------------------------------------------------

def _size_eval(pb: Playbook, _holdout):
    # a toy "bigger playbook scores higher" evaluator
    return min(1.0, 0.5 + 0.1 * len(pb.bullets))


def test_gate_promotes_improvement():
    store = InMemoryPlaybookStore()
    d = promote(store, [Add("useful", "general")], holdout=["x"], evaluate=_size_eval)
    assert d.promoted and d.delta > 0
    assert store.version() == 1


def test_gate_rejects_and_quarantines_non_improvement():
    store = InMemoryPlaybookStore()

    def flat_eval(_pb, _holdout):
        return 0.6  # candidate never beats current

    d = promote(store, [Add("noise", "general")], holdout=["x"], evaluate=flat_eval,
                policy=PromotionPolicy(min_delta=0.01))
    assert not d.promoted
    assert store.version() == 0
    assert len(store.quarantined()) == 1


# -- facade ----------------------------------------------------------------

def test_augment_is_pure_string_injection():
    store = InMemoryPlaybookStore()
    ace = ACE(store=store)
    assert ace.augment("BASE") == "BASE"  # empty playbook -> unchanged
    store.append([Add("Answer concisely.", "style")])
    out = ace.augment("BASE")
    assert "BASE" in out and "Answer concisely." in out


def test_observe_requires_a_signal_or_feedback():
    ace = ACE(store=InMemoryPlaybookStore())
    with pytest.raises(ValueError, match="feedback= or a signal="):
        ace.observe(input="q", output="a")


def test_learn_reflects_curates_and_promotes():
    store = InMemoryPlaybookStore()
    # NoopReflector reads a lesson off interaction.meta; DirectCurator adds it.
    ace = ACE(
        store=store,
        reflector=NoopReflector(),
        curator=DirectCurator(section="learned"),
        signal=GroundTruth(metric=lambda g, o: 0.0),  # everything "fails" -> learn
    )
    ace.observe(input="q1", output="wrong", gold="right",
                meta={"lesson": "Prefer the exact tag name."})
    ace.observe(input="q2", output="wrong", gold="right",
                meta={"lesson": "State the unit explicitly."})
    assert ace.pending == 2

    ace.learn(holdout=["x"], evaluate=_size_eval)  # gated, playbook grows
    contents = [b.content for b in store.head().bullets]
    assert "Prefer the exact tag name." in contents
    assert "State the unit explicitly." in contents
    assert ace.pending == 0
    assert store.version() == 1


def test_learn_dev_mode_without_gate():
    store = InMemoryPlaybookStore()
    ace = ACE(store=store, reflector=NoopReflector(), curator=DirectCurator())
    ace.observe(input="q", output="o", feedback=Feedback(0.0, "bad"),
                meta={"lesson": "do better"})
    ace.learn()  # no holdout -> promote unconditionally
    assert store.version() == 1
    assert store.head().bullets[0].content == "do better"
