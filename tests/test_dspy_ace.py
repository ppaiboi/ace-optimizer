"""End-to-end dspy.ACE test driven by DummyLM (no network, no API keys).

Covers the citing generator: playbook is an input field, the generator emits
bullet_ids, and only cited bullets are eligible for tagging.
"""

from __future__ import annotations

import dspy
from dspy.utils import DummyLM

from ace import Playbook
from dspy_ace import ACE, DspyAdapter
from dspy_ace.generator import ACEGenerator


def exact_match(example, pred, *args):
    return 1.0 if str(getattr(pred, "answer", "")).strip() == example.answer else 0.0


def _program():
    return dspy.Predict("question -> answer")


def test_generator_augments_signature_and_binds_playbook():
    adapter = DspyAdapter(_program(), exact_match)
    sig = adapter._predictor.signature
    assert "playbook" in sig.input_fields  # playbook is a first-class input
    assert "bullet_ids" in sig.output_fields  # generator must cite
    assert "answer" in sig.output_fields  # task output preserved

    pb = Playbook.parse(
        "## GEOGRAPHY\n[geog-00001] helpful=0 harmful=0 :: Answer with the city name only."
    )
    prog = adapter.build_program(pb)
    assert isinstance(prog, ACEGenerator)
    assert "city name only" in prog.playbook_text
    assert adapter.build_program(Playbook()).playbook_text == "(empty)"


def test_single_predictor_required():
    class Multi(dspy.Module):
        def __init__(self):
            super().__init__()
            self.a = dspy.Predict("question -> answer")
            self.b = dspy.Predict("answer -> grade")

    import pytest

    with pytest.raises(ValueError, match="single-predictor"):
        DspyAdapter(Multi(), exact_match)


def test_ace_compile_end_to_end_cites_and_adds_a_bullet():
    train = [dspy.Example(question="Capital of France?", answer="Paris").with_inputs("question")]
    val = [dspy.Example(question="Capital of France?", answer="Paris").with_inputs("question")]

    # Faithful loop, max_num_rounds=1. Ordered LM calls:
    #   gen(seed-eval), gen(step-initial, wrong), reflect, gen(regenerate),
    #   curate, gen(final-eval)
    lm = DummyLM(
        [
            {"answer": "Paris", "bullet_ids": ""},
            {"answer": "Lyon", "bullet_ids": ""},
            {"bullet_tags": "", "lessons": "For capital questions, answer with the city name only."},
            {"answer": "Paris", "bullet_ids": ""},
            {"additions": "geography :: For capital questions, answer with the city name only."},
            {"answer": "Paris", "bullet_ids": "geog-00001"},
        ]
    )
    dspy.configure(lm=lm)

    optimized = ACE(metric=exact_match, max_num_rounds=1).compile(
        _program(), trainset=train, valset=val
    )

    assert isinstance(optimized, ACEGenerator)
    pb = optimized.ace_playbook
    assert len(pb.bullets) == 1
    assert pb.bullets[0].id.startswith("geog-")  # abbreviated slug id
    assert "city name only" in pb.bullets[0].content
    assert "city name only" in optimized.playbook_text  # injected into the returned program
    assert optimized.ace_result.total_metric_calls == 4  # seed + initial + regen + final
