"""Optimization-loop tests using a pure stub adapter (no dspy, no LM)."""

from __future__ import annotations

from ace import Add, EvaluationBatch, Playbook, optimize
from ace.online import OnlinePlaybook


class StubAdapter:
    """Every sample 'fails' (score 0) so multi-round reflection fires; each
    curation adds `adds_per_curate` bullets; valset score rises with size."""

    def __init__(self, adds_per_curate: int = 1):
        self.adds_per_curate = adds_per_curate
        self.gen_calls = 0

    def evaluate(self, batch, playbook, capture_traces=False):
        score = min(1.0, 0.5 + 0.1 * len(playbook.bullets))
        return EvaluationBatch(outputs=[None] * len(batch), scores=[score] * len(batch))

    def generate_one(self, sample, playbook, reflection="(empty)"):
        self.gen_calls += 1
        return {"pred": None, "score": 0.0, "feedback": "wrong", "cited": []}

    def reflect_one(self, sample, gen, playbook):
        return {"tags": [], "lessons": ["a lesson"], "reflection_text": "try harder"}

    def curate(self, playbook, lessons):
        return [Add(f"b{i}", "general") for i in range(self.adds_per_curate)]


def test_optimize_grows_playbook_and_improves_score():
    adapter = StubAdapter()
    res = optimize(
        Playbook(), ["s0", "s1"], adapter, valset=["v"],
        max_num_rounds=2, curator_frequency=1, eval_steps=1, seed=1,
    )
    assert res.improved
    assert len(res.final_playbook.bullets) == 2  # one curation per step, 2 steps
    assert res.best_score > res.seed_score


def test_multiround_generation_count():
    adapter = StubAdapter()
    optimize(Playbook(), ["s0", "s1"], adapter, valset=["v"],
             max_num_rounds=2, eval_steps=1)
    # per sample: 1 initial + 2 reflect/regenerate rounds = 3; two samples
    assert adapter.gen_calls == 6


def test_max_num_rounds_one_is_single_shot():
    adapter = StubAdapter()
    optimize(Playbook(), ["s0"], adapter, valset=["v"], max_num_rounds=1, eval_steps=1)
    assert adapter.gen_calls == 2  # 1 initial + 1 regenerate


def test_online_playbook_accumulates_across_episodes():
    mem = OnlinePlaybook(StubAdapter(adds_per_curate=2))
    assert len(mem.playbook.bullets) == 0
    mem.learn(["episode-1"])
    assert len(mem.playbook.bullets) == 2
    mem.learn(["episode-2"])
    assert len(mem.playbook.bullets) == 4  # memory persists and grows
