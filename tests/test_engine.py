"""Engine-loop tests using a pure stub adapter (no dspy, no LM)."""

from __future__ import annotations

from ace import Add, EvaluationBatch, Playbook, optimize, optimize_faithful
from ace.online import OnlinePlaybook


class GrowStubAdapter:
    """Score improves with playbook size; each reflection adds one bullet.

    Lets us assert the loop accumulates, tracks best, and counts metric calls
    deterministically without any model.
    """

    def __init__(self, adds_per_round: int = 1):
        self.adds_per_round = adds_per_round

    def evaluate(self, batch, playbook, capture_traces=False):
        score = min(1.0, 0.5 + 0.1 * len(playbook.bullets))
        traj = [{"i": i} for i in range(len(batch))] if capture_traces else None
        return EvaluationBatch(
            outputs=[None] * len(batch), scores=[score] * len(batch), trajectories=traj
        )

    def make_reflective_dataset(self, playbook, eval_batch):
        return {"generator": [{"note": "x"} for _ in eval_batch.scores]}

    def propose_deltas(self, playbook, reflective_dataset):
        return [Add(f"lesson {i}", "general") for i in range(self.adds_per_round)]


def test_optimize_grows_playbook_and_improves_score():
    train = [f"ex{i}" for i in range(4)]
    res = optimize(
        Playbook(), train, GrowStubAdapter(), valset=["v"], minibatch_size=2, seed=1
    )
    assert res.improved
    assert res.best_score > res.seed_score
    assert len(res.best_playbook.bullets) >= 2
    # 2 minibatches -> 2 iterations recorded
    assert len(res.history) == 2
    assert res.history[-1].playbook_size == len(res.final_playbook.bullets)


def test_metric_calls_are_counted():
    train = ["a", "b", "c", "d"]
    val = ["v1", "v2"]
    res = optimize(Playbook(), train, GrowStubAdapter(), valset=val, minibatch_size=2)
    # seed full-eval (2) + per iter: minibatch(2) + full-eval(2); 2 iters
    assert res.total_metric_calls == len(val) + 2 * (2 + len(val))


def test_max_metric_calls_stops_early():
    train = ["a", "b", "c", "d", "e", "f", "g", "h"]
    res = optimize(
        Playbook(), train, GrowStubAdapter(), valset=["v"], minibatch_size=1,
        max_metric_calls=5,
    )
    assert res.total_metric_calls >= 5
    assert len(res.history) < len(train)  # did not run every batch


class FaithfulStub:
    """Stub with the fine-grained primitives; every sample fails so multi-round
    reflection fires, and each curation adds one bullet."""

    def __init__(self):
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
        return [Add("distilled bullet", "general")]


def test_faithful_multiround_and_curation():
    adapter = FaithfulStub()
    res = optimize_faithful(
        Playbook(), ["s0", "s1"], adapter, valset=["v"],
        max_num_rounds=2, curator_frequency=1, eval_steps=1, seed=1,
    )
    # each sample: 1 initial gen + 2 reflect/regenerate rounds = 3 gens; 2 samples
    assert adapter.gen_calls == 6
    # curator adds one bullet per step (2 steps)
    assert len(res.final_playbook.bullets) == 2
    assert res.improved  # score rises with playbook size


def test_online_playbook_accumulates_across_episodes():
    mem = OnlinePlaybook(GrowStubAdapter(adds_per_round=2))
    assert len(mem.playbook.bullets) == 0
    mem.learn(["episode-1"])
    assert len(mem.playbook.bullets) == 2
    mem.learn(["episode-2"])
    assert len(mem.playbook.bullets) == 4  # memory persists and grows
