"""Optimization-loop tests using a pure stub adapter (no dspy, no LM)."""

from __future__ import annotations

from ace import Add, EvaluationBatch, Playbook, optimize
from ace.checkpoint import Checkpoint, load_checkpoint
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


def test_checkpoint_is_written_and_resume_skips_completed_work(tmp_path):
    path = str(tmp_path / "ckpt.json")
    a1 = StubAdapter()
    r1 = optimize(Playbook(), ["s0", "s1"], a1, valset=["v"], max_num_rounds=1,
                  eval_steps=1, checkpoint_path=path, checkpoint_every=1)
    assert a1.gen_calls > 0
    ckpt = load_checkpoint(path)
    assert ckpt is not None and ckpt.step == 2  # ran to completion (2 steps)

    # resuming a completed checkpoint does no training work
    a2 = StubAdapter()
    r2 = optimize(Playbook(), ["s0", "s1"], a2, valset=["v"], max_num_rounds=1,
                  eval_steps=1, checkpoint_path=path, checkpoint_every=1)
    assert a2.gen_calls == 0  # loop skipped entirely
    assert len(r2.final_playbook.bullets) == len(r1.final_playbook.bullets)


def test_resume_ignored_on_config_mismatch(tmp_path):
    path = str(tmp_path / "ckpt.json")
    optimize(Playbook(), ["s0", "s1"], StubAdapter(), valset=["v"], max_num_rounds=1,
             eval_steps=1, checkpoint_path=path)
    # a different trainset size => different config_hash => checkpoint ignored
    a = StubAdapter()
    optimize(Playbook(), ["s0", "s1", "s2"], a, valset=["v"], max_num_rounds=1,
             eval_steps=1, checkpoint_path=path)
    assert a.gen_calls > 0  # ran fresh, did not resume


def test_checkpoint_roundtrip():
    pb = Playbook.parse("## S\n[s-00001] helpful=1 harmful=0 :: keep it short")
    ck = Checkpoint(config_hash="abc", step=5, seed_score=0.5, best_score=0.7,
                    metric_calls=42, playbook=pb, best_playbook=pb,
                    lessons_buffer=["a lesson"], history=[])
    back = Checkpoint.from_dict(ck.to_dict())
    assert back.step == 5 and back.best_score == 0.7
    assert back.playbook.bullets[0].content == "keep it short"


def test_interim_valset_used_for_periodic_checks():
    # interim scoring set differs from final; both should be exercised
    seen = {"sets": []}

    class RecordingAdapter(StubAdapter):
        def evaluate(self, batch, playbook, capture_traces=False):
            seen["sets"].append(tuple(batch))
            return super().evaluate(batch, playbook, capture_traces)

    a = RecordingAdapter()
    optimize(Playbook(), ["s0", "s1"], a, valset=["full1", "full2"],
             interim_valset=["interim"], max_num_rounds=1, eval_steps=1)
    assert ("interim",) in seen["sets"]          # periodic checks used the subset
    assert ("full1", "full2") in seen["sets"]    # final used the full valset


def test_online_playbook_accumulates_across_episodes():
    mem = OnlinePlaybook(StubAdapter(adds_per_curate=2))
    assert len(mem.playbook.bullets) == 0
    mem.learn(["episode-1"])
    assert len(mem.playbook.bullets) == 2
    mem.learn(["episode-2"])
    assert len(mem.playbook.bullets) == 4  # memory persists and grows
