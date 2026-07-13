"""Anthropic integration + LLM-backed Reflector/Curator, with fakes (no SDK)."""

from __future__ import annotations

from types import SimpleNamespace

from ace import ACE, Add, InMemoryPlaybookStore, LLMCurator, LLMReflector
from ace.integrations.anthropic import completion_fn, wrap


class FakeAnthropic:
    """Minimal stand-in exposing the messages.create surface we use."""

    def __init__(self, reply: str = "ok"):
        self.reply = reply
        self.calls: list[dict] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(text=self.reply)])


def test_completion_fn_extracts_text():
    complete = completion_fn(FakeAnthropic(reply="hello"), model="claude-haiku-4-5")
    assert complete("prompt") == "hello"


def test_wrap_injects_playbook_into_system():
    store = InMemoryPlaybookStore()
    store.append([Add("Cite sources.", "style")])
    client = FakeAnthropic()
    wrapped = wrap(client, ACE(store=store))
    wrapped.messages.create(model="m", max_tokens=10, messages=[{"role": "user", "content": "hi"}])
    sent_system = client.calls[0]["system"]
    assert "Cite sources." in sent_system


def test_llm_reflector_and_curator_end_to_end():
    # fake LLM: reflector asks for lessons, curator asks for SECTION :: content
    def fake_llm(prompt: str) -> str:
        if "reusable lessons" in prompt or "transferable lessons" in prompt:
            return "Always name the exact GAAP tag.\nState the reporting period."
        return "TAGGING :: Always name the exact GAAP tag."

    store = InMemoryPlaybookStore()
    ace = ACE(
        store=store,
        reflector=LLMReflector(llm=fake_llm),
        curator=LLMCurator(llm=fake_llm),
    )
    from ace import Feedback

    ace.observe(input="q", output="wrong", feedback=Feedback(0.0, "missed the tag"))
    ace.learn()  # dev mode
    contents = [b.content for b in store.head().bullets]
    assert any("GAAP tag" in c for c in contents)
    assert store.head().bullets[0].section == "tagging"  # section slug normalized
