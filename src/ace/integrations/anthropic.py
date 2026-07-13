"""Anthropic integration: a completion callable for the Reflector/Curator, and
``wrap()`` — a drop-in client proxy that injects the playbook on the hot path.

The ``anthropic`` SDK is imported lazily (only ``completion_fn``/``wrap`` need a
real client), so importing this module never requires the SDK, and everything is
testable with a fake client exposing the same ``messages.create`` surface.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ace.facade import ACE


def _text_of(response: Any) -> str:
    """Pull the text out of an Anthropic messages response (or a fake)."""
    content = getattr(response, "content", None)
    if content:
        parts = [getattr(b, "text", "") for b in content]
        return "".join(p for p in parts if p)
    return str(response)


def completion_fn(
    client: Any, model: str = "claude-haiku-4-5", *, max_tokens: int = 1024
) -> Callable[[str], str]:
    """Build a ``str -> str`` completion callable for the Reflector/Curator."""

    def complete(prompt: str) -> str:
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return _text_of(resp)

    return complete


class _WrappedMessages:
    def __init__(self, inner: Any, ace: ACE) -> None:
        self._inner = inner
        self._ace = ace

    def create(self, **kwargs: Any) -> Any:
        """Same signature as ``client.messages.create``, but the playbook is
        injected into ``system`` first (hot path — pure string op)."""
        kwargs["system"] = self._ace.augment(kwargs.get("system", "") or "")
        return self._inner.create(**kwargs)


class _WrappedClient:
    def __init__(self, client: Any, ace: ACE) -> None:
        self._client = client
        self.messages = _WrappedMessages(client.messages, ace)

    def __getattr__(self, name: str) -> Any:  # pass through everything else
        return getattr(self._client, name)


def wrap(client: Any, ace: ACE) -> _WrappedClient:
    """Wrap an Anthropic client so ``messages.create`` auto-injects the playbook.

    The wrapped client has the same interface; only the system prompt changes.
    Capturing feedback (``ace.observe``) stays explicit — the wrapper can't know
    whether an output was good."""
    return _WrappedClient(client, ace)
