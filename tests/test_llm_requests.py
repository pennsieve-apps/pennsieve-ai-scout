"""Request-shaping rules in `scout.llm` — which models accept a prefill
and which still accept sampling params.

`pennsieve_llm` is an internal SDK that isn't installed in every dev env, so
stub it before importing `scout.llm`. Role resolution itself lives in
`scout.models` and is tested in `test_models.py` (no stub needed there).
"""
from __future__ import annotations

import sys
import types

import pytest

if "pennsieve_llm" not in sys.modules:  # pragma: no cover - env dependent
    stub = types.ModuleType("pennsieve_llm")
    stub.Governor = object
    sys.modules["pennsieve_llm"] = stub

from scout import llm  # noqa: E402


@pytest.mark.parametrize(
    "model_id, prefill_ok, sampling_ok",
    [
        ("us.anthropic.claude-sonnet-4-5-20250929-v1:0", True, True),
        ("us.anthropic.claude-haiku-4-5-20251001-v1:0", True, True),
        ("us.anthropic.claude-3-5-haiku-20241022-v1:0", True, True),
        ("us.anthropic.claude-sonnet-4-6", False, True),   # 4.6 dropped prefill
        ("us.anthropic.claude-opus-4-6", False, True),
        ("us.anthropic.claude-opus-4-7", False, False),    # 4.7 dropped sampling too
        ("us.anthropic.claude-opus-4-8", False, False),
        ("us.anthropic.claude-opus-5", False, False),
        ("us.anthropic.claude-sonnet-5", False, False),
        ("anthropic.claude-fable-5", False, False),
    ],
)
def test_capability_rules(model_id, prefill_ok, sampling_ok):
    assert llm.supports_prefill(model_id) is prefill_ok
    assert llm.supports_sampling(model_id) is sampling_ok


def test_prefill_dropped_for_models_that_reject_it():
    messages = [{"role": "user", "content": "hi"}]

    sent, used = llm._apply_prefill(messages, "{", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    assert used == "{" and sent[-1]["role"] == "assistant"

    sent, used = llm._apply_prefill(messages, "{", "us.anthropic.claude-sonnet-4-6")
    assert used is None and sent == messages


def test_apply_prefill_never_mutates_the_caller_list():
    messages = [{"role": "user", "content": "hi"}]
    llm._apply_prefill(messages, "{", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    assert len(messages) == 1


# --- transport: streaming is load-bearing, not an optimization -----------
#
# The governor's lambda-invoke transport only hands httpx a lazily-consumed
# stream for text/event-stream responses; for a non-streaming response it
# drains the whole Lambda EventStream inside handle_request. Botocore's read
# timeout is 60s, so a large non-streaming completion reliably dies with
# ReadTimeoutError. These tests pin the streaming path down.

_MSG = type("M", (), {
    "content": [type("B", (), {"type": "text", "text": "hi"})()],
    "usage": type("U", (), {"input_tokens": 1, "output_tokens": 1})(),
    "stop_reason": "end_turn",
    "model": "m",
})()


class _StreamCtx:
    def __init__(self, log):
        self._log = log

    def __enter__(self):
        self._log.append("stream")
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return _MSG


class _StreamingMessages:
    def __init__(self, log):
        self._log = log

    def stream(self, **kwargs):
        return _StreamCtx(self._log)

    def create(self, **kwargs):
        self._log.append("create")
        return _MSG


class _NoStreamMessages:
    """MockClient's shape — offline/test backend with no streaming surface."""

    def __init__(self, log):
        self._log = log

    def create(self, **kwargs):
        self._log.append("create")
        return _MSG


class _Client:
    def __init__(self, messages):
        self.messages = messages
        self.options = None

    def with_options(self, **kwargs):
        self.options = kwargs
        return self


def _client_for(messages_cls, log):
    client = _Client(messages_cls(log))
    scout_client = llm.LLMClient()
    scout_client._gov = type("G", (), {"client": lambda self: client})()
    return scout_client, client


def test_real_calls_go_through_streaming():
    log = []
    client, _ = _client_for(_StreamingMessages, log)
    client._send({"model": "m", "messages": [], "max_tokens": 16384})
    assert log == ["stream"], "a non-streaming create() would hit the 60s Lambda timeout"


def test_falls_back_to_create_when_the_backend_cannot_stream():
    log = []
    client, _ = _client_for(_NoStreamMessages, log)
    client._send({"model": "m", "messages": [], "max_tokens": 16384})
    assert log == ["create"]


def test_sdk_retry_budget_is_capped():
    """Botocore already retries a 60s timeout; the SDK retrying on top of that
    is what turned one doomed call into 15 minutes."""
    log = []
    client, raw = _client_for(_StreamingMessages, log)
    client._send({"model": "m", "messages": [], "max_tokens": 8})
    assert raw.options == {"max_retries": llm.SDK_MAX_RETRIES}
    assert llm.SDK_MAX_RETRIES <= 1
