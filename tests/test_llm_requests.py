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
