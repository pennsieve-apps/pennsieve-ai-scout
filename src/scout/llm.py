"""Pennsieve LLM Governor client — adapter over `pennsieve-llm` Python SDK.

Replaces the previous hand-marshaled `boto3.lambda.invoke()` path. The
governor migrated to a Lambda Function URL with response streaming, so
direct lambda:Invoke no longer works. All LLM access now goes through the
official `pennsieve-llm` SDK, which handles SigV4-signed HTTPS calls
against the governor URL internally.

The agent-facing interface (`LLMClient.messages_create(...)`) is unchanged
so the existing agent code continues to work without modification.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime

from pennsieve_llm import (
    Governor,
    GovernorError as _SDKGovernorError,
    InvokeRequest,
    assistant_message,
    text_block,
    user_message,
)


# Logical model name → Bedrock inference-profile ID. The governor's
# SSM-managed allow-list is the source of truth for what's actually
# permitted; defaults below match `var.llm_allowed_models` in the
# provisioner. Override via env vars to bump a model ID without rebuilding
# the image.
#
# us.* keeps inference in US regions (HIPAA-friendly default).
MODEL_IDS: dict[str, str] = {
    "opus":   os.environ.get("VP_MODEL_OPUS",   "us.anthropic.claude-opus-4-7"),
    "sonnet": os.environ.get("VP_MODEL_SONNET", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"),
    "haiku":  os.environ.get("VP_MODEL_HAIKU",  "us.anthropic.claude-haiku-4-5-20251001-v1:0"),
}


class GovernorError(RuntimeError):
    """Raised when the governor returns a structured error response.
    Wraps pennsieve_llm.GovernorError so callers see a single exception type
    from this module."""


@dataclass
class _Block:
    text: str
    type: str = "text"


@dataclass
class _Usage:
    input_tokens: int
    output_tokens: int


@dataclass
class _Message:
    """Anthropic-SDK-shaped message object. Mirrors the fields agents read:
    `.content[i].text`, `.usage.input_tokens`, `.usage.output_tokens`,
    `.stop_reason`, `.model`."""
    content: list[_Block]
    usage: _Usage
    model: str
    stop_reason: str = ""


class LLMClient:
    """Adapter that exposes the legacy `messages_create(...)` interface
    while delegating actual LLM calls to the `pennsieve-llm` Governor.

    Agent code (in `agents/_common.py:call_json` and friends) keeps calling
    `client.messages_create(model="opus", ...)` and gets back a
    Message-shaped object. This layer translates to/from the SDK's
    InvokeRequest/InvokeResponse types.
    """

    def __init__(self) -> None:
        # Governor() auto-selects:
        #   - LLM_GOVERNOR_URL set → HTTPS+SigV4 to deployed governor
        #   - ANTHROPIC_API_KEY set → direct to api.anthropic.com (local dev)
        #   - neither → MockBackend (tests)
        self._gov = Governor()
        self.usage_log: list[dict] = []

    def messages_create(
        self,
        *,
        model: str,                       # "opus" | "sonnet" | "haiku"
        system: str,
        messages: list[dict],
        max_tokens: int = 16384,
        prefill: str | None = None,
        run_id: str | None = None,
        agent: str | None = None,
        temperature: float = 0.0,
    ) -> _Message:
        """Drop-in for the previous boto3-backed client. Same signature,
        same return shape; SDK handles transport + SigV4 + Anthropic
        Messages API translation internally."""
        bedrock_model = MODEL_IDS.get(model)
        if not bedrock_model:
            raise RuntimeError(
                f"No model ID configured for logical name {model!r}. "
                f"Set VP_MODEL_{model.upper()} or use 'opus'/'sonnet'/'haiku'."
            )

        # Canonical prefill trick — append an assistant turn that the model
        # continues from. Caller (agents/_common.call_json) handles the
        # Opus-extended-thinking exception by passing prefill=None.
        scout_messages = list(messages)
        if prefill is not None:
            scout_messages = scout_messages + [{"role": "assistant", "content": prefill}]

        sdk_messages = [_to_sdk_message(m) for m in scout_messages]

        prompt_chars = sum(_chars(m.get("content", "")) for m in scout_messages) + len(system or "")
        prefill_note = f" prefill={prefill!r}" if prefill else ""
        print(
            f"  [llm] → {agent}  model={model}  prompt~{prompt_chars:,} chars{prefill_note}",
            flush=True,
        )

        req = InvokeRequest(
            model=bedrock_model,
            system=system,
            messages=sdk_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            execution_run_id=run_id or os.environ.get("EXECUTION_RUN_ID", "local-dev"),
        )

        t0 = time.monotonic()
        try:
            resp = self._gov.invoke(req)
        except _SDKGovernorError as e:
            # Re-raise as scout's GovernorError so call sites can catch a
            # single exception type from this module.
            print(f"  [llm] ✗ {agent}  governor error: {e.code}: {e.msg}", flush=True)
            raise GovernorError(f"{e.code}: {e.msg}") from e
        elapsed = time.monotonic() - t0

        text = resp.text()
        in_tok = resp.usage.input_tokens
        out_tok = resp.usage.output_tokens
        stop = resp.stop_reason or ""

        # Caller convention (agents/_common.call_json): when prefill was
        # supplied, the caller re-prepends it. So .text returned here is
        # just the model's continuation, matching the Anthropic SDK's
        # behavior. We do NOT prepend prefill here.

        stop_note = f" stop={stop}" if stop and stop != "end_turn" else ""
        print(
            f"  [llm] ← {agent}  in={in_tok:,} tok  out={out_tok:,} tok  ({elapsed:.1f}s){stop_note}",
            flush=True,
        )

        self.usage_log.append({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "run_id": run_id,
            "agent": agent,
            "model": model,
            "model_id": bedrock_model,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "elapsed_s": round(elapsed, 2),
            "stop_reason": stop,
        })

        return _Message(
            content=[_Block(text=text)],
            usage=_Usage(input_tokens=in_tok, output_tokens=out_tok),
            model=resp.model or bedrock_model,
            stop_reason=stop,
        )


def _to_sdk_message(m: dict):
    """Convert one of Scout's message dicts {role, content} into the SDK's
    Message dataclass. Content may be a plain string or a list of block dicts."""
    role = m.get("role", "user")
    content = m.get("content", "")
    if isinstance(content, str):
        blocks = [text_block(content)]
    else:
        # Already a list of block dicts (from agents that build content blocks
        # explicitly). Convert each to an SDK ContentBlock via text_block —
        # we currently only emit text blocks from scout; extend here if image
        # / document blocks become needed.
        blocks = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                blocks.append(text_block(b.get("text", "")))
            elif isinstance(b, str):
                blocks.append(text_block(b))
    if role == "assistant":
        return assistant_message(*blocks)
    return user_message(*blocks)


def _chars(content) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(len(b.get("text", "")) if isinstance(b, dict) else 0 for b in content)
    return 0
