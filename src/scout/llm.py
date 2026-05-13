"""Pennsieve LLM Governor client.

Drop-in replacement for the original Bedrock-backed `LLMClient` from the
monorepo. Same call signature (`messages_create(model=, system=, messages=,
max_tokens=, prefill=, run_id=, agent=)`) and same return shape (a
Message-like object with `.content`, `.usage.input_tokens`,
`.usage.output_tokens`, `.stop_reason`, `.model`). Agents don't change.

Backend: invokes the governor Lambda (`$LLM_GOVERNOR_FUNCTION`) via
`boto3.client("lambda").invoke()`. Request/response envelope is defined in
`~/code/pennsieve/compute-node-aws-provisioner-v2/cmd/llm-governor/main.go`.

When the governor adds an Anthropic-compatible HTTP surface (Phase 1), swap
the body of `messages_create` for `anthropic.Anthropic(base_url=...).messages
.create(...)` and delete the hand-marshaling. The agent-facing API doesn't
change either way.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime

import boto3
from botocore.config import Config

# Logical model name → Bedrock model ID. Override via env vars so we don't
# redeploy the image to bump a model ID. The governor's SSM-managed
# allow-list is the source of truth for what's actually permitted.
#
# We use `us.*` (US-only) inference profiles by default — keeps data in US
# regions (HIPAA-friendly for Pennsieve customers). `global.*` profiles route
# anywhere AWS has capacity, including non-US regions, so they're unsuitable
# for PHI-bearing workloads.
MODEL_IDS: dict[str, str] = {
    "opus":   os.environ.get("VP_MODEL_OPUS",   "us.anthropic.claude-opus-4-7"),
    "sonnet": os.environ.get("VP_MODEL_SONNET", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"),
    "haiku":  os.environ.get("VP_MODEL_HAIKU",  "us.anthropic.claude-haiku-4-5-20251001-v1:0"),
}

# Lambda invoke uses sigv4 + sync RequestResponse mode. The governor itself
# has a Lambda timeout (~5-15 min); we set our client read timeout slightly
# longer than the longest expected governor call.
_LAMBDA_READ_TIMEOUT_S = int(os.environ.get("VP_LLM_TIMEOUT_S", "600"))


class GovernorError(RuntimeError):
    """Raised when the governor returns a structured error response."""


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
    """Drop-in for the monorepo's LLMClient, backed by the Pennsieve LLM
    Governor Lambda instead of direct Bedrock."""

    def __init__(self) -> None:
        function_name = os.environ.get("LLM_GOVERNOR_FUNCTION")
        if not function_name:
            raise RuntimeError(
                "LLM_GOVERNOR_FUNCTION is not set — this client must run inside "
                "a Pennsieve compute node, or set the env var manually for local "
                "testing against a deployed governor."
            )
        self._function_name = function_name
        self._execution_run_id = os.environ.get("EXECUTION_RUN_ID", "local-dev")
        cfg = Config(read_timeout=_LAMBDA_READ_TIMEOUT_S, connect_timeout=10, retries={"max_attempts": 0})
        self._client = boto3.client("lambda", config=cfg)
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
        """Invoke the governor. Same surface as the monorepo's LLMClient."""
        bedrock_model = MODEL_IDS.get(model)
        if not bedrock_model:
            raise RuntimeError(
                f"No model ID configured for logical name {model!r}. "
                f"Set VP_MODEL_{model.upper()} or use 'opus'/'sonnet'/'haiku'."
            )

        # Canonical prefill trick — append an assistant turn that the model
        # continues from. Used by agents to force JSON output via prefill='{'.
        if prefill is not None:
            messages = list(messages) + [{"role": "assistant", "content": prefill}]

        prompt_chars = sum(_chars(m.get("content", "")) for m in messages) + len(system or "")
        prefill_note = f" prefill={prefill!r}" if prefill else ""
        print(
            f"  [llm] → {agent}  model={model}  prompt~{prompt_chars:,} chars{prefill_note}",
            flush=True,
        )

        payload: dict = {
            "action": "invoke",
            "model": bedrock_model,
            "messages": _to_governor_messages(messages),
            "maxTokens": max_tokens,
            "executionRunId": self._execution_run_id,
        }
        if system:
            payload["system"] = system
        if temperature > 0:
            payload["temperature"] = temperature

        t0 = time.monotonic()
        resp = self._client.invoke(
            FunctionName=self._function_name,
            InvocationType="RequestResponse",
            Payload=json.dumps(payload).encode("utf-8"),
        )
        body = json.loads(resp["Payload"].read())
        elapsed = time.monotonic() - t0

        if "error" in body:
            print(f"  [llm] ✗ {agent}  governor error: {body.get('error')}: {body.get('message', '')}", flush=True)
            raise GovernorError(f"{body['error']}: {body.get('message', '')}")

        text = "".join(b.get("text", "") for b in body.get("content", []) if b.get("type") == "text")
        usage = body.get("usage") or {}
        in_tok = int(usage.get("inputTokens", 0))
        out_tok = int(usage.get("outputTokens", 0))
        stop = body.get("stopReason", "") or ""

        # Caller convention (see agents/_common.call_json): when prefill was
        # supplied, the caller re-prepends it to .content[0].text. So .text
        # here is just the model's continuation, matching the Anthropic SDK's
        # behavior.

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
            "budget_remaining": body.get("budgetRemaining"),
        })
        return _Message(
            content=[_Block(text=text)],
            usage=_Usage(input_tokens=in_tok, output_tokens=out_tok),
            model=bedrock_model,
            stop_reason=stop,
        )


def _chars(content) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(len(b.get("text", "")) if isinstance(b, dict) else 0 for b in content)
    return 0


def _to_governor_messages(messages: list[dict]) -> list[dict]:
    """Convert `{role, content: str | list[block]}` → governor's content-block shape."""
    out: list[dict] = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            blocks: list[dict] = [{"type": "text", "text": content}]
        else:
            blocks = list(content)
        out.append({"role": msg["role"], "content": blocks})
    return out
