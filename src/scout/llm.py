"""LLM client for Scout — wraps the `pennsieve-llm` SDK governor client.

`pennsieve_llm.Governor().client()` returns a configured
`anthropic.Anthropic` pointed at the governor Lambda with SigV4 auth and the
`x-execution-run-id` header wired up. This file adds the Scout-specific
concerns on top: turning a pipeline *role* into a concrete model ID (see
`scout.models`), building a request the target model will actually accept,
per-call usage logging written to `usage.jsonl`, and the structured
per-agent print lines that show up in the notebook.

Model selection is both preflighted and self-healing. `preflight()` learns
the backend's allow-list before any real work and fails loud if a
reasoning-critical role cannot be served. Should a model still be rejected
mid-run, the allow-list is re-learned from the 403 body, the role is
re-resolved, and the call is retried once.
"""
from __future__ import annotations

import os
import re
import time
from datetime import datetime

import anthropic
from pennsieve_llm import Governor

from scout.models import (
    ModelNotAvailable,
    ModelResolver,
    discover_allowed_models,
    is_model_not_allowed,
    parse_allowed_models,
    render_manifest,
    select_interactively,
)

__all__ = ["LLMClient", "ModelNotAvailable", "supports_prefill", "supports_sampling"]


# Claude 4.6 and newer reject an assistant-turn prefill (the conversation
# must end on a user turn) — a request carrying one comes back 400. Matched
# on the family fragment so dated and undated profile IDs both hit.
_NO_PREFILL: tuple[str, ...] = (
    "opus-4-6", "opus-4-7", "opus-4-8", "opus-5",
    "sonnet-4-6", "sonnet-5",
    "fable", "mythos",
)

# Opus 4.7+ and Sonnet 5 removed the sampling parameters outright; sending
# `temperature` to one is a 400. The 4.6 generation still accepts it.
_NO_SAMPLING: tuple[str, ...] = (
    "opus-4-7", "opus-4-8", "opus-5",
    "sonnet-5",
    "fable", "mythos",
)

_DATE_SUFFIX = re.compile(r"-\d{8}(?:-v\d+:\d+)?$")


def _family(model_id: str) -> str:
    return _DATE_SUFFIX.sub("", model_id.strip())


def supports_prefill(model_id: str) -> bool:
    """True if `model_id` accepts a trailing assistant-turn prefill."""
    return not any(tag in _family(model_id) for tag in _NO_PREFILL)


def supports_sampling(model_id: str) -> bool:
    """True if `model_id` still accepts `temperature` / `top_p`."""
    return not any(tag in _family(model_id) for tag in _NO_SAMPLING)


# The governor transport ignores httpx timeouts — it dispatches through boto3,
# whose read timeout is 60s with no Config override. A failing call therefore
# costs 60s x botocore retries x this, so keep the SDK's own retry budget low
# or a doomed request burns 15 minutes before surfacing.
SDK_MAX_RETRIES = int(os.environ.get("VP_LLM_MAX_RETRIES", "1"))


class LLMClient:
    """Thin facade around the `anthropic.Anthropic` client returned by the
    governor SDK. Agents call `messages_create` with a pipeline role name;
    the concrete model is resolved here."""

    def __init__(self) -> None:
        # Governor() reads LLM_GOVERNOR_URL / LLM_GOVERNOR_FUNCTION from the
        # environment and picks its transport:
        #   - HTTPS + SigV4 if a URL is set
        #   - lambda-invoke if only a function name is set
        #   - else MockClient (tests / offline)
        self._gov = Governor()
        self._resolver = ModelResolver()
        self.usage_log: list[dict] = []

    # -- preflight ------------------------------------------------------

    def preflight(self) -> dict:
        """Discover the allow-list, assign every role, print the manifest.

        Returns role → assignment provenance, suitable for recording on the
        blackboard so a downstream reader can tell what produced a claim.
        Raises `ModelNotAvailable` if a role's floor cannot be met, which is
        the point: better to fail here than to spend a full run producing
        output that isn't worth reading.
        """
        allowed = discover_allowed_models(self._probe)
        if allowed:
            self._resolver.learn(allowed)
        select_interactively(self._resolver)
        assignments = self._resolver.resolve_all()
        print(render_manifest(self._resolver, assignments), flush=True)
        return self._resolver.assignments()

    def _send(self, kwargs: dict):
        """Issue the request, streaming when the client supports it.

        Streaming is not an optimization here, it is load-bearing. The
        governor's lambda-invoke transport hands httpx a lazily-consumed byte
        stream only for `text/event-stream` responses; for a non-streaming one
        it drains the entire Lambda EventStream inside `handle_request` before
        returning. Botocore's read timeout is 60s, and a large completion takes
        far longer than that to generate, so a non-streaming call on a big
        prompt reliably dies with ReadTimeoutError. Streaming keeps bytes
        moving so no single read comes close to the limit.
        """
        client = self._gov.client()
        if hasattr(client, "with_options"):
            client = client.with_options(max_retries=SDK_MAX_RETRIES)
        stream = getattr(client.messages, "stream", None)
        if stream is None:
            # MockClient (offline / tests) has no streaming surface.
            return client.messages.create(**kwargs)
        with stream(**kwargs) as s:
            return s.get_final_message()

    def _probe(self, model_id: str) -> None:
        """Minimal request used to trip the governor's allow-list rejection."""
        self._gov.client().messages.create(
            model=model_id,
            max_tokens=1,
            messages=[{"role": "user", "content": "."}],
        )

    def model_assignments(self) -> dict:
        """Role → assignment provenance for whatever has been resolved."""
        return self._resolver.assignments()

    # -- resolution -----------------------------------------------------

    def resolve_model(self, role: str) -> str:
        """Pipeline role (or legacy vendor-tier alias) → concrete model ID."""
        return self._resolver.resolve(role)

    def supports_prefill(self, role: str) -> bool:
        """True if the model currently serving `role` accepts a prefill.

        Callers that reconstruct response text from a prefill must ask here
        rather than assume per role name — a role can be served by a model
        with different prefill rules than its first choice.
        """
        return supports_prefill(self.resolve_model(role))

    # -- requests -------------------------------------------------------

    def messages_create(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict],
        max_tokens: int = 8192,
        temperature: float = 1.0,
        prefill: str | None = None,
        run_id: str = "",
        agent: str = "",
    ):
        """Send one request and log its usage.

        `model` is a pipeline role name, not a model ID. Returns the SDK's
        `Message` directly — `.content`, `.usage`, `.stop_reason`, `.model`
        are all read by agent code.

        On a `model_not_allowed` 403 the allow-list is re-learned from the
        error body, the role re-resolved, and the call retried once. If the
        retry lands on a model that rejects prefill, the prefill is silently
        dropped — callers that reconstruct text from a prefill should ask
        `supports_prefill(role)` *after* the call, which reflects the model
        that actually served it.
        """
        model_id = self.resolve_model(model)
        prompt_chars = sum(_chars(m.get("content", "")) for m in messages) + len(system or "")

        for attempt in (1, 2):
            send_messages, prefill_used = _apply_prefill(messages, prefill, model_id)

            prefill_note = f" prefill={prefill_used!r}" if prefill_used else ""
            print(
                f"  [llm] → {agent}  role={model}  model={model_id}  "
                f"prompt~{prompt_chars:,} chars{prefill_note}",
                flush=True,
            )

            kwargs = {
                "model": model_id,
                "system": system,
                "messages": send_messages,
                "max_tokens": max_tokens,
            }
            # Opus 4.7+ / Sonnet 5 removed sampling params — sending one 400s.
            if supports_sampling(model_id):
                kwargs["temperature"] = temperature

            t0 = time.monotonic()
            try:
                resp = self._send(kwargs)
            except anthropic.PermissionDeniedError as exc:
                allowed = parse_allowed_models(exc) if is_model_not_allowed(exc) else None
                if allowed is None or attempt == 2:
                    raise
                print(
                    f"  [llm] ! {model_id} rejected by the allow-list; "
                    f"re-resolving role {model!r}",
                    flush=True,
                )
                self._resolver.learn(allowed)
                model_id = self.resolve_model(model)
                continue
            break

        elapsed = time.monotonic() - t0

        in_tok = resp.usage.input_tokens
        out_tok = resp.usage.output_tokens
        stop = resp.stop_reason or ""
        stop_note = f" stop={stop}" if stop and stop != "end_turn" else ""
        print(
            f"  [llm] ← {agent}  in={in_tok:,} tok  out={out_tok:,} tok  ({elapsed:.1f}s){stop_note}",
            flush=True,
        )

        self.usage_log.append({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "run_id": run_id,
            "agent": agent,
            "role": model,
            "model_id": model_id,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "elapsed_s": round(elapsed, 2),
            "stop_reason": stop,
        })

        return resp


def _apply_prefill(
    messages: list[dict], prefill: str | None, model_id: str
) -> tuple[list[dict], str | None]:
    """Canonical prefill trick — append an assistant turn the model continues
    from. Dropped for models that reject it (Claude 4.6+), which can happen
    when a role degrades onto a newer model than it started on.
    """
    if prefill is None or not supports_prefill(model_id):
        return list(messages), None
    return [*messages, {"role": "assistant", "content": prefill}], prefill


def _chars(content) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(len(b.get("text", "")) if isinstance(b, dict) else 0 for b in content)
    return 0
