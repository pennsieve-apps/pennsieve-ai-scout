from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scout.blackboard import Blackboard
    from scout.llm import LLMClient

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.md").read_text()


def extract_json(text: str) -> dict:
    """Extract a JSON object from LLM output that may have surrounding prose or fences.

    Two-stage parsing:
      1. `json.loads(strict=False)` — handles literal newlines/tabs inside strings
         (which LLMs commonly emit in free-text fields).
      2. Fall back to `json_repair` for unescaped quotes, trailing commas, and
         other common LLM JSON foibles. ~$0 cost; pure-Python repair.
    """
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        candidate = fence.group(1)
    else:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if not brace:
            raise ValueError(f"No JSON object found in model output:\n{text[:500]}")
        candidate = brace.group(0)

    try:
        return json.loads(candidate, strict=False)
    except json.JSONDecodeError as strict_err:
        try:
            from json_repair import repair_json
        except ImportError:
            raise strict_err
        try:
            repaired = repair_json(candidate, return_objects=True)
        except Exception:
            raise strict_err
        if isinstance(repaired, dict):
            return repaired
        raise ValueError(f"json_repair returned non-dict ({type(repaired).__name__})") from strict_err


_ATTACK_TAGS = ("[CONSTRAINT]", "[REASONING]", "[ALIGNMENT]")


def attack_kind(attack: str) -> str:
    """Return 'constraint' | 'reasoning' | 'alignment' | 'untagged' for a Critic attack string."""
    s = (attack or "").strip()
    if s.startswith("[CONSTRAINT]"):
        return "constraint"
    if s.startswith("[REASONING]"):
        return "reasoning"
    if s.startswith("[ALIGNMENT]"):
        return "alignment"
    return "untagged"


def strip_attack_tag(attack: str) -> str:
    """Remove the leading [TAG] prefix if present, for readable display."""
    s = (attack or "").strip()
    for tag in _ATTACK_TAGS:
        if s.startswith(tag):
            return s[len(tag):].strip()
    return s


def format_datasets(bb: "Blackboard") -> str:
    """Render dataset summaries with depositor README, file-type histogram,
    metadata-record previews, and linked publications. Used by HypGen and
    Critic so they reason from the same picture of what the data is."""
    chunks = []
    for d in bb.dataset_summaries:
        head = (
            f"[{d.id}] {d.title}\n"
            f"    Species: {d.species}, n={d.n_subjects}\n"
            f"    Measurements: {', '.join(str(m) for m in d.measurements)}\n"
            f"    {d.description}"
        )
        extras = []
        if d.bucket_uri:
            extras.append(f"    Bucket URI: {d.bucket_uri}  (access: {d.access_type or 'unknown'})")
        if d.readme:
            extras.append(f"    README: {d.readme[:1500].strip()}")
        if d.file_types:
            ft = ", ".join(f"{k}={v}" for k, v in list(d.file_types.items())[:5])
            extras.append(f"    File types: {ft}")
        if d.file_tree:
            tree_block = "\n".join("    " + line for line in d.file_tree.split("\n"))
            extras.append(f"    File tree (from manifest.json — these are the ONLY paths that exist):\n{tree_block}")
        if d.record_tables:
            for fn, info in d.record_tables.items():
                cols = ", ".join(info.get("columns", [])[:8])
                more = " ..." if len(info.get("columns", [])) > 8 else ""
                preview = info.get("head", [])[:3]
                preview_str = "; ".join(
                    "{" + ", ".join(f"{k}={v}" for k, v in row.items()) + "}"
                    for row in preview
                )
                extras.append(
                    f"    Record table `{fn}` ({info.get('n_rows', 0)} rows): "
                    f"columns=[{cols}{more}]; first rows: {preview_str[:400]}"
                )
        if d.external_publications:
            pubs = ", ".join(p.get("doi", "?") for p in d.external_publications[:3])
            extras.append(f"    Linked publications: {pubs}")
        chunks.append(head + ("\n" + "\n".join(extras) if extras else ""))
    return "\n\n".join(chunks)


def call_json(
    client: "LLMClient",
    *,
    agent: str,
    system: str,
    user: str,
    run_id: str,
    model: str = "opus",
    max_tokens: int = 16384,
) -> dict:
    """Call the model serving `model` (a pipeline role) and parse JSON from
    its response.

    Where the target model supports it we use the canonical "prefill `{`"
    trick to force JSON output. Claude 4.6 and newer reject an
    assistant-message prefill — the conversation must end on a user turn —
    so for those we skip it and rely on `extract_json` to recover JSON from
    whatever comes back (fenced or unfenced).
    """
    # Ask the client rather than keying off the role name: a role can be
    # served by a different concrete model than its first choice when the
    # backend's allow-list doesn't permit it, and prefill rules differ by
    # model.
    supports_prefill = client.supports_prefill(model)

    msg = client.messages_create(
        model=model,
        system=system,
        messages=[{"role": "user", "content": user}],
        prefill="{" if supports_prefill else None,
        max_tokens=max_tokens,
        run_id=run_id,
        agent=agent,
    )
    # Re-ask after the call rather than reusing the value from before it: a
    # role rejected mid-call is re-resolved onto another model, which may
    # have different prefill rules. Asking the client again reflects whatever
    # actually served the request, and avoids depending on any attribute of
    # the SDK's response object.
    prefilled = supports_prefill and client.supports_prefill(model)
    text = ("{" if prefilled else "") + first_text(msg)
    return extract_json(text)


def first_text(msg) -> str:
    """The first text block's text.

    Never index `content[0]` directly: models with thinking enabled put a
    `thinking` block first. Opus 5 runs adaptive thinking even when the
    request omits the `thinking` parameter, so a tier pinned or degraded
    onto it would otherwise raise AttributeError on a `ThinkingBlock`.
    """
    for block in msg.content:
        if getattr(block, "type", None) == "text":
            return block.text
    kinds = ", ".join(getattr(b, "type", "?") for b in msg.content) or "(empty)"
    raise ValueError(f"No text block in model response; got: {kinds}")
