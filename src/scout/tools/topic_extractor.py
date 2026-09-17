"""LLM topic extraction for RAG ingest tagging — tiered version.

Returns categorized topics: `domain`, `broader`, `modality`, `species`,
`technique`. Only `domain + broader` are used as the topical filter for
retrieval (`topics_for_retrieval`); the other categories are kept for
downstream display and methods analysis.

Used at three points:
  1. Deep Reader, when adding a new paper — tags chunks at ingest.
  2. Backfill script — retroactively tags legacy chunks.
  3. Orchestrator dataset-first run — extracts the seed dataset's topics
     to anchor the per-run topical filter (using domain + broader, not
     the depositor's free-form measurements list).

Sonnet, ~$0.003 per paper. ~$0.40 to backfill the existing corpus.
"""
from __future__ import annotations

from scout.agents._common import call_json, load_prompt
from pennsieve_ai_utils.llm import LLMClient

MODEL = "bulk"
AGENT = "topic_extractor"

CATEGORIES = ("domain", "broader", "modality", "species", "technique")
RETRIEVAL_CATEGORIES = ("domain", "broader")  # what anchors topical filtering


def extract_topics(
    client: LLMClient,
    *,
    title: str,
    abstract: str,
    body_excerpt: str = "",
    run_id: str | None = None,
) -> dict[str, list[str]]:
    """Return structured tags by category. All five keys present (possibly
    empty). Empty dict on extraction failure."""
    if not (title or abstract or body_excerpt):
        return {k: [] for k in CATEGORIES}
    system = load_prompt("topic_extractor")
    user = "\n\n".join([
        f"## Title\n{title.strip()}" if title else "",
        f"## Abstract\n{abstract.strip()}" if abstract else "",
        f"## Body excerpt (first ~1000 chars)\n{body_excerpt.strip()[:1500]}" if body_excerpt else "",
    ]).strip()
    try:
        parsed = call_json(
            client,
            agent=AGENT,
            system=system,
            user=user,
            run_id=run_id or "",
            model=MODEL,
            max_tokens=512,
        )
    except Exception as e:
        # Print for the live log, then let it propagate — the pipeline records
        # the failure on bb.degraded so the blackboard doesn't pass an
        # LLM-failure-shaped hole off as "this dataset genuinely has no topics".
        import sys
        print(f"  [topic_extractor] failed: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
        raise

    out: dict[str, list[str]] = {}
    for cat in CATEGORIES:
        seen: set[str] = set()
        clean: list[str] = []
        for t in (parsed.get(cat, []) or []):
            if not isinstance(t, str):
                continue
            norm = t.strip().lower().replace(" ", "-").replace("_", "-")
            if not norm or norm in seen:
                continue
            seen.add(norm)
            clean.append(norm)
        out[cat] = clean
    return out


def topics_for_retrieval(structured: dict[str, list[str]]) -> list[str]:
    """Pull domain + broader tags as a flat list — these are what should
    anchor topical retrieval. Modality, species, and technique are too
    generic to discriminate across domains."""
    out: list[str] = []
    seen: set[str] = set()
    for key in RETRIEVAL_CATEGORIES:
        for t in (structured.get(key, []) or []):
            if t and t not in seen:
                seen.add(t)
                out.append(t)
    return out
