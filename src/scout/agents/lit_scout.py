from __future__ import annotations

from scout.agents._common import call_json, load_prompt
from scout.blackboard import Blackboard, Claim
from scout.llm import LLMClient
from scout.tools import europe_pmc

MODEL = "sonnet"
AGENT = "lit_scout"
PER_QUERY_PAGE_SIZE = 25
BATCH_SIZE = 25  # papers per LLM call — keeps prompt size predictable and yield consistent


def _format_papers(papers: list[dict], offset: int = 0) -> str:
    chunks = []
    for i, p in enumerate(papers):
        chunks.append(
            f"[{offset + i}] {p['title']} ({p['year']})\n"
            f"    DOI: {p['doi']}\n"
            f"    Abstract: {p['abstract'][:2000]}"
        )
    return "\n\n".join(chunks)


def _format_director_context(bb: Blackboard) -> str:
    if not bb.director_brief:
        return ""
    b = bb.director_brief
    return (
        "## Director's strategic frame (use this to prioritize claims)\n\n"
        f"**Strategy.** {b.research_strategy}\n\n"
        f"**Focus areas:**\n" + "\n".join(f"- {fa}" for fa in b.focus_areas)
    )


def _dedupe_key(p: dict) -> str:
    return p.get("doi") or p.get("pmid") or p.get("title", "").strip().lower()


def _gather_papers(queries: list[str], bb: Blackboard) -> list[dict]:
    seen: dict[str, dict] = {}
    for q in queries:
        try:
            results = europe_pmc.search(q, page_size=PER_QUERY_PAGE_SIZE)
        except Exception as e:
            bb.log(AGENT, "search_error", f"query={q!r} error={e!r}")
            continue
        new_count = 0
        for p in results:
            key = _dedupe_key(p)
            if not key or key in seen:
                continue
            seen[key] = p
            new_count += 1
        bb.log(AGENT, "search", f"query={q!r} → {len(results)} hits, {new_count} new (total unique: {len(seen)})")
    return list(seen.values())


def _claim_dedupe_key(c: dict) -> tuple[str, str]:
    """Within-run dedupe so two batches don't both extract the same finding."""
    src = (c.get("source", "") or "").strip().lower()
    text_head = (c.get("text", "") or "").strip().lower()[:140]
    return (src, text_head)


def run(bb: Blackboard, client: LLMClient) -> None:
    base_queries = list(bb.search_queries) if bb.search_queries else [bb.search_query or bb.subsystem]
    director_queries = bb.director_brief.suggested_queries if bb.director_brief else []
    queries = base_queries + director_queries

    papers = _gather_papers(queries, bb)
    bb.papers_seen = papers
    bb.log(
        AGENT,
        "union",
        f"{len(base_queries)} base + {len(director_queries)} director queries → {len(papers)} unique papers with abstracts",
    )

    if not papers:
        bb.log(AGENT, "warn", "no papers found, skipping claim extraction")
        return

    system = load_prompt("lit_scout")
    director_context = _format_director_context(bb)
    batches = [papers[i:i + BATCH_SIZE] for i in range(0, len(papers), BATCH_SIZE)]
    bb.log(AGENT, "batched", f"{len(papers)} papers split into {len(batches)} batches of up to {BATCH_SIZE}")

    seen_claim_keys: set[tuple[str, str]] = set()

    for batch_idx, batch in enumerate(batches, start=1):
        offset = (batch_idx - 1) * BATCH_SIZE
        user = (
            f"Research question: {bb.research_question}\n"
            f"Subsystem: {bb.subsystem}\n\n"
            + (f"{director_context}\n\n" if director_context else "")
            + f"## Batch {batch_idx} of {len(batches)} — {len(batch)} papers (indices {offset}–{offset + len(batch) - 1})\n\n"
            + _format_papers(batch, offset=offset)
            + f"\n\nExtract claims from THIS batch. Aim for 3–8 specific claims if the papers contain real findings. "
            + "Skip vague papers without forcing extractions."
        )

        try:
            parsed = call_json(
                client,
                agent=AGENT,
                system=system,
                user=user,
                run_id=bb.run_id,
                model=MODEL,
            )
        except Exception as e:
            bb.log(AGENT, "batch_error", f"batch {batch_idx} failed: {e!r}")
            continue

        added = 0
        for c in parsed.get("claims", []):
            key = _claim_dedupe_key(c)
            if not key[0] or key in seen_claim_keys:
                continue
            seen_claim_keys.add(key)
            bb.literature_claims.append(Claim(
                text=c["text"],
                source=c["source"],
                evidence_strength=c.get("evidence_strength", "moderate"),
                quote=c.get("quote", ""),
            ))
            added += 1

        note = parsed.get("note", "") or ""
        msg = f"batch {batch_idx}/{len(batches)}: +{added} claims (running total: {len(bb.literature_claims)})"
        if note:
            msg += f" — note: {note[:140]}"
        bb.log(AGENT, "batch", msg)

    bb.log(AGENT, "extract_total", f"{len(bb.literature_claims)} unique claims across {len(batches)} batches")
