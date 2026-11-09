from __future__ import annotations

import re

from scout.agents._common import call_json, load_prompt
from pennsieve_ai_utils.blackboard import Blackboard, Claim
from pennsieve_ai_utils.llm import LLMClient
from scout.tools import europe_pmc

MODEL = "synthesis"
AGENT = "lit_scout"
PER_QUERY_PAGE_SIZE = 25
BATCH_SIZE = 25  # papers per LLM call — keeps prompt size predictable and yield consistent
BROADEN_TO = 5   # terms kept when a 0-hit query is retried broader
MAX_BROADENED = 6  # cap extra requests so a bad query set can't hammer the API

# Words that carry no retrieval signal when broadening a query.
_QUERY_STOPWORDS = frozenset(
    "a an and are as at be by for from in into of on or the to via versus vs "
    "with without using between across under toward".split()
)


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


def _dedupe_queries(queries: list[str]) -> list[str]:
    """Order-preserving dedupe, case/whitespace-insensitive.

    The director's suggested queries are copied onto bb.search_queries by the
    pipeline, so without this every query ran twice — doubling Europe PMC
    traffic for zero new papers (and tripping its throttling)."""
    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        key = " ".join(q.lower().split())
        if key and key not in seen:
            seen.add(key)
            out.append(q.strip())
    return out


def _broaden(query: str, keep: int = BROADEN_TO) -> str:
    """A wider version of a query that matched nothing.

    Europe PMC ANDs bare terms, so the framer's 8–12 word natural-language
    queries routinely match zero papers. Keep the first few content words —
    the framer front-loads the core concept — and drop the qualifier tail.
    Returns "" when broadening wouldn't change the query.
    """
    terms = [t for t in re.split(r"[^\w-]+", query) if t and t.lower() not in _QUERY_STOPWORDS]
    if len(terms) <= keep:
        return ""
    return " ".join(terms[:keep])


def _fallback_queries(bb: Blackboard) -> list[str]:
    """Last-resort queries built from what we know is real: the seed dataset's
    own title and extracted topics. These exist precisely so a run can never
    end with zero papers just because the generated queries were too narrow."""
    out: list[str] = []
    if bb.dataset_summaries:
        seed = bb.dataset_summaries[0]
        title_terms = [t for t in re.split(r"[^\w-]+", seed.title)
                       if t and t.lower() not in _QUERY_STOPWORDS]
        if title_terms:
            out.append(" ".join(title_terms[:BROADEN_TO]))
        topics = (seed.extracted_topics or [])[:2]
        if topics:
            out.append(" ".join(t.replace("-", " ") for t in topics))
    return out


def _gather_papers(queries: list[str], bb: Blackboard) -> list[dict]:
    seen: dict[str, dict] = {}
    zero_hit: list[str] = []

    def _run_query(q: str, label: str) -> None:
        try:
            results = europe_pmc.search(q, page_size=PER_QUERY_PAGE_SIZE)
        except Exception as e:
            bb.log(AGENT, f"{label}_error", f"query={q!r} error={e!r}")
            return
        new_count = 0
        for p in results:
            key = _dedupe_key(p)
            if not key or key in seen:
                continue
            seen[key] = p
            new_count += 1
        if not results and label == "search":
            zero_hit.append(q)
        bb.log(AGENT, label, f"query={q!r} → {len(results)} hits, {new_count} new (total unique: {len(seen)})")

    for q in queries:
        _run_query(q, "search")

    # Second pass: retry the misses with the qualifier tail dropped.
    broadened = []
    for q in zero_hit:
        wide = _broaden(q)
        if wide and wide.lower() not in {b.lower() for b in broadened}:
            broadened.append(wide)
    for q in broadened[:MAX_BROADENED]:
        _run_query(q, "broadened")

    return list(seen.values())


def _claim_dedupe_key(c: dict) -> tuple[str, str]:
    """Within-run dedupe so two batches don't both extract the same finding."""
    src = (c.get("source", "") or "").strip().lower()
    text_head = (c.get("text", "") or "").strip().lower()[:140]
    return (src, text_head)


def run(bb: Blackboard, client: LLMClient) -> None:
    base_queries = list(bb.search_queries) if bb.search_queries else [bb.search_query or bb.subsystem]
    director_queries = list(bb.director_brief.suggested_queries) if bb.director_brief else []
    queries = _dedupe_queries(base_queries + director_queries)

    papers = _gather_papers(queries, bb)
    if not papers:
        # Generated queries found nothing — fall back to queries derived from
        # the seed dataset itself before giving up.
        fallbacks = [q for q in _dedupe_queries(_fallback_queries(bb))
                     if q.lower() not in {x.lower() for x in queries}]
        if fallbacks:
            bb.log(AGENT, "fallback", f"0 papers from {len(queries)} queries; "
                   f"trying seed-derived: {fallbacks!r}")
            papers = _gather_papers(fallbacks, bb)

    # Keep the linked, citing and reference papers the pipeline already put
    # on the blackboard; the search results are added after them.
    known = {_dedupe_key(p) for p in bb.papers_seen}
    bb.papers_seen = list(bb.papers_seen) + [p for p in papers if _dedupe_key(p) not in known]
    bb.log(
        AGENT,
        "union",
        f"{len(queries)} unique queries → {len(papers)} unique papers with abstracts",
    )

    if not papers:
        bb.degraded.append({
            "stage": "lit_scout",
            "error": f"0 papers found across {len(queries)} queries (+fallbacks); "
                     f"no literature claims — downstream hypothesis generation will skip",
        })
        bb.log(AGENT, "degraded", "no papers found, skipping claim extraction")
        return

    system = load_prompt("lit_scout")
    director_context = _format_director_context(bb)
    batches = [papers[i:i + BATCH_SIZE] for i in range(0, len(papers), BATCH_SIZE)]
    bb.log(AGENT, "batched", f"{len(papers)} papers split into {len(batches)} batches of up to {BATCH_SIZE}")

    seen_claim_keys: set[tuple[str, str]] = set()
    failed_batches = 0

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
            failed_batches += 1
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
    if failed_batches and not bb.literature_claims:
        # Papers were found but every extraction call failed — that's an LLM
        # outage wearing an "0 claims" costume; put it on the ledger.
        bb.degraded.append({
            "stage": "lit_scout",
            "error": f"claim extraction failed for {failed_batches}/{len(batches)} batches; "
                     f"{len(papers)} papers found but 0 claims extracted",
        })
