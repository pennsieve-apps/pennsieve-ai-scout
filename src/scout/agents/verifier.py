"""Verifier — guards against citation hallucination.

Two passes over Lit Scout's claims:

1. **Mechanical** (free, deterministic): does the cited DOI/PMID match a paper in
   the input batch, and does the verbatim `quote` substring-match that paper's
   abstract? Catches fabricated DOIs and fabricated quotes.

2. **Entailment** (LLM, ~$0.10-0.30/run): for claims that pass the mechanical
   check, ask Sonnet whether the abstract actually supports / contradicts /
   doesn't-address the claim's assertion. Catches semantic mis-interpretation.

Claims are tagged in place — none are dropped (permissive mode). Downstream
agents see verification status and can dock plausibility on unsupported claims.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

from scout.agents._common import call_json, load_prompt
from pennsieve_ai_utils.blackboard import Blackboard, Claim
from pennsieve_ai_utils.llm import LLMClient

MODEL = "synthesis"
AGENT = "verifier"

EXACT_THRESHOLD = 1.0  # full substring match (case-insensitive)
PARTIAL_THRESHOLD = 0.7  # difflib ratio above this = partial


def _normalize(s: str) -> str:
    """Lowercase + collapse whitespace, for substring/ratio matching."""
    return re.sub(r"\s+", " ", s.strip().lower())


def _strip_id_prefix(source: str) -> str:
    """Take 'doi:10.x/y' → '10.x/y'; 'pmid:123' → '123'; else unchanged."""
    s = source.strip().lower()
    for prefix in ("doi:", "pmid:"):
        if s.startswith(prefix):
            return s[len(prefix):]
    return s


def _build_paper_index(papers: list[dict]) -> dict[str, dict]:
    """Map normalized DOI/PMID/title-prefix → paper dict."""
    idx: dict[str, dict] = {}
    for p in papers:
        for raw_key in (p.get("doi", ""), p.get("pmid", ""), (p.get("title", "") or "")[:120]):
            k = _normalize(str(raw_key))
            if k:
                idx[k] = p
    return idx


def _find_paper(claim_source: str, idx: dict[str, dict]) -> dict | None:
    candidates = [
        _normalize(claim_source),
        _normalize(_strip_id_prefix(claim_source)),
    ]
    for c in candidates:
        if c in idx:
            return idx[c]
    return None


def _mechanical_check(claim: Claim, paper: dict | None) -> tuple[str, str]:
    """Returns (quote_match, verification_note)."""
    if paper is None:
        return "missing", "cited source not found in Lit Scout's input batch"
    abstract = paper.get("abstract", "") or ""
    if not claim.quote.strip():
        return "missing", "no quote provided"
    norm_q = _normalize(claim.quote)
    norm_a = _normalize(abstract)
    if not norm_a:
        return "missing", "source paper has no abstract on file"
    if norm_q in norm_a:
        return "exact", "verbatim substring match"
    ratio = SequenceMatcher(None, norm_q, norm_a).ratio()
    if ratio >= PARTIAL_THRESHOLD:
        return "partial", f"approximate match (ratio={ratio:.2f}) — possibly paraphrased"
    return "missing", f"quote not found in abstract (best ratio={ratio:.2f})"


def _format_pairs(claims_with_papers: list[tuple[int, Claim, dict]]) -> str:
    chunks = []
    for ord_id, c, paper in claims_with_papers:
        abstract = (paper.get("abstract", "") or "")[:3000]
        chunks.append(
            f"[C{ord_id}]\n"
            f"CLAIM: {c.text}\n"
            f"ABSTRACT: {abstract}"
        )
    return "\n\n".join(chunks)


def run(bb: Blackboard, client: LLMClient) -> None:
    if not bb.literature_claims:
        bb.log(AGENT, "skip", "no claims to verify")
        return

    paper_idx = _build_paper_index(bb.papers_seen or [])
    bb.log(AGENT, "index", f"{len(paper_idx)} keys over {len(bb.papers_seen or [])} papers")

    # ---- Pass 1: mechanical ----
    eligible: list[tuple[int, Claim, dict]] = []
    for i, c in enumerate(bb.literature_claims):
        paper = _find_paper(c.source, paper_idx)
        c.quote_match, c.verification_note = _mechanical_check(c, paper)
        if c.quote_match in ("exact", "partial") and paper is not None:
            eligible.append((i, c, paper))
        else:
            c.entailment = "not_checked"
    bb.log(
        AGENT,
        "mechanical",
        f"exact={sum(1 for c in bb.literature_claims if c.quote_match=='exact')}, "
        f"partial={sum(1 for c in bb.literature_claims if c.quote_match=='partial')}, "
        f"missing={sum(1 for c in bb.literature_claims if c.quote_match=='missing')}",
    )

    # ---- Pass 2: LLM entailment ----
    if not eligible:
        bb.log(AGENT, "skip_entailment", "no claims passed mechanical check")
        return

    system = load_prompt("verifier")
    user = (
        "Verify whether each abstract supports its paired claim. "
        "Return JSON only.\n\n"
        f"{_format_pairs(eligible)}"
    )

    parsed = call_json(
        client,
        agent=AGENT,
        system=system,
        user=user,
        run_id=bb.run_id,
        model=MODEL,
    )

    verdicts_by_id: dict[str, dict] = {v.get("id", ""): v for v in parsed.get("verdicts", [])}
    for ord_id, claim, _paper in eligible:
        v = verdicts_by_id.get(f"C{ord_id}")
        if v is None:
            claim.entailment = "not_checked"
            claim.verification_note = (claim.verification_note + "; entailment missing").strip("; ")
        else:
            verdict = v.get("verdict", "not_checked")
            claim.entailment = verdict if verdict in ("supports", "contradicts", "not_addressed") else "not_checked"
            note = v.get("note", "")
            claim.verification_note = (claim.verification_note + " | " + note).strip(" |")

    counts = {"supports": 0, "contradicts": 0, "not_addressed": 0, "not_checked": 0}
    for c in bb.literature_claims:
        counts[c.entailment] = counts.get(c.entailment, 0) + 1
    bb.log(
        AGENT,
        "entailment",
        f"supports={counts.get('supports',0)}, contradicts={counts.get('contradicts',0)}, "
        f"not_addressed={counts.get('not_addressed',0)}, not_checked={counts.get('not_checked',0)}",
    )
