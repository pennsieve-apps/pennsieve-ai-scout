"""Scout pipeline — extracted and adapted from `run_from_dataset` in the
monorepo's orchestrator. Runs Stage 0–1c of the pennsieve-ai research pipeline:
seed inspection, topic extraction, similar-dataset discovery, paper pulls,
dataset framer + compatibility analysis, lit scout, verifier, and (if the
RAG layer is mounted) deep-evidence retrieval.

The output is a populated Blackboard, written to `$OUTPUT_DIR/blackboard.json`.
"""
from __future__ import annotations

import json
import os

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

from scout.agents import dataset_compatibility, dataset_framer, lit_scout, verifier
from pennsieve_ai_utils.blackboard import Blackboard, DatasetSummary
from pennsieve_ai_utils.models import degraded_ok
from pennsieve_ai_utils.tools import crossref
from scout.tools import (
    dandi_discover,
    dandi_inspect,
    europe_pmc,
    openneuro_discover,
    openneuro_inspect,
    pennsieve_discover,
    pennsieve_inspect,
    similar_datasets,
    topic_extractor,
)

if TYPE_CHECKING:
    from pennsieve_ai_utils.llm import LLMClient


# Constants mirror the monorepo's orchestrator defaults
# The RAG step reads this file from Scout's output folder.
RAG_PAPERS_FILE = "rag_papers.json"

# Generic tags that should never anchor topical retrieval — they're modality
# markers, not specific topics. (Copied from orchestrator._GENERIC_TAG_STOPWORDS.)
_GENERIC_TAG_STOPWORDS = {
    "sparc", "dandi", "pennsieve", "human", "humans", "mouse", "mice",
    "rat", "rats", "data", "dataset", "modeling", "imaging",
    "electrophysiology", "open", "oad",
}


def _stage(bb: Blackboard, name: str, fn, *args, required: bool = True, **kwargs) -> bool:
    """Run one LLM-dependent stage. Returns True if it completed.

    A `required` stage failing is fatal unless `VP_DEGRADED_OK` is set, in
    which case it's recorded on `bb.degraded` and the run continues. Optional
    stages never abort the run but are still recorded, so a partial blackboard
    always says which parts are missing instead of looking complete.
    """
    try:
        fn(*args, **kwargs)
        return True
    except Exception as e:
        if required and not degraded_ok():
            raise
        detail = f"{type(e).__name__}: {e}"
        bb.degraded.append({"stage": name, "error": detail[:500]})
        bb.log(name, "degraded", detail[:300])
        print(f"  [degraded] {name} failed and was skipped: {detail[:200]}", flush=True)
        return False


def run_scout(bb: Blackboard, client: "LLMClient", seed_dataset_id: str,
              scope: str | None = None) -> None:
    """Run the scout pipeline in place on `bb`. Loads dataset, runs literature
    discovery and framing, populates dataset_summaries / papers_seen /
    literature_claims / deep_evidence / director_brief."""

    bb.deep_read_stats = {"status": "abstracts_only"}
    # --- Stage 0: deep dataset inspection ---
    print("--- inspecting seed dataset ---")
    detail, inspection = _fetch_seed(seed_dataset_id)
    if not detail:
        raise RuntimeError(f"could not fetch detail for {seed_dataset_id}")
    seed_summary = _seed_dataset_summary(seed_dataset_id, detail, inspection,
                                         relevance_note="user-selected seed")
    print(f"  README: {len(seed_summary.readme)} chars · "
          f"file types: {len(seed_summary.file_types)} · "
          f"record tables: {len(seed_summary.record_tables)} · "
          f"linked pubs: {len(seed_summary.external_publications)}")
    bb.dataset_summaries.append(seed_summary)
    bb.log("seed_inspection", "diagnostics", str(inspection.get("diagnostics", [])))
    if not seed_summary.file_types and not seed_summary.record_tables:
        message = (f"{seed_dataset_id}: no inspectable files or record tables. "
                   "Check Discover metadata/download access and legacy S3 Requester Pays "
                   "permissions before generating hypotheses.")
        bb.log("seed_inspection", "blocked", message)
        if not degraded_ok():
            raise RuntimeError(message)
        bb.degraded.append({"stage": "seed_inspection", "error": message})

    # Topic extraction (LLM): drives the retrieval topical filter
    try:
        structured = topic_extractor.extract_topics(
            client,
            title=seed_summary.title,
            abstract=seed_summary.description,
            body_excerpt=seed_summary.readme,
            run_id=bb.run_id,
        )
        seed_summary.extracted_topics = topic_extractor.topics_for_retrieval(structured)
        if structured:
            bb.log(
                "pipeline", "seed_topics_full",
                "domain=[" + ",".join(structured.get("domain", [])) + "] "
                "broader=[" + ",".join(structured.get("broader", [])) + "] "
                "modality=[" + ",".join(structured.get("modality", [])) + "] "
                "species=[" + ",".join(structured.get("species", [])) + "] "
                "technique=[" + ",".join(structured.get("technique", [])) + "]",
            )
    except Exception as e:
        # Topics anchor the RAG topical filter; a failed extraction quietly
        # weakens retrieval for the whole run, so it goes on the ledger.
        detail = f"{type(e).__name__}: {e}"
        bb.degraded.append({"stage": "topic_extractor", "error": detail[:500]})
        bb.log("topic_extractor", "degraded", detail[:300])
    if seed_summary.extracted_topics:
        bb.log("pipeline", "seed_topics",
               f"using for filter: {','.join(seed_summary.extracted_topics)}")

    # --- Stage 0b: similar datasets (cross-repo) ---
    print("\n--- finding similar datasets (cross-repo) ---")
    similar = similar_datasets.find_similar_cross_repo(
        detail, top_n_pennsieve=4, top_n_dandi=3, top_n_openneuro=3,
    )
    n_p = sum(1 for s in similar if s.get("repo") == "pennsieve")
    n_d = sum(1 for s in similar if s.get("repo") == "dandi")
    n_o = sum(1 for s in similar if s.get("repo") == "openneuro")
    print(f"  found {len(similar)} similar datasets ({n_p} pennsieve + {n_d} dandi + {n_o} openneuro)")
    for s in similar:
        summary = _similar_to_summary(s)
        sim_repo = s.get("repo", "pennsieve")
        sim_id = s["id"].split(":", 1)[1] if ":" in s["id"] else s["id"]
        try:
            if sim_repo == "pennsieve":
                sim_detail = pennsieve_discover.fetch_detail(sim_id)
                if sim_detail and sim_detail.get("uri"):
                    sim_inspection = pennsieve_inspect.inspect(sim_detail["uri"], sim_detail)
                    summary.external_publications = list(
                        sim_detail.get("externalPublications") or []
                    )
                else:
                    sim_inspection = {}
            elif sim_repo == "dandi":
                sim_detail = dandi_discover.fetch_detail(sim_id)
                if sim_detail:
                    summary.title = sim_detail.get("title") or summary.title
                    summary.description = sim_detail.get("description") or ""
                    summary.species = sim_detail.get("species") or ""
                    summary.n_subjects = int(sim_detail.get("n_subjects") or 0)
                    summary.measurements = list(sim_detail.get("measurements") or [])
                    summary.doi = sim_detail.get("doi") or ""
                    summary.url = sim_detail.get("url") or ""
                    summary.external_publications = list(
                        sim_detail.get("externalPublications") or []
                    )
                    sim_inspection = dandi_inspect.inspect(
                        sim_id, version=sim_detail.get("version") or "draft",
                    )
                else:
                    sim_inspection = {}
            elif sim_repo == "openneuro":
                sim_detail = openneuro_discover.fetch_detail(sim_id)
                if sim_detail:
                    summary.title = sim_detail.get("title") or summary.title
                    summary.description = sim_detail.get("description") or ""
                    summary.species = sim_detail.get("species") or ""
                    summary.n_subjects = int(sim_detail.get("n_subjects") or 0)
                    summary.measurements = list(sim_detail.get("measurements") or [])
                    summary.doi = sim_detail.get("doi") or ""
                    summary.url = sim_detail.get("url") or ""
                    summary.external_publications = list(
                        sim_detail.get("externalPublications") or []
                    )
                    sim_inspection = openneuro_inspect.inspect(sim_id)
                else:
                    sim_inspection = {}
            else:
                sim_inspection = {}
            summary.readme = sim_inspection.get("readme") or ""
            summary.file_types = sim_inspection.get("file_types") or {}
            summary.file_tree = sim_inspection.get("file_tree") or ""
            summary.record_tables = sim_inspection.get("records") or {}
            summary.bucket_uri = sim_inspection.get("bucket_uri") or ""
            summary.access_type = sim_inspection.get("access") or ""
        except Exception as e:
            bb.log("pipeline", "similar_inspect_warn", f"{s['id']}: {e!r}")
        bb.dataset_summaries.append(summary)
        bb.log("pipeline", "similar",
               f"{s['id']} ({summary.access_type or '?'}): "
               f"readme={len(summary.readme)} chars, file_tree={'yes' if summary.file_tree else 'no'}")

    # --- Stage 0c: linked + citing + reference papers ---
    print("\n--- pulling linked publications, citing papers, and one-hop references ---")
    linked_papers = _pull_linked_paper_records(seed_summary.external_publications, bb)
    citing_papers = _pull_citing_papers(seed_summary.external_publications, bb, limit=30)
    reference_papers = _pull_reference_papers(seed_summary.external_publications, bb, limit=10)
    print(f"  linked: {len(linked_papers)}, citing: {len(citing_papers)}, references: {len(reference_papers)}")
    for p in linked_papers + citing_papers + reference_papers:
        if p not in bb.papers_seen:
            bb.papers_seen.append(p)

    # --- Stage 0c2: Dataset Compatibility ---
    if len(bb.dataset_summaries) >= 2:
        print("\n--- dataset_compatibility ---")
        # Optional: cross-dataset verdicts are useful but nothing downstream
        # requires them, so this has never aborted the run.
        # Required: the framer and Hypothesize rely on these verdicts, and a
        # missing assessment would otherwise read as "nothing is compatible".
        _stage(bb, "dataset_compatibility", dataset_compatibility.run,
               bb, client, required=True)

    # --- Stage 0d: Dataset Framer ---
    print("\n--- dataset_framer ---")
    _stage(
        bb, "dataset_framer", dataset_framer.run,
        bb, client,
        seed_dataset_id=seed_dataset_id,
        linked_papers=linked_papers,
        citing_papers=citing_papers,
        reference_papers=reference_papers,
        scope_override=scope,
    )

    if bb.director_brief and bb.director_brief.suggested_queries:
        bb.search_queries = list(bb.director_brief.suggested_queries)
    else:
        bb.search_queries = []

    # --- Stage 1: Lit Scout ---
    print("\n--- lit_scout ---")
    _stage(bb, "lit_scout", lit_scout.run, bb, client)

    # --- Stage 1b: Verifier ---
    print("\n--- verifier ---")
    _stage(bb, "verifier", verifier.run, bb, client)
    _enrich_dois(bb)

    # --- Stage 1c: choose papers for the RAG index ---
    # The RAG step runs after Scout and indexes these papers' full text along
    # with the dataset's own links; Hypothesize then searches the index.
    requested = _papers_for_rag(bb)
    bb.deep_read_stats = {**bb.deep_read_stats, "requested_papers": requested}
    if os.environ.get("OUTPUT_DIR"):
        out = Path(os.environ["OUTPUT_DIR"])
        out.mkdir(parents=True, exist_ok=True)
        (out / RAG_PAPERS_FILE).write_text(json.dumps({"papers": requested}, indent=2) + "\n")
    bb.log("pipeline", "rag_papers",
           f"{len(requested)} papers chosen for the RAG index: "
           + ", ".join(p["doi"] for p in requested))
    print(f"  chose {len(requested)} papers for the RAG index ({RAG_PAPERS_FILE})")

    # Re-enrich DOIs once more
    _enrich_dois(bb)
    print(f"\nScout complete: {len(bb.dataset_summaries)} datasets, "
          f"{len(bb.papers_seen)} papers, "
          f"{len(bb.literature_claims)} claims, "
          f"{len(bb.deep_evidence)} deep evidence chunks.")


# ---------------------------------------------------------------------------
# Helpers (lifted from orchestrator.py with imports adapted)
# ---------------------------------------------------------------------------

def _parse_seed_id(seed_dataset_id: str) -> tuple[str, str]:
    if ":" in seed_dataset_id:
        repo, sid = seed_dataset_id.split(":", 1)
        return repo.lower(), sid
    return "pennsieve", seed_dataset_id


def _fetch_seed(seed_dataset_id: str) -> tuple[dict | None, dict]:
    repo, sid = _parse_seed_id(seed_dataset_id)
    if repo == "pennsieve":
        detail = pennsieve_discover.fetch_detail(sid)
        if not detail:
            return None, {}
        uri = detail.get("uri") or ""
        inspection = (pennsieve_inspect.inspect(uri, detail) if uri
                      else {"readme": "", "file_types": {}, "records": {}})
        return detail, inspection
    if repo == "dandi":
        detail = dandi_discover.fetch_detail(sid)
        if not detail:
            return None, {}
        inspection = dandi_inspect.inspect(sid, version=detail.get("version") or "draft")
        return detail, inspection
    if repo == "openneuro":
        detail = openneuro_discover.fetch_detail(sid)
        if not detail:
            return None, {}
        inspection = openneuro_inspect.inspect(sid)
        return detail, inspection
    raise ValueError(
        f"Unsupported repository prefix in {seed_dataset_id!r}. "
        f"Supported: pennsieve:<id>, dandi:<id>, openneuro:<id>."
    )


def _seed_dataset_summary(seed_dataset_id: str, detail: dict, inspection: dict,
                          relevance_note: str) -> DatasetSummary:
    repo, sid = _parse_seed_id(seed_dataset_id)
    if repo == "pennsieve":
        url = f"https://discover.pennsieve.io/datasets/{sid}"
        title = detail.get("name", "") or ""
        size_bytes = int(detail.get("size", 0) or 0)
        file_count = int(detail.get("fileCount", 0) or 0)
        n_subjects = 0
        species = ""
        measurements = list(detail.get("tags") or [])
    elif repo == "dandi":
        url = detail.get("url", "") or ""
        title = detail.get("title", "") or ""
        size_bytes = int(detail.get("size_bytes", 0) or 0)
        file_count = int(detail.get("file_count", 0) or 0)
        n_subjects = int(detail.get("n_subjects", 0) or 0)
        species = detail.get("species", "") or ""
        measurements = list(detail.get("measurements") or []) + list(detail.get("tags") or [])
    elif repo == "openneuro":
        url = detail.get("url", "") or ""
        title = detail.get("title", "") or ""
        size_bytes = int(detail.get("size_bytes", 0) or 0)
        file_count = int(detail.get("file_count", 0) or 0)
        n_subjects = int(detail.get("n_subjects", 0) or 0)
        species = detail.get("species", "") or ""
        measurements = list(detail.get("measurements") or []) + list(detail.get("tags") or [])
    else:
        url = ""
        title = detail.get("title", "") or detail.get("name", "") or ""
        size_bytes = 0
        file_count = 0
        n_subjects = 0
        species = ""
        measurements = list(detail.get("tags") or [])

    return DatasetSummary(
        id=f"{repo}:{sid}",
        title=title,
        description=detail.get("description", "") or "",
        measurements=measurements,
        species=species,
        n_subjects=n_subjects,
        repo=repo,
        url=url,
        doi=detail.get("doi", "") or "",
        license=detail.get("license", "") or "",
        size_bytes=size_bytes,
        file_count=file_count,
        relevance_note=relevance_note,
        readme=inspection.get("readme") or "",
        file_types=inspection.get("file_types") or {},
        file_tree=inspection.get("file_tree") or "",
        record_tables=inspection.get("records") or {},
        external_publications=list(detail.get("externalPublications") or []),
        bucket_uri=inspection.get("bucket_uri") or "",
        access_type=inspection.get("access") or "",
    )


def _similar_to_summary(rec: dict) -> DatasetSummary:
    return DatasetSummary(
        id=rec["id"],
        title=rec.get("title", "") or "",
        description=rec.get("description", "") or "",
        measurements=list(rec.get("measurements") or []),
        species=rec.get("species", "") or "",
        n_subjects=int(rec.get("n_subjects", 0) or 0),
        repo=rec.get("repo", ""),
        url=rec.get("url", "") or "",
        doi=rec.get("doi", "") or "",
        license=rec.get("license", "") or "",
        size_bytes=int(rec.get("size_bytes", 0) or 0),
        file_count=int(rec.get("file_count", 0) or 0),
        relevance_note="similar by tag overlap",
    )


def _pull_linked_paper_records(external_pubs: list[dict], bb: Blackboard) -> list[dict]:
    """Convert externalPublications DOIs into full paper records with abstracts."""
    out: list[dict] = []
    for pub in external_pubs or []:
        doi = (pub.get("doi") or "").strip()
        if not doi:
            continue
        rec: dict | None = None
        if "protocols.io" not in doi:
            try:
                hits = europe_pmc.search(f'DOI:"{doi}"', page_size=1)
                if hits:
                    rec = hits[0]
            except Exception as e:
                bb.log("pipeline", "linked_pub_pmc_warn", f"{doi}: {e!r}")
        if rec is None:
            try:
                rec = crossref.fetch_doi_metadata(doi)
            except Exception as e:
                bb.log("pipeline", "linked_pub_crossref_warn", f"{doi}: {e!r}")
        if rec is None:
            rec = {"doi": doi, "title": "", "abstract": "", "year": "",
                   "journal": "", "source": "stub"}
        out.append(rec)
    bb.log("pipeline", "linked_pubs",
           f"{len(out)} records (incl. protocols.io: "
           f"{sum(1 for r in out if 'protocols.io' in (r.get('doi') or ''))})")
    return out


def _pull_reference_papers(external_pubs: list[dict], bb: Blackboard,
                           limit: int = 10) -> list[dict]:
    """One-hop backward citation: papers REFERENCED by each non-protocol linked
    publication. Top-N by cited_by_count. Dedups by DOI across all parents."""
    seen: dict[str, dict] = {}
    for pub in external_pubs or []:
        doi = (pub.get("doi") or "").strip()
        if not doi or "protocols.io" in doi:
            continue
        try:
            refs = europe_pmc.references(doi, limit=limit)
        except Exception as e:
            bb.log("pipeline", "refs_warn", f"{doi}: {e!r}")
            continue
        for r in refs or []:
            rdoi = (r.get("doi") or "").lower()
            if rdoi and rdoi not in seen:
                seen[rdoi] = r
    bb.log("pipeline", "reference_papers", f"{len(seen)} unique references")
    return list(seen.values())


def _pull_citing_papers(external_pubs: list[dict], bb: Blackboard,
                        limit: int = 30) -> list[dict]:
    """Forward citation: papers CITING each linked publication."""
    seen: dict[str, dict] = {}
    for pub in external_pubs or []:
        doi = (pub.get("doi") or "").strip()
        if not doi or "protocols.io" in doi:
            continue
        try:
            cites = europe_pmc.citations(doi, limit=limit)
        except Exception as e:
            bb.log("pipeline", "citing_warn", f"{doi}: {e!r}")
            continue
        for c in cites or []:
            cdoi = (c.get("doi") or "").lower()
            if cdoi and cdoi not in seen:
                seen[cdoi] = c
    bb.log("pipeline", "citing_papers", f"{len(seen)} unique citing papers")
    return list(seen.values())


def _enrich_dois(bb: Blackboard) -> None:
    """Populate bb.papers_enriched with authoritative metadata (CrossRef) for
    every DOI cited across literature_claims, papers_seen, and prior_papers."""
    cited: set[str] = set()
    for c in bb.literature_claims:
        s = (c.source or "").strip().lower()
        if s.startswith("doi:"):
            cited.add(s[4:])
    for p in bb.papers_seen:
        doi = (p.get("doi") or "").strip().lower()
        if doi:
            cited.add(doi)
    pending = [doi for doi in cited if doi not in bb.papers_enriched]
    if not pending:
        return

    def fetch(doi: str) -> tuple[str, dict | None]:
        try:
            return doi, crossref.fetch_doi_metadata(doi)
        except Exception:
            return doi, None

    with ThreadPoolExecutor(max_workers=8) as pool:
        for doi, rec in pool.map(fetch, pending):
            if rec:
                bb.papers_enriched[doi] = {
                    "title": rec.get("title", ""),
                    "authors": rec.get("authors", []),
                    "year": rec.get("year", ""),
                    "journal": rec.get("journal", ""),
                }


def _papers_for_rag(bb: Blackboard, limit: int = 10) -> list[dict]:
    """Papers whose full text is worth indexing: those cited by verified claims.

    Ranked by open access first (only open-access papers have full text to
    index), then by how many supported claims cite them. Papers no supported
    claim cites are left out.
    """
    counts: dict[str, int] = {}
    for c in bb.literature_claims:
        if c.entailment != "supports":
            continue
        doi = _bare_doi(c.source)
        if doi:
            counts[doi] = counts.get(doi, 0) + 1
    meta = {_bare_doi(p.get("doi", "")): p for p in bb.papers_seen}
    ranked = sorted(counts, key=lambda d: (not meta.get(d, {}).get("is_open_access", False),
                                           -counts[d], d))
    return [{"doi": d, "supported_claims": counts[d],
             "open_access": bool(meta.get(d, {}).get("is_open_access", False)),
             "title": meta.get(d, {}).get("title", "")}
            for d in ranked[:limit]]


def _bare_doi(source: str) -> str:
    s = (source or "").strip()
    for prefix in ("doi:", "https://doi.org/", "http://doi.org/"):
        if s.lower().startswith(prefix):
            s = s[len(prefix):]
    return s.lower() if s.startswith("10.") else ""


def _run_topics_filter(bb: Blackboard) -> set[str] | None:
    """Topical filter for retrieval — prefers extracted topics, falls back to
    filtered measurements."""
    if not bb.dataset_summaries:
        return None
    seed = bb.dataset_summaries[0]
    if seed.extracted_topics:
        return {t.strip().lower() for t in seed.extracted_topics if t and t.strip()}
    topics: set[str] = set()
    for tag in (seed.measurements or []):
        t = (tag or "").strip().lower().replace(" ", "-").replace("_", "-")
        if t and t not in _GENERIC_TAG_STOPWORDS:
            topics.add(t)
    return topics or None
