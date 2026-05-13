"""Find datasets similar to a given one — across repositories.

Strategy: cheap and obvious — search each supported repository's catalog
using the seed's tags / measurements / keywords, union the results, score
by tag overlap (and a small boost for same-organization on Pennsieve),
return top N per repository.

The cross-repository case (e.g. Pennsieve seed with DANDI candidates) is
useful for finding *parallel modalities* that the seed's own repo doesn't
host — e.g. a Pennsieve peripheral-nerve ephys dataset and a DANDI cortical
ephys dataset, where the comparison is "same modality, different cohort/
target tissue." The Compatibility Analyst then evaluates whether each
candidate's deposit actually supports a cross-dataset hypothesis.

This catches "obviously similar" cases by keyword overlap; for better
coverage we'd embed READMEs and rank by cosine — that's v2.
"""
from __future__ import annotations

from collections import Counter

from scout.tools import pennsieve_discover


def find_similar_pennsieve(seed: dict, top_n: int = 8) -> list[dict]:
    """Given a seed dataset record (the dict returned by `pennsieve_discover.fetch_detail`
    or a normalized search result), return up to `top_n` other Pennsieve datasets
    ranked by tag overlap + same-organization boost. The seed itself is excluded.

    The returned shape matches `pennsieve_discover.search()`'s normalized output."""
    seed_id = seed.get("id")
    if isinstance(seed_id, int):
        seed_pid = f"pennsieve:{seed_id}"
    elif isinstance(seed_id, str) and seed_id.startswith("pennsieve:"):
        seed_pid = seed_id
    else:
        seed_pid = f"pennsieve:{seed_id}"

    seed_tags = [t for t in (seed.get("tags") or []) if t]
    seed_org = (seed.get("organizationName") or seed.get("_raw_organization") or "").strip()

    # Tag-search round
    candidates: dict[str, dict] = {}
    tag_hits: Counter[str] = Counter()
    for tag in seed_tags:
        try:
            results = pennsieve_discover.search(tag, limit=25)
        except Exception:
            continue
        for r in results:
            if r["id"] == seed_pid:
                continue
            candidates[r["id"]] = r
            tag_hits[r["id"]] += 1

    if not candidates:
        return []

    def score(rec: dict) -> float:
        s = float(tag_hits[rec["id"]])
        if seed_org and rec.get("_raw_organization", "") == seed_org:
            s += 0.5
        return s

    ranked = sorted(candidates.values(), key=score, reverse=True)
    return ranked[:top_n]


def find_similar_dandi(seed: dict, top_n: int = 4) -> list[dict]:
    """Given a seed (Pennsieve or DANDI), find DANDI dandisets that match
    by keyword. DANDI's catalog API supports free-text search via
    `?search=<term>`; we union results across the seed's tags and
    measurements, score by hit count, return top N.

    Returns dicts shaped like `pennsieve_discover.search`'s output (so the
    orchestrator can treat them uniformly):
        {id, title, description, measurements, species, n_subjects, repo,
         url, doi, license, size_bytes, file_count}
    """
    import httpx
    from scout.tools import dandi_discover

    DANDI_API = "https://api.dandiarchive.org/api"
    seed_id = seed.get("id")
    seed_dandi_id: str | None = None
    if isinstance(seed_id, str) and seed_id.startswith("dandi:"):
        seed_dandi_id = seed_id.split(":", 1)[1].zfill(6)

    # Build query terms from tags + measurements (drop overly-generic ones)
    GENERIC = {"human", "mouse", "rat", "cat", "dog", "imaging", "ephys",
               "electrophysiology", "data", "open"}
    terms_raw = list(seed.get("tags") or []) + list(seed.get("measurements") or [])
    terms = [t.strip() for t in terms_raw if t and t.strip().lower() not in GENERIC]
    if not terms:
        return []

    # Search each term, union and score
    candidates: dict[str, dict] = {}
    hits: Counter[str] = Counter()
    for term in terms[:8]:  # cap to 8 queries to bound API cost
        try:
            r = httpx.get(f"{DANDI_API}/dandisets/",
                          params={"search": term, "page_size": 10},
                          timeout=20)
            if r.status_code != 200:
                continue
            data = r.json()
        except httpx.HTTPError:
            continue
        for d in (data.get("results") or []):
            ident = d.get("identifier") or ""
            if not ident:
                continue
            if seed_dandi_id and ident.zfill(6) == seed_dandi_id:
                continue
            v = d.get("most_recent_published_version") or d.get("draft_version") or {}
            size = int(v.get("size") or 0)
            files = int(v.get("asset_count") or 0)
            name = v.get("name") or d.get("name") or ""
            candidates[ident] = {
                "id": f"dandi:{ident}",
                "title": name,
                "description": "",  # populated by deep inspection
                "measurements": [],
                "species": "",
                "n_subjects": 0,
                "repo": "dandi",
                "url": f"https://dandiarchive.org/dandiset/{ident}",
                "doi": "",
                "license": "",
                "size_bytes": size,
                "file_count": files,
            }
            hits[ident] += 1

    ranked = sorted(candidates.values(),
                    key=lambda r: hits[r["id"].split(":")[-1]],
                    reverse=True)
    return ranked[:top_n]


def find_similar_openneuro(seed: dict, top_n: int = 3) -> list[dict]:
    """Given a seed, find OpenNeuro datasets that match by modality + keyword.

    OpenNeuro's catalog API supports server-side `modality` filtering plus
    client-side substring matching on tags/tasks. We map the seed's
    measurements/tags onto BIDS modality codes when we can (ieeg, eeg, meg,
    mri, fmri, pet, behavioral), then text-filter on each remaining term.

    Returns dicts shaped like `pennsieve_discover.search()`'s normalized
    output so the orchestrator can treat them uniformly.
    """
    from collections import Counter

    from scout.tools import openneuro_discover

    seed_id = seed.get("id")
    seed_openneuro_id: str | None = None
    if isinstance(seed_id, str) and seed_id.startswith("openneuro:"):
        seed_openneuro_id = seed_id.split(":", 1)[1]

    GENERIC = {"human", "mouse", "rat", "cat", "dog", "data", "open"}
    terms_raw = list(seed.get("tags") or []) + list(seed.get("measurements") or [])
    terms = [t.strip() for t in terms_raw
             if t and t.strip().lower() not in GENERIC]
    if not terms:
        return []

    # Heuristic BIDS modality detection — pick the first modality token that
    # matches a known BIDS code; otherwise leave None (catalog-wide).
    BIDS_MODALITIES = {"ieeg", "ecog", "seeg", "eeg", "meg",
                       "mri", "fmri", "pet", "behavioral"}
    modality: str | None = None
    for t in terms:
        low = t.strip().lower()
        # Direct match (ieeg/ecog/seeg all listed as `ieeg` on OpenNeuro)
        if low in {"ecog", "seeg", "ieeg"}:
            modality = "ieeg"
            break
        if low in BIDS_MODALITIES:
            modality = low
            break
        # Substring (e.g. "fmri" inside "task-fmri")
        for code in ("ieeg", "ecog", "seeg", "fmri", "mri", "meg", "eeg", "pet"):
            if code in low:
                modality = "ieeg" if code in ("ecog", "seeg") else code
                break
        if modality:
            break

    candidates: dict[str, dict] = {}
    hits: Counter[str] = Counter()
    for term in terms[:8]:
        try:
            results = openneuro_discover.search(term, limit=10, modality=modality)
        except Exception:
            continue
        for r in results:
            if seed_openneuro_id and r["id"].endswith(seed_openneuro_id):
                continue
            candidates[r["id"]] = r
            hits[r["id"]] += 1

    ranked = sorted(candidates.values(),
                    key=lambda r: hits[r["id"]],
                    reverse=True)
    return ranked[:top_n]


def find_similar_cross_repo(seed: dict, *, top_n_pennsieve: int = 4,
                            top_n_dandi: int = 3,
                            top_n_openneuro: int = 3) -> list[dict]:
    """Aggregate similar-dataset search across Pennsieve, DANDI, and OpenNeuro.
    Returns the union of top candidates from each repo, tagged with `repo`.
    Each record carries enough metadata for the orchestrator to dispatch to
    the correct adapter for deep inspection.

    Falls back gracefully — if one repository's search fails, results from
    the others still come through. Order: Pennsieve, then DANDI, then OpenNeuro.
    """
    pennsieve_results: list[dict] = []
    dandi_results: list[dict] = []
    openneuro_results: list[dict] = []
    try:
        pennsieve_results = find_similar_pennsieve(seed, top_n=top_n_pennsieve)
    except Exception:
        pass
    try:
        dandi_results = find_similar_dandi(seed, top_n=top_n_dandi)
    except Exception:
        pass
    try:
        openneuro_results = find_similar_openneuro(seed, top_n=top_n_openneuro)
    except Exception:
        pass
    for r in pennsieve_results:
        r.setdefault("repo", "pennsieve")
    for r in dandi_results:
        r.setdefault("repo", "dandi")
    for r in openneuro_results:
        r.setdefault("repo", "openneuro")
    return pennsieve_results + dandi_results + openneuro_results
