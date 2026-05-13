"""OpenAlex — find papers that cite a given DOI.

Used by the Dataset Framer to learn what's been done with a dataset since its
primary publication. Free, no auth required, generous rate limits.

Endpoint: https://api.openalex.org/works?filter=cites:W<id>
where W<id> is OpenAlex's internal work id, looked up via the DOI.
"""
from __future__ import annotations

import httpx

BASE = "https://api.openalex.org"


def _work_id_for_doi(doi: str) -> str | None:
    doi_norm = doi.strip().lower().removeprefix("https://doi.org/").removeprefix("doi:")
    try:
        r = httpx.get(f"{BASE}/works/doi:{doi_norm}", timeout=20)
        if r.status_code != 200:
            return None
        return (r.json().get("id") or "").rsplit("/", 1)[-1] or None
    except httpx.HTTPError:
        return None


def citing_papers(doi: str, limit: int = 50) -> list[dict]:
    """Return up to `limit` papers that cite the given DOI, newest first.

    Each entry: {doi, title, year, abstract, authors, journal}.
    Empty list on failure (no work id found, network error, etc.)."""
    wid = _work_id_for_doi(doi)
    if not wid:
        return []
    out: list[dict] = []
    cursor = "*"
    while cursor and len(out) < limit:
        try:
            r = httpx.get(
                f"{BASE}/works",
                params={
                    "filter": f"cites:{wid}",
                    "per-page": min(50, limit - len(out)),
                    "sort": "publication_date:desc",
                    "cursor": cursor,
                },
                timeout=30,
            )
            r.raise_for_status()
        except httpx.HTTPError:
            break
        body = r.json()
        for w in body.get("results", []):
            out.append(_normalize(w))
            if len(out) >= limit:
                break
        cursor = (body.get("meta") or {}).get("next_cursor")
    return out


def referenced_papers(doi: str, limit: int = 10) -> list[dict]:
    """Return up to `limit` papers that the given DOI cites, ranked by their
    own citation count (most-cited first — a relevance proxy: foundational
    references tend to be the heavily-cited ones across the field).

    These are the dataset's primary publication's *backward* citations —
    complement to `citing_papers` which goes forward. Useful for the Framer's
    'what's established' field-landscape paragraph.
    """
    wid = _work_id_for_doi(doi)
    if not wid:
        return []
    try:
        r = httpx.get(f"{BASE}/works/{wid}", timeout=20)
        if r.status_code != 200:
            return []
        ref_urls = r.json().get("referenced_works") or []
    except httpx.HTTPError:
        return []
    if not ref_urls:
        return []
    # Cap raw refs before batch lookup; OpenAlex `ids.openalex` filter accepts
    # up to ~50 per page.
    ref_ids = [u.rsplit("/", 1)[-1] for u in ref_urls[:200] if u.rsplit("/", 1)[-1].startswith("W")]
    if not ref_ids:
        return []
    out: list[dict] = []
    # Page through in chunks of 50, sorted by cited_by_count desc, take top-N overall
    for i in range(0, len(ref_ids), 50):
        chunk = ref_ids[i:i + 50]
        try:
            r = httpx.get(
                f"{BASE}/works",
                params={
                    "filter": f"openalex:{'|'.join(chunk)}",
                    "per-page": 50,
                    "sort": "cited_by_count:desc",
                },
                timeout=30,
            )
            r.raise_for_status()
        except httpx.HTTPError:
            continue
        for w in r.json().get("results", []):
            out.append(_normalize(w))
        if len(out) >= limit * 3:  # collected enough to pick the top-N
            break
    out.sort(key=lambda p: -(p.get("cited_by_count") or 0))
    return out[:limit]


def _normalize(w: dict) -> dict:
    abst = _reconstruct_abstract(w.get("abstract_inverted_index"))
    venue = ((w.get("primary_location") or {}).get("source") or {}).get("display_name", "")
    authors = [
        (a.get("author") or {}).get("display_name", "")
        for a in (w.get("authorships") or [])[:8]
    ]
    return {
        "doi": (w.get("doi") or "").removeprefix("https://doi.org/"),
        "title": w.get("title") or w.get("display_name") or "",
        "year": w.get("publication_year") or "",
        "abstract": abst,
        "authors": ", ".join(a for a in authors if a),
        "journal": venue,
        "openalex_id": (w.get("id") or "").rsplit("/", 1)[-1],
        "cited_by_count": w.get("cited_by_count") or 0,
    }


def _reconstruct_abstract(inv_idx: dict | None) -> str:
    """OpenAlex stores abstracts as an inverted index for licensing reasons.
    Reconstruct the prose by sorting words by their positions."""
    if not inv_idx:
        return ""
    pairs: list[tuple[int, str]] = []
    for word, positions in inv_idx.items():
        for p in positions:
            pairs.append((p, word))
    pairs.sort()
    return " ".join(w for _, w in pairs)
