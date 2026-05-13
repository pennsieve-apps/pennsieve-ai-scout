"""Pennsieve Discover — public neuroscience dataset API.
No auth required. Verified live 2026-05-05.

Search: GET /discover/search/datasets?query=<term>&limit=N
List:   GET /discover/datasets?limit=N
"""
from __future__ import annotations

import httpx

BASE = "https://api.pennsieve.io/discover"


def _normalize(raw: dict) -> dict:
    """Map a Pennsieve Discover record to our standardized dataset shape."""
    measurements: list[str] = []
    for tag in raw.get("tags", []) or []:
        if tag and tag not in measurements:
            measurements.append(tag)
    # modelCount tells us subjects/samples — pull subject count if present
    n_subjects = 0
    for m in raw.get("modelCount", []) or []:
        name = (m.get("modelName") or "").lower()
        if name in ("animal_subject", "human_subject", "subject"):
            n_subjects = max(n_subjects, m.get("count", 0))

    pid = raw.get("id")
    return {
        "repo": "pennsieve",
        "id": f"pennsieve:{pid}",
        "title": raw.get("name", "") or "",
        "description": raw.get("description", "") or "",
        "url": f"https://discover.pennsieve.io/datasets/{pid}",
        "doi": raw.get("doi", "") or "",
        "license": raw.get("license", "") or "",
        "size_bytes": raw.get("size", 0) or 0,
        "file_count": raw.get("fileCount", 0) or 0,
        "n_subjects": n_subjects,
        "species": "",  # not directly in search response
        "measurements": measurements,
        "_raw_organization": raw.get("organizationName", ""),
    }


def fetch_detail(dataset_id) -> dict | None:
    """GET /discover/datasets/{id}. Includes the S3 `uri`, `readme` URL,
    `externalPublications` (linked papers), `contributors`, and `tags` —
    none of which are in the search response."""
    pid = str(dataset_id).split(":")[-1]
    try:
        r = httpx.get(f"{BASE}/datasets/{pid}", timeout=30)
        r.raise_for_status()
    except Exception:
        return None
    return r.json()


def resolve_doi(doi: str) -> dict | None:
    """GET /discover/datasets/doi/{prefix}/{suffix}. Resolves a Pennsieve
    dataset DOI to the same detail record `fetch_detail` returns. Accepts
    DOIs with or without the `https://doi.org/` prefix."""
    s = (doi or "").strip()
    for p in ("https://doi.org/", "http://doi.org/", "doi.org/", "doi:"):
        if s.lower().startswith(p):
            s = s[len(p):]
            break
    if "/" not in s:
        return None
    prefix, suffix = s.split("/", 1)
    try:
        r = httpx.get(f"{BASE}/datasets/doi/{prefix}/{suffix}", timeout=30)
        r.raise_for_status()
    except Exception:
        return None
    return r.json()


def search(query: str, limit: int = 25) -> list[dict]:
    """Keyword search Pennsieve Discover. Returns normalized dataset dicts."""
    r = httpx.get(
        f"{BASE}/search/datasets",
        params={"query": query, "limit": limit},
        timeout=30,
    )
    r.raise_for_status()
    return [_normalize(d) for d in r.json().get("datasets", []) or []]
