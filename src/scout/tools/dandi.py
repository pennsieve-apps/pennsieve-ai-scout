"""DANDI Archive — public neuroscience data (mostly NWB-format neurophysiology).
No auth required for OpenAccess dandisets. Verified live 2026-05-05.

Search: GET /api/dandisets/?search=<term>&page_size=N
Detail: GET /api/dandisets/<id>/versions/<version>/info/
"""
from __future__ import annotations

import httpx

BASE = "https://api.dandiarchive.org/api"


def _pick_version(search_record: dict) -> str | None:
    """Prefer most_recent_published_version, fall back to draft."""
    mrpv = search_record.get("most_recent_published_version")
    if mrpv:
        return mrpv.get("version")
    dv = search_record.get("draft_version")
    if dv:
        return dv.get("version")
    return None


def _detail(identifier: str, version: str) -> dict | None:
    """Fetch full version metadata for a single dandiset."""
    try:
        r = httpx.get(f"{BASE}/dandisets/{identifier}/versions/{version}/info/", timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _normalize(search_record: dict, info: dict | None) -> dict:
    """Map (search-record from list endpoint, optional version-info from detail
    endpoint) to our standard shape. Search-record shape:
      {identifier, most_recent_published_version, draft_version, ...}
    Detail-info shape (when fetched):
      {version, name, asset_count, size, dandiset: {...}, metadata: {...}, ...}
    """
    identifier = search_record["identifier"]
    info = info or {}
    meta = info.get("metadata") or {}
    # Fall back to versions-from-search-record when detail wasn't fetched
    version_blob = (
        search_record.get("most_recent_published_version")
        or search_record.get("draft_version")
        or {}
    )

    measurements: list[str] = []
    for kw in meta.get("keywords", []) or []:
        if kw and kw not in measurements:
            measurements.append(kw)
    # `about` contains UBERON anatomy + DOID disorder terms — useful for matching
    for a in meta.get("about", []) or []:
        name = a.get("name") if isinstance(a, dict) else None
        if name and name not in measurements:
            measurements.append(name)

    # species: assetsSummary.species[].name (each is dict with name)
    species = ""
    asum = meta.get("assetsSummary") or {}
    sp_list = asum.get("species") or []
    if sp_list:
        species = ", ".join(s.get("name", "") for s in sp_list if isinstance(s, dict))

    n_subjects = asum.get("numberOfSubjects", 0) or 0

    # license is a list of SPDX strings
    lic_list = meta.get("license") or []
    lic = ", ".join(lic_list) if isinstance(lic_list, list) else str(lic_list or "")

    # DOI: extract from citation if present (format: "... https://doi.org/10.x/y")
    doi = ""
    citation = meta.get("citation", "") or ""
    if "doi.org/" in citation:
        doi = citation.split("doi.org/")[-1].strip().rstrip(".")

    name = info.get("name") or version_blob.get("name") or ""
    description = meta.get("description", "") or name

    return {
        "repo": "dandi",
        "id": f"dandi:{identifier}",
        "title": name,
        "description": description,
        "url": f"https://dandiarchive.org/dandiset/{identifier}",
        "doi": doi,
        "license": lic,
        "size_bytes": info.get("size") or version_blob.get("size") or 0,
        "file_count": info.get("asset_count") or version_blob.get("asset_count") or 0,
        "n_subjects": n_subjects,
        "species": species,
        "measurements": measurements,
    }


def search(query: str, limit: int = 25, fetch_details: bool = True) -> list[dict]:
    """Keyword search DANDI. If fetch_details=True (default), also pulls each
    dandiset's version-info for richer metadata (one extra request per hit)."""
    r = httpx.get(
        f"{BASE}/dandisets/",
        params={"search": query, "page_size": limit},
        timeout=30,
    )
    r.raise_for_status()
    results = r.json().get("results", []) or []

    out: list[dict] = []
    for rec in results:
        version = _pick_version(rec)
        info = _detail(rec["identifier"], version) if (fetch_details and version) else None
        out.append(_normalize(rec, info))
    return out
