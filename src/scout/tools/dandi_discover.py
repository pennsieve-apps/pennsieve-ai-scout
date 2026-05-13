"""DANDI Archive catalog client — minimal mirror of `pennsieve_discover`.

DANDI hosts neurophysiology and neuroimaging datasets, mostly NWB-formatted,
under the BRAIN Initiative. Public read API at https://api.dandiarchive.org —
no auth required for open datasets.

Surfaces:
  - `fetch_detail(dandiset_id)` → record with title, description, access mode,
    contributors, license, version, asset count, total size, and embedded
    metadata.
  - `fetch_assets(dandiset_id, version, limit)` → list of asset records
    `{path, size, asset_id, encoding_format}`.

The dataset URI for downstream stages is the dandiset's API base, since
DANDI assets stream via the API rather than a direct S3 prefix the agent
can browse.
"""
from __future__ import annotations

import httpx

BASE = "https://api.dandiarchive.org/api"


def fetch_detail(dandiset_id: str, version: str = "draft") -> dict | None:
    """Top-level dandiset metadata. `version` is typically 'draft' for the
    actively-edited copy or a date-versioned tag like '0.230101.1234'.

    Returns a dict shaped like:
        {
          "id": "dandi:000001",
          "title": ...,
          "description": ...,
          "tags": [...],
          "license": ...,
          "size_bytes": int,
          "file_count": int,
          "n_subjects": int,
          "species": str,
          "measurements": [...],
          "external_publications": [{doi, relationshipType}, ...],
          "doi": str,
          "url": "https://dandiarchive.org/dandiset/000001/draft",
          "uri": <api endpoint base used by the inspector>,
          "version": <resolved version string>,
        }
    """
    norm = dandiset_id.split(":", 1)[-1].zfill(6) if dandiset_id else ""
    if not norm:
        return None
    try:
        # Resolve the version if 'draft' isn't appropriate
        r = httpx.get(f"{BASE}/dandisets/{norm}/", timeout=20)
        if r.status_code != 200:
            return None
        d = r.json()
    except httpx.HTTPError:
        return None

    # Pick the most-recent published version, falling back to draft
    versions = d.get("most_recent_published_version") or d.get("draft_version") or {}
    resolved_version = versions.get("version") or version

    try:
        v = httpx.get(
            f"{BASE}/dandisets/{norm}/versions/{resolved_version}/info/",
            timeout=20,
        )
        if v.status_code != 200:
            return None
        meta = v.json().get("metadata") or {}
    except httpx.HTTPError:
        meta = {}

    # DANDI metadata follows the dandi-schema; extract what we need
    name = meta.get("name") or d.get("name") or ""
    description = meta.get("description") or ""
    keywords = meta.get("keywords") or []
    license_ = meta.get("license") or []
    license_str = ", ".join(license_) if isinstance(license_, list) else str(license_)
    contributors = meta.get("contributor") or []
    species_set = set()
    measurements = set()
    n_subjects = 0
    if isinstance(meta.get("assetsSummary"), dict):
        sm = meta["assetsSummary"]
        n_subjects = int(sm.get("numberOfSubjects") or 0)
        for s in (sm.get("species") or []):
            n = (s.get("name") or "").strip()
            if n:
                species_set.add(n)
        for m in (sm.get("measurementTechnique") or []):
            n = (m.get("name") or "").strip()
            if n:
                measurements.add(n)
        size_bytes = int(sm.get("numberOfBytes") or 0)
        file_count = int(sm.get("numberOfFiles") or 0)
    else:
        size_bytes = int(d.get("draft_version", {}).get("size") or 0)
        file_count = int(d.get("draft_version", {}).get("asset_count") or 0)

    related_publications = []
    for rp in (meta.get("relatedResource") or []):
        ident = rp.get("identifier") or ""
        rel = rp.get("relation") or ""
        if "doi" in ident.lower() or "10." in ident:
            doi = ident.replace("https://doi.org/", "").replace("doi:", "").strip()
            if doi:
                related_publications.append({"doi": doi, "relationshipType": rel})

    return {
        "id": f"dandi:{norm}",
        "title": name,
        "description": description,
        "tags": keywords,
        "license": license_str,
        "size_bytes": size_bytes,
        "file_count": file_count,
        "n_subjects": n_subjects,
        "species": ", ".join(sorted(species_set)) if species_set else "",
        "measurements": sorted(measurements),
        "externalPublications": related_publications,
        "doi": (meta.get("doi") or "").strip(),
        "url": f"https://dandiarchive.org/dandiset/{norm}/{resolved_version}",
        "uri": f"{BASE}/dandisets/{norm}/versions/{resolved_version}",
        "version": resolved_version,
        "_raw_metadata": meta,  # for downstream consumers that need full schema
    }


def fetch_assets(dandiset_id: str, version: str = "draft",
                 limit: int | None = None) -> list[dict]:
    """List a dandiset's assets. Each asset is a NWB (or other) file with a
    `path`, `size`, `asset_id`, and `encodingFormat`. DANDI paginates at
    1000-per-page; we walk forward until exhausted or `limit` is reached.

    Returns a list of `{path, size, asset_id, encoding_format}` dicts —
    shaped to feed `inspector_helpers.render_file_tree` directly.
    """
    norm = dandiset_id.split(":", 1)[-1].zfill(6) if dandiset_id else ""
    if not norm:
        return []
    out: list[dict] = []
    url: str | None = (
        f"{BASE}/dandisets/{norm}/versions/{version}/assets/?page_size=1000"
    )
    while url:
        try:
            r = httpx.get(url, timeout=30)
            if r.status_code != 200:
                break
            data = r.json()
        except httpx.HTTPError:
            break
        for a in (data.get("results") or []):
            out.append({
                "path": a.get("path", ""),
                "size": int(a.get("size") or 0),
                "asset_id": a.get("asset_id") or a.get("identifier") or "",
                "encoding_format": a.get("encodingFormat") or
                                    a.get("encoding_format") or "",
                # Map encodingFormat → fileType for compatibility with the
                # generic inspector_helpers.file_type_summary
                "fileType": _format_to_filetype(
                    a.get("encodingFormat") or "", a.get("path", "")
                ),
            })
            if limit is not None and len(out) >= limit:
                return out
        url = data.get("next")
    return out


def _format_to_filetype(encoding: str, path: str) -> str:
    """Map DANDI encodingFormat (e.g. 'application/x-nwb', 'video/mp4') to
    a short file-type label compatible with our histograms. Falls back to
    file extension."""
    if "nwb" in encoding.lower():
        return "NWB"
    if encoding.startswith("image/"):
        return "Image"
    if encoding.startswith("video/"):
        return "Video"
    if encoding.startswith("application/json"):
        return "Json"
    if path:
        ext = path.rsplit(".", 1)[-1].upper() if "." in path else "?"
        return ext
    return "?"


def fetch_readme(dandiset_id: str, version: str = "draft") -> str:
    """DANDI doesn't have a README.md by convention; the description in the
    metadata serves the same role. This function returns it for symmetry
    with `pennsieve_inspect.fetch_readme`. Falls back to ''."""
    detail = fetch_detail(dandiset_id, version)
    return (detail or {}).get("description") or ""
