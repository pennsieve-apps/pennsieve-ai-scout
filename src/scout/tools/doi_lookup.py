"""Caching DOI metadata lookup — single source of truth for author/year/title
strings used in inline citations across the pipeline.

Backed by CrossRef (universal DOI registry; covers protocols.io, papers,
datasets, preprints, conference papers). Persistent cache at
`runs/_knowledge/doi_cache.json` so the same DOI is fetched once and reused
by every agent that needs to render an inline citation. This eliminates
the hallucinated-author failure mode that bit the Paper Writer.

Public API:
  - `lookup(doi)` → metadata dict or None (with persistent caching)
  - `lookup_batch(dois)` → {doi: metadata}
  - `author_string(doi, fallback)` → "Bayles et al., 2018" formatting helper
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from scout.tools import crossref, datacite

_CACHE_PATH = Path("runs/_knowledge/doi_cache.json")
_lock = threading.Lock()
_cache: dict | None = None


def _normalize(doi: str) -> str:
    s = (doi or "").strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s.strip()


def _load_cache() -> dict:
    global _cache
    if _cache is not None:
        return _cache
    if _CACHE_PATH.exists():
        try:
            _cache = json.loads(_CACHE_PATH.read_text())
        except json.JSONDecodeError:
            _cache = {}
    else:
        _cache = {}
    return _cache


def _persist() -> None:
    if _cache is None:
        return
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(_cache, indent=2))


def lookup(doi: str) -> dict | None:
    """Return metadata for a DOI: `{doi, title, authors, year, journal, source}`.

    Tries CrossRef first (covers nearly all journal articles, conference
    proceedings, protocols.io). Falls back to DataCite (covers datasets
    from Pennsieve, Zenodo, Dryad, Figshare, etc., plus institutional
    repositories and DataCite-side preprints) when CrossRef has no record.

    Cached on disk; first lookup ~200ms, subsequent ~0ms. Negative results
    (neither registry has the DOI) are also cached so we don't retry
    persistently-missing DOIs.
    """
    d = _normalize(doi)
    if not d:
        return None
    with _lock:
        cache = _load_cache()
        if d in cache:
            return cache[d]  # may be None (negative cache)
        md = crossref.fetch_doi_metadata(d)
        if not md:
            md = datacite.fetch_doi_metadata(d)
        cache[d] = md  # store None as a real None
        _persist()
    return md


def lookup_batch(dois: list[str]) -> dict[str, dict]:
    """Resolve many DOIs at once. Returns `{doi: metadata}` for any that
    resolve; missing/failed DOIs are silently dropped. Persists cache after
    each lookup."""
    out: dict[str, dict] = {}
    for doi in dois:
        md = lookup(doi)
        if md:
            out[_normalize(doi)] = md
    return out


def author_string(doi: str, fallback: str = "") -> str:
    """Render authors as 'Bayles et al., 2018' for inline citations.
    Returns `fallback` (default empty) if metadata isn't available."""
    md = lookup(doi)
    if not md:
        return fallback
    authors = (md.get("authors") or "").strip()
    year = (md.get("year") or "").strip()
    if not authors:
        return year or fallback
    # First author's last name (CrossRef gives "First Last, First Last, ...")
    parts = authors.split(",")
    first_full = parts[0].strip().split()
    last_name = first_full[-1] if first_full else ""
    if not last_name:
        return year or fallback
    suffix = " et al." if len(parts) > 1 else ""
    base = f"{last_name}{suffix}"
    return f"{base}, {year}" if year else base
