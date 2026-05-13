"""CrossRef DOI metadata — universal fallback for any DOI.

Used when Europe PMC doesn't have a record (which is the case for protocols.io
entries, datasets, preprints from non-PMC sources, etc.). CrossRef indexes
basically every DOI registered with it, including protocol papers.

Free, no auth required. Polite rate limit ~50 calls/sec.
"""
from __future__ import annotations

import re

import httpx

BASE = "https://api.crossref.org"


def _strip_doi(doi: str) -> str:
    s = (doi or "").strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if s.lower().startswith(prefix):
            s = s[len(prefix):]
    return s


def _strip_jats(text: str) -> str:
    """Remove JATS XML tags that often appear in CrossRef abstracts.
    Returns plain prose suitable for prompt input."""
    if not text:
        return ""
    return re.sub(r"<[^>]+>", " ", text).strip()


def fetch_doi_metadata(doi: str) -> dict | None:
    """Return a paper-record-shaped dict for any DOI registered with CrossRef.

    Handles protocols.io, datasets, preprints, conference papers — anything
    with a DOI. Returns None if CrossRef has no record (rare). The returned
    shape matches `europe_pmc.search()` output so callers can treat both
    sources uniformly:
        {doi, title, abstract, year, authors, journal, source}
    """
    norm = _strip_doi(doi)
    if not norm:
        return None
    try:
        r = httpx.get(f"{BASE}/works/{norm}", timeout=20,
                      headers={"User-Agent": "pennsieve-ai-scout (mailto:joost.wagenaar2@pennmedicine.upenn.edu)"})
        if r.status_code != 200:
            return None
        msg = r.json().get("message") or {}
    except httpx.HTTPError:
        return None

    title = " ".join(msg.get("title") or []).strip()
    abstract = _strip_jats(msg.get("abstract", "") or "")
    authors_raw = msg.get("author") or []
    authors = ", ".join(
        f"{a.get('given','').strip()} {a.get('family','').strip()}".strip()
        for a in authors_raw[:8]
        if a.get("given") or a.get("family")
    )
    # Year: prefer published-print, then published-online, then created
    for key in ("published-print", "published-online", "issued", "created"):
        date_obj = msg.get(key) or {}
        parts = (date_obj.get("date-parts") or [[None]])[0]
        if parts and parts[0]:
            year = str(parts[0])
            break
    else:
        year = ""

    venue = " ".join(msg.get("container-title") or []).strip()
    if not venue:
        # protocols.io and similar don't have a journal; fall back to the type
        venue = msg.get("type", "") or msg.get("publisher", "")

    return {
        "doi": norm,
        "title": title,
        "abstract": abstract,
        "year": year,
        "authors": authors,
        "journal": venue,
        "source": "crossref",
    }
