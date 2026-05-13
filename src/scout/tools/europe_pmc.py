"""Paper search and full-text — free, no auth required.

Search is via Europe PMC (good metadata, OA flags, abstracts).
Full-text XML is via NCBI EFetch — Europe PMC's own /fullTextXML endpoint
is unreliable (404s for most papers we tested, even older ones).
PDF download is via Europe PMC's render endpoint (works reliably).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import httpx

BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"
NCBI_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def search(query: str, page_size: int = 25) -> list[dict]:
    """Search Europe PMC. Returns paper records with abstracts (papers without abstracts are dropped).
    Each record includes `is_open_access` and `source` + `pmcid` so the Deep Reader knows
    which papers can be downloaded full-text."""
    params = {
        "query": query,
        "format": "json",
        "pageSize": page_size,
        "resultType": "core",
    }
    r = httpx.get(f"{BASE}/search", params=params, timeout=30)
    r.raise_for_status()
    results = r.json().get("resultList", {}).get("result", [])
    out = []
    for x in results:
        if not x.get("abstractText"):
            continue
        is_oa = (x.get("isOpenAccess") == "Y") or (x.get("inEPMC") == "Y" and x.get("hasPDF") == "Y")
        out.append({
            "title": x.get("title", ""),
            "abstract": x.get("abstractText", ""),
            "doi": x.get("doi", ""),
            "year": x.get("pubYear", ""),
            "journal": x.get("journalTitle", ""),
            "pmid": x.get("pmid", ""),
            "pmcid": x.get("pmcid", ""),
            "source": x.get("source", ""),  # MED | PMC | etc — needed for full-text URL
            "is_open_access": bool(is_oa),
            "authors": x.get("authorString", ""),
        })
    return out


def fetch_fulltext_xml(pmcid: str) -> str | None:
    """Fetch NLM JATS XML for a PMC paper via NCBI EFetch.

    NCBI EFetch is the canonical source for PMC full-text XML and works for
    both recent and old papers. Europe PMC's own /fullTextXML endpoint
    returns 404 for most papers we tested, so we don't use it.

    Returns the XML for the inner `<article>` element (with the
    `<pmc-articleset>` wrapper stripped) so our chunker sees a clean JATS
    article. Returns None on failure (network error, not in PMC, parse error).
    """
    if not pmcid:
        return None
    digits = pmcid.replace("PMC", "")  # NCBI EFetch wants the numeric ID only
    try:
        r = httpx.get(
            NCBI_EFETCH,
            params={"db": "pmc", "id": digits, "rettype": "full", "retmode": "xml"},
            timeout=60,
            follow_redirects=True,
        )
        if r.status_code != 200:
            return None
        body = r.text
        if not body.lstrip().startswith("<"):
            return None
    except Exception:
        return None

    # Unwrap <pmc-articleset> if present — extract the inner <article>
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return None
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    if root.tag in ("pmc-articleset", "PubmedArticleSet"):
        article = root.find("article") or root.find(".//article")
        if article is None:
            return None
        return ET.tostring(article, encoding="unicode")
    return body


def fetch_fulltext_pdf(pmcid: str) -> bytes | None:
    """Download the published PDF for an open-access PMC paper.
    Verified working URL pattern: https://europepmc.org/articles/{pmcid}?pdf=render
    Returns PDF bytes or None on failure (paywall, not in PMC, network error)."""
    if not pmcid:
        return None
    url = f"https://europepmc.org/articles/{pmcid}?pdf=render"
    try:
        r = httpx.get(url, timeout=30, follow_redirects=True)
        if r.status_code != 200:
            return None
        ct = r.headers.get("content-type", "")
        looks_like_pdf = ct.startswith("application/pdf") or r.content[:4] == b"%PDF"
        if not looks_like_pdf:
            return None
        return r.content
    except Exception:
        return None
