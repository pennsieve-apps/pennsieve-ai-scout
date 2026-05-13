"""OpenNeuro inspection adapter — uniform inspection-dict contract.

OpenNeuro datasets are BIDS-structured: README at top level, dataset_description.json
schema, participants.tsv with subject demographics, and modality-specific
folders per BIDS spec (anat/, func/, eeg/, ieeg/, meg/, etc.).

This adapter populates:
  - `readme`: depositor-authored README text (from `latestSnapshot.readme`)
  - `file_types`: histogram from asset extensions (EDF, NWB, NII, TSV, JSON, ...)
  - `file_tree`: rendered tree via shared `inspector_helpers.render_file_tree`
                 (BIDS-aware subject-collapsing already in place)
  - `records`: parsed participants.tsv + any *_scans.tsv aggregates as
                a dict {filename: {columns, n_rows, head}}, lightweight via
                httpx without auth
  - `bucket_uri`: anonymous S3 base for direct downloads
  - `bucket`: 'openneuro.org'
  - `access`: 'https' (anonymous S3, supports byte-range reads)

Returns the same shape as `pennsieve_inspect.inspect()` / `dandi_inspect.inspect()`.
"""
from __future__ import annotations

import csv
import io

import httpx

from scout.tools import openneuro_discover
from scout.tools.inspector_helpers import (
    file_type_summary,
    render_file_tree,
)


S3_BASE = "https://s3.amazonaws.com/openneuro.org"


def _try_fetch_text(url: str, max_bytes: int = 200_000) -> str | None:
    """GET a (small) text file. Returns None on any error or oversized response."""
    try:
        r = httpx.get(url, timeout=20, headers={"Range": f"bytes=0-{max_bytes-1}"})
        if r.status_code not in (200, 206):
            return None
        return r.text
    except httpx.HTTPError:
        return None


def _parse_tsv(text: str, filename: str, head_rows: int = 5) -> dict | None:
    """Parse a TSV string into a record-table dict {columns, n_rows, head}."""
    try:
        reader = csv.reader(io.StringIO(text), delimiter="\t")
        rows = list(reader)
    except Exception:
        return None
    if not rows:
        return None
    columns = [c.strip() for c in rows[0]]
    body = rows[1:]
    head = [dict(zip(columns, r)) for r in body[:head_rows]]
    return {
        "filename": filename,
        "columns": columns,
        "n_rows": len(body),
        "head": head,
    }


def inspect(ds_id: str) -> dict:
    """One-shot inspection of an OpenNeuro dataset.

    Returns:
        {
          "readme":      <depositor README>,
          "file_types":  <histogram from asset extensions>,
          "file_tree":   <rendered tree with BIDS sub-* collapsing>,
          "records":     {<filename>: {columns, n_rows, head}} for participants.tsv
                        and top-level *_scans.tsv if present,
          "bucket_uri":  's3://openneuro.org/<dsid>/' (logical),
          "bucket":      'openneuro.org',
          "access":      'https',
        }

    Failures degrade gracefully — partial results are still useful.
    """
    norm = ds_id.split(":", 1)[-1] if ds_id else ""
    if not norm:
        return {
            "readme": "",
            "file_types": {},
            "file_tree": f"(invalid openneuro id {ds_id!r})",
            "records": {},
            "bucket_uri": "",
            "bucket": "openneuro.org",
            "access": "https",
        }

    detail = openneuro_discover.fetch_detail(norm)
    if detail is None:
        return {
            "readme": "",
            "file_types": {},
            "file_tree": f"(could not resolve openneuro dataset {norm!r})",
            "records": {},
            "bucket_uri": "",
            "bucket": "openneuro.org",
            "access": "https",
        }

    # Asset listing: cap at 1500 files for inspection (BIDS dirs can balloon
    # past 10K files; the agent has access to fetch_assets directly if needed).
    assets = openneuro_discover.fetch_assets(norm, limit=1500)

    records: dict = {}
    # Pull participants.tsv directly from anonymous S3 (small, ~few KB)
    for top_tsv in ("participants.tsv", "samples.tsv"):
        url = f"{S3_BASE}/{norm}/{top_tsv}"
        text = _try_fetch_text(url)
        if text:
            parsed = _parse_tsv(text, top_tsv)
            if parsed:
                records[top_tsv] = parsed

    return {
        "readme": detail.get("description") or "",
        "file_types": file_type_summary(assets),
        "file_tree": render_file_tree(assets),
        "records": records,
        "bucket_uri": f"{S3_BASE}/{norm}/",
        "bucket": "openneuro.org",
        "access": "https",
    }
