"""DANDI dandiset inspection — repository-specific implementation of the
generic inspector contract documented in `inspector_helpers`.

DANDI's storage model differs from Pennsieve's in two important ways for
inspection:

  1. **No deposit-level README file.** The dataset description lives in the
     dandi-schema metadata block returned by the API; we surface it as the
     `readme` field of the inspection dict so downstream agents see the
     same shape.
  2. **No CSV record tables.** DANDI deposits are typically NWB files;
     subject metadata is embedded in each NWB file's HDF5 header rather
     than in flat record tables. For a first version we leave the
     `records` field empty — extracting NWB-internal metadata at scale
     would require streaming the headers via `pynwb` or `h5py` and is a
     separate engineering task. The agent has access to the asset list and
     can probe individual files at runtime.

Returns the same dict shape as `pennsieve_inspect.inspect()`, so downstream
stages (Dataset Framer, Compatibility Analyst, Agentic Analyst) need no
repository-specific code.
"""
from __future__ import annotations

from scout.tools import dandi_discover
from scout.tools.inspector_helpers import (
    file_type_summary,
    render_file_tree,
)


def inspect(dandiset_id: str, version: str = "draft") -> dict:
    """One-shot inspection of a DANDI dandiset. Returns:

        {
          "readme":       <description from dandi-schema metadata>,
          "file_types":   <histogram from asset encodingFormat / extension>,
          "file_tree":    <rendered tree from asset paths>,
          "records":      {},            # NWB-internal subject metadata not extracted
          "bucket_uri":   <DANDI API base for this dandiset>,
          "bucket":       'dandi-archive', # logical bucket name
          "access":       'https',        # DANDI streams via HTTPS, not S3 directly
        }

    Failures degrade gracefully — partial results are still useful.
    """
    detail = dandi_discover.fetch_detail(dandiset_id, version=version)
    if detail is None:
        return {
            "readme": "",
            "file_types": {},
            "file_tree": f"(could not resolve dandiset {dandiset_id!r})",
            "records": {},
            "bucket_uri": "",
            "bucket": "dandi-archive",
            "access": "https",
        }

    resolved_version = detail.get("version") or version
    assets = dandi_discover.fetch_assets(
        dandiset_id, version=resolved_version,
    )

    return {
        "readme": detail.get("description") or "",
        "file_types": file_type_summary(assets),
        "file_tree": render_file_tree(assets),
        "records": {},
        "bucket_uri": detail.get("uri") or "",
        "bucket": "dandi-archive",
        "access": "https",
    }
