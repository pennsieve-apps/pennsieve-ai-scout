"""OpenNeuro catalog client — third repository adapter alongside Pennsieve
and DANDI.

OpenNeuro hosts ~7,700 BIDS-formatted human neuroimaging datasets (fMRI,
MRI, EEG, MEG, iEEG/ECoG/SEEG, PET, behavioral). Public GraphQL API at
`https://openneuro.org/crn/graphql` — no auth required for reads. Per-file
download URLs returned in-line; anonymous S3 access also works at
`https://s3.amazonaws.com/openneuro.org/<dsid>/<path>`.

Mirrors the surface of `pennsieve_discover` / `dandi_discover`:
  - `fetch_detail(ds_id)` → uniform record (title, description, tags,
    license, size_bytes, file_count, n_subjects, species, measurements,
    external_publications, doi, url, uri).
  - `fetch_assets(ds_id)` → list of `{path, size, fileType, url}` records
    walked across the BIDS tree.
  - `search(query, limit)` → catalog text search → normalized hits.

The dataset URI is the GraphQL endpoint base (downstream stages don't
browse OpenNeuro via a single S3 prefix; they hit per-file object URLs).
"""
from __future__ import annotations

import httpx

GRAPHQL = "https://openneuro.org/crn/graphql"
S3_BASE = "https://s3.amazonaws.com/openneuro.org"


def _post(query: str, variables: dict | None = None) -> dict | None:
    try:
        r = httpx.post(GRAPHQL, json={"query": query, "variables": variables or {}},
                       timeout=30)
        if r.status_code != 200:
            return None
        data = r.json()
        if data.get("errors"):
            return None
        return data.get("data")
    except httpx.HTTPError:
        return None


def fetch_detail(ds_id: str) -> dict | None:
    """Top-level dataset metadata. `ds_id` may be `ds003029` or `openneuro:ds003029`.

    Returns a dict shaped like Pennsieve/DANDI detail records:
        {
          "id": "openneuro:ds003029",
          "title": ...,
          "description": ...,  # README, since OpenNeuro lacks a separate description
          "tags": [...],       # modalities + tasks fold into here
          "license": ...,
          "size_bytes": int,
          "file_count": int,   # 0 if not exposed in metadata; resolved via fetch_assets
          "n_subjects": int,
          "species": str,
          "measurements": [...],  # modalities + tasks for cross-repo similarity
          "externalPublications": [{doi, relationshipType}, ...],
          "doi": str,
          "url": "https://openneuro.org/datasets/<dsid>/versions/<tag>",
          "uri": GRAPHQL,
          "version": <snapshot tag>,
          "_raw_metadata": {...},
        }
    """
    norm = ds_id.split(":", 1)[-1] if ds_id else ""
    if not norm:
        return None

    query = """
    query($id: ID!) {
      dataset(id: $id) {
        id
        name
        created
        publishDate
        metadata {
          datasetUrl
          associatedPaperDOI
          species
          modalities
          studyDesign
        }
        latestSnapshot {
          tag
          readme
          size
          description {
            Name
            BIDSVersion
            DatasetType
            Authors
            License
          }
          summary {
            subjects
            modalities
            secondaryModalities
            tasks
            sessions
          }
        }
      }
    }
    """
    data = _post(query, {"id": norm})
    if not data or not data.get("dataset"):
        return None
    d = data["dataset"]
    snap = d.get("latestSnapshot") or {}
    meta = d.get("metadata") or {}
    desc = snap.get("description") or {}
    summ = snap.get("summary") or {}

    tag = snap.get("tag") or ""
    subjects = summ.get("subjects") or []
    modalities = summ.get("modalities") or []
    tasks = summ.get("tasks") or []
    secondary = summ.get("secondaryModalities") or []
    measurements = sorted(set(filter(None, modalities + tasks + secondary)))

    # Build tags = modalities ∪ tasks ∪ secondaryModalities; deduped
    tags = sorted(set(filter(None, modalities + tasks + secondary +
                              (meta.get("modalities") or []) +
                              ([meta.get("studyDesign")] if meta.get("studyDesign") else []))))

    # Authors string for downstream rendering
    authors = desc.get("Authors") or []
    if not isinstance(authors, list):
        authors = [authors] if authors else []

    # External publications from associatedPaperDOI (comma- or whitespace-separated)
    raw_dois = (meta.get("associatedPaperDOI") or "").strip()
    pubs = []
    if raw_dois:
        for token in raw_dois.replace(",", " ").replace(";", " ").split():
            token = token.strip()
            if not token or "10." not in token:
                continue
            # Normalize: drop URL prefix
            doi = token.replace("https://doi.org/", "").replace("http://doi.org/", "")
            doi = doi.replace("doi:", "").strip()
            if doi:
                pubs.append({"doi": doi, "relationshipType": "IsDescribedBy"})

    species = (meta.get("species") or "").strip()

    return {
        "id": f"openneuro:{norm}",
        "title": (desc.get("Name") or d.get("name") or "").strip(),
        "description": (snap.get("readme") or "").strip(),
        "tags": tags,
        "license": (desc.get("License") or "").strip(),
        "size_bytes": int(snap.get("size") or 0),
        "file_count": 0,  # OpenNeuro doesn't expose total file count in summary; resolved via fetch_assets
        "n_subjects": len(subjects),
        "species": species,
        "measurements": measurements,
        "externalPublications": pubs,
        # OpenNeuro snapshots are DataCite-registered with predictable DOIs of
        # the form `10.18112/openneuro.<id>.v<tag>`. Construct it deterministically
        # so the downstream DOI-enrichment + paper-writer bibliography pipeline
        # can resolve and cite the dataset. (DataCite may return placeholder
        # metadata for some snapshots; the bibliography renderer falls back to
        # `metadata unavailable` and never invents authors.)
        "doi": f"10.18112/openneuro.{norm}.v{tag}" if tag else "",
        "url": f"https://openneuro.org/datasets/{norm}/versions/{tag}" if tag else f"https://openneuro.org/datasets/{norm}",
        "uri": GRAPHQL,
        "version": tag,
        "bids_version": (desc.get("BIDSVersion") or "").strip(),
        "dataset_type": (desc.get("DatasetType") or "").strip(),
        "authors": authors,
        "_raw_subjects": subjects,
        "_raw_modalities": modalities,
        "_raw_tasks": tasks,
    }


def fetch_assets(ds_id: str, limit: int | None = None,
                 max_depth: int = 6) -> list[dict]:
    """Walk the BIDS tree via the GraphQL `files(tree: ...)` field.

    OpenNeuro exposes a per-directory listing; we recurse to gather a flat
    list of file records `{path, size, fileType, url}`. Stops at `limit`
    files or `max_depth` directory levels (BIDS tops out around 5 — sub/ses/
    modality/task/file).

    Each record has:
      - `path`: BIDS-relative POSIX path (e.g. 'sub-pt01/ses-01/ieeg/...')
      - `size`: bytes
      - `fileType`: short label from extension (EDF, NWB, TSV, JSON, etc.)
      - `url`: per-file object URL (works without auth)
      - `s3_url`: alternative anonymous S3 URL (useful for byte-range reads)
    """
    norm = ds_id.split(":", 1)[-1] if ds_id else ""
    if not norm:
        return []

    # OpenNeuro's `files(tree: ...)` takes a *directory id* (the `id` field on
    # a directory entry returned by the parent listing), NOT a path string.
    # Root listing is `files` with no tree arg.
    query_root = """
    query($id: ID!) {
      dataset(id: $id) {
        latestSnapshot {
          files {
            id
            filename
            size
            directory
            urls
          }
        }
      }
    }
    """
    query_subtree = """
    query($id: ID!, $tree: String!) {
      dataset(id: $id) {
        latestSnapshot {
          files(tree: $tree) {
            id
            filename
            size
            directory
            urls
          }
        }
      }
    }
    """

    out: list[dict] = []

    def _walk(parent_path: str, dir_id: str | None, depth: int) -> None:
        if limit is not None and len(out) >= limit:
            return
        if depth > max_depth:
            return
        if dir_id is None:
            data = _post(query_root, {"id": norm})
        else:
            data = _post(query_subtree, {"id": norm, "tree": dir_id})
        if not data or not data.get("dataset"):
            return
        files = (data["dataset"].get("latestSnapshot") or {}).get("files") or []
        for f in files:
            name = f.get("filename") or ""
            if not name or name.startswith("."):
                continue
            full = f"{parent_path}/{name}" if parent_path else name
            is_dir = bool(f.get("directory"))
            if is_dir:
                _walk(full, f.get("id"), depth + 1)
                if limit is not None and len(out) >= limit:
                    return
            else:
                size = int(f.get("size") or 0)
                ext = name.rsplit(".", 1)[-1].upper() if "." in name else "?"
                urls = f.get("urls") or []
                out.append({
                    "path": full,
                    "size": size,
                    "fileType": ext,
                    "url": urls[0] if urls else "",
                    "s3_url": f"{S3_BASE}/{norm}/{full}",
                    "asset_id": f.get("id") or "",
                })

    _walk("", None, 0)
    return out


def search(query_text: str, limit: int = 25,
           modality: str | None = None) -> list[dict]:
    """Modality-scoped catalog listing — the closest thing to a search the
    OpenNeuro GraphQL exposes without Elasticsearch DSL. `query_text` is
    used as a client-side substring filter on dataset name + summary tasks;
    `modality` (e.g. 'ieeg', 'eeg', 'mri', 'meg', 'pet') is passed to the
    server-side `datasets(modality: ...)` filter. Pass `modality=None` to
    list across all modalities and rely entirely on the text filter.

    Returns normalized records shaped like `pennsieve_discover.search()`'s
    output so the cross-repo similar-dataset finder can union them uniformly.
    """
    norm_query = (query_text or "").strip().lower()
    page_size = max(limit * 3, 50)  # fetch more, then text-filter down to `limit`

    gql = """
    query($first: Int!, $modality: String) {
      datasets(first: $first, modality: $modality) {
        edges {
          node {
            id
            name
            latestSnapshot {
              tag
              size
              description { Name }
              summary { subjects modalities tasks }
            }
            metadata { species associatedPaperDOI studyDesign }
          }
        }
      }
    }
    """
    data = _post(gql, {"first": page_size, "modality": modality})
    if not data or not data.get("datasets"):
        return []

    def _matches(node: dict) -> bool:
        if not norm_query:
            return True
        name = (node.get("name") or "").lower()
        snap = node.get("latestSnapshot") or {}
        tasks = " ".join((snap.get("summary") or {}).get("tasks") or []).lower()
        return norm_query in name or norm_query in tasks

    out: list[dict] = []
    for edge in (data["datasets"].get("edges") or []):
        n = edge.get("node") or {}
        if not _matches(n):
            continue
        snap = n.get("latestSnapshot") or {}
        summ = snap.get("summary") or {}
        meta = n.get("metadata") or {}
        subs = summ.get("subjects") or []
        modalities = summ.get("modalities") or []
        tasks = summ.get("tasks") or []
        ds_id = n.get("id") or ""
        out.append({
            "id": f"openneuro:{ds_id}",
            "title": n.get("name") or (snap.get("description") or {}).get("Name") or "",
            "description": "",
            "measurements": sorted(set(filter(None, modalities + tasks))),
            "species": (meta.get("species") or ""),
            "n_subjects": len(subs),
            "repo": "openneuro",
            "url": f"https://openneuro.org/datasets/{ds_id}",
            "doi": "",
            "license": "",
            "size_bytes": int(snap.get("size") or 0),
            "file_count": 0,
        })
        if len(out) >= limit:
            break
    return out
