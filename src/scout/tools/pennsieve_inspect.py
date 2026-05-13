"""Pennsieve dataset metadata inspection.

Reads README, manifest, and structured metadata records (CSV/TSV files under
`metadata/records/`) for a published Pennsieve Discover dataset.

Two access paths, auto-detected from the bucket name:

1. **OAD buckets** (name contains `-aod-`) — fully open, plain HTTPS, no auth.
2. **Legacy buckets** — Requester Pays, falls back to authenticated boto3 with
   `RequestPayer=requester`. Uses the default AWS credential chain (env vars,
   `~/.aws/credentials`, IAM role).

The dataset URI (e.g. `s3://edots-prod-aod-discover-publish50-use1/530/`) is
read from the `/discover/datasets/{id}` API response (`fetch_detail` in
`pennsieve_discover.py`). All keys returned by `list_prefix` and accepted by
`get_object` are **dataset-relative** (e.g. `manifest.json`, `metadata/schema.json`).
"""
from __future__ import annotations

import csv
import io
import json
import re
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

import httpx

S3_LIST_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def parse_uri(uri: str) -> tuple[str, str]:
    """s3://bucket/prefix/ → (bucket, prefix). Prefix has no leading or trailing slash."""
    p = urlparse(uri.strip())
    if p.scheme != "s3":
        raise ValueError(f"not an s3 uri: {uri!r}")
    return p.netloc, p.path.strip("/")


def is_oad_bucket(bucket: str) -> bool:
    return "-aod-" in bucket


def _get_oad(bucket: str, key: str, timeout: float = 30.0) -> bytes | None:
    url = f"https://{bucket}.s3.amazonaws.com/{key}"
    try:
        r = httpx.get(url, timeout=timeout, follow_redirects=True)
    except httpx.HTTPError:
        return None
    return r.content if r.status_code == 200 else None


def _get_requester_pays(bucket: str, key: str) -> bytes | None:
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        return None
    try:
        s3 = boto3.client("s3")
        resp = s3.get_object(Bucket=bucket, Key=key, RequestPayer="requester")
        return resp["Body"].read()
    except ClientError:
        return None
    except Exception:
        return None


def get_object(uri: str, sub_key: str) -> bytes | None:
    """Fetch one object at <uri>/<sub_key>. Auto-detects access path."""
    bucket, prefix = parse_uri(uri)
    full = f"{prefix}/{sub_key}".lstrip("/")
    if is_oad_bucket(bucket):
        return _get_oad(bucket, full)
    return _get_requester_pays(bucket, full)


def list_prefix(uri: str, sub_prefix: str) -> list[str]:
    """List keys under <uri>/<sub_prefix>/. Returns dataset-relative keys."""
    bucket, prefix = parse_uri(uri)
    full_prefix = f"{prefix}/{sub_prefix}".rstrip("/") + "/"
    if is_oad_bucket(bucket):
        url = f"https://{bucket}.s3.amazonaws.com/?list-type=2&prefix={full_prefix}"
        try:
            r = httpx.get(url, timeout=30.0)
        except httpx.HTTPError:
            return []
        if r.status_code != 200:
            return []
        try:
            root = ET.fromstring(r.text)
        except ET.ParseError:
            return []
        keys: list[str] = []
        strip = len(prefix) + 1
        for c in root.iter(f"{S3_LIST_NS}Contents"):
            k = c.find(f"{S3_LIST_NS}Key")
            if k is not None and k.text and k.text.startswith(prefix + "/"):
                keys.append(k.text[strip:])
        return keys
    # Requester-pays path
    try:
        import boto3
    except ImportError:
        return []
    try:
        s3 = boto3.client("s3")
        keys: list[str] = []
        strip = len(prefix) + 1
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=full_prefix, RequestPayer="requester"):
            for obj in page.get("Contents", []) or []:
                k = obj["Key"]
                if k.startswith(prefix + "/"):
                    keys.append(k[strip:])
        return keys
    except Exception:
        return []


def fetch_readme(uri: str) -> str | None:
    body = get_object(uri, "readme.md")
    return body.decode("utf-8", errors="replace") if body else None


def fetch_manifest(uri: str) -> dict | None:
    body = get_object(uri, "manifest.json")
    if not body:
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


def fetch_metadata_records(uri: str, max_rows_per_csv: int = 50,
                           max_files: int = 10) -> dict[str, dict]:
    """Read metadata/records/*.{csv,tsv} as previews.

    Returns {filename: {columns: [...], n_rows: int, head: [{col: val, ...}]}}.
    Parquet files are noted but not parsed (would need pyarrow)."""
    keys = list_prefix(uri, "metadata/records")
    out: dict[str, dict] = {}
    for k in keys:
        filename = k.rsplit("/", 1)[-1]
        if filename.lower().endswith(".parquet"):
            out[filename] = {"columns": [], "n_rows": 0, "head": [], "note": "parquet (not parsed)"}
            continue
        if not filename.lower().endswith((".csv", ".tsv")):
            continue
        body = get_object(uri, k)
        if not body:
            continue
        text = body.decode("utf-8", errors="replace")
        sep = "," if filename.lower().endswith(".csv") else "\t"
        reader = csv.reader(io.StringIO(text), delimiter=sep)
        try:
            header = next(reader)
        except StopIteration:
            continue
        head: list[dict] = []
        n = 0
        for row in reader:
            n += 1
            if len(head) < max_rows_per_csv and len(row) == len(header):
                head.append(dict(zip(header, row)))
        out[filename] = {"columns": header, "n_rows": n, "head": head}
        if len(out) >= max_files:
            break
    return out


def file_type_summary(manifest: dict | None, top_n: int = 8) -> dict[str, int]:
    """Pennsieve manifest → repo-agnostic file_type histogram via inspector_helpers."""
    from scout.tools.inspector_helpers import file_type_summary as _generic
    return _generic((manifest or {}).get("files") or [], top_n=top_n)


def _file_type_summary_legacy(manifest: dict | None, top_n: int = 8) -> dict[str, int]:
    if not manifest:
        return {}
    counts: dict[str, int] = {}
    for f in manifest.get("files", []) or []:
        ft = f.get("fileType") or f.get("packageType") or "Unknown"
        counts[ft] = counts.get(ft, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1])[:top_n])


def _human_size(n: int) -> str:
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


# Directory-name patterns that mark per-subject organization. Matches whole
# directory names like `sub-1001`, `subject_42`, `participant-01`, `S03`, `P02`.
_SUBJECT_RE = re.compile(
    r"^(sub[-_]|subject[-_]|participant[-_]|s|p)\w+$",
    re.IGNORECASE,
)


def _is_subject_dir(name: str) -> bool:
    return bool(_SUBJECT_RE.match(name))


def _summarize_subject_ids(names: list[str], max_examples: int = 6) -> str:
    """Compact id list: 'sub-1001, sub-1002, ..., sub-2006 (21 total)'."""
    sorted_names = sorted(set(names))
    if len(sorted_names) <= max_examples:
        return ", ".join(sorted_names)
    head = ", ".join(sorted_names[:3])
    tail = ", ".join(sorted_names[-2:])
    return f"{head}, ..., {tail} ({len(sorted_names)} total)"


def _build_dir_tree(files: list[dict]) -> dict:
    """Nested dict from manifest paths. Leaves carry size + type; internal
    nodes are themselves dicts."""
    tree: dict = {}
    for f in files:
        path = (f.get("path") or "").strip()
        if not path:
            continue
        parts = path.split("/")
        node = tree
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = {
            "_file": True,
            "size": int(f.get("size") or 0),
            "type": f.get("fileType") or f.get("packageType") or "?",
        }
    return tree


def _is_file_node(node) -> bool:
    return isinstance(node, dict) and node.get("_file") is True


def render_file_tree(manifest: dict | None,
                     max_paths_per_dir: int = 30,
                     max_total_lines: int = 300) -> str:
    """Pennsieve manifest → rendered file tree via inspector_helpers.

    Thin wrapper that extracts the `files` array and delegates to the
    repository-agnostic renderer."""
    from scout.tools.inspector_helpers import render_file_tree as _generic
    if not manifest:
        return "(no manifest available)"
    return _generic(manifest.get("files") or [],
                    max_paths_per_dir=max_paths_per_dir,
                    max_total_lines=max_total_lines)


def _render_file_tree_legacy(manifest: dict | None,
                              max_paths_per_dir: int = 30,
                              max_total_lines: int = 300) -> str:
    """(legacy, kept for reference) Compact tree summary; superseded by
    inspector_helpers.render_file_tree."""
    from collections import Counter
    if not manifest:
        return "(no manifest available)"
    files = manifest.get("files") or []
    if not files:
        return "(no files in manifest)"

    tree = _build_dir_tree(files)
    lines: list[str] = []

    def _ext_summary(names: list[str], top_n: int = 5) -> str:
        c: Counter[str] = Counter()
        for n in names:
            ext = n.rsplit(".", 1)[-1].lower() if "." in n else "(none)"
            c[ext] += 1
        return ", ".join(f"{e}:{k}" for e, k in c.most_common(top_n))

    def render(node: dict, dir_path: str) -> None:
        """Render one directory node and recurse. dir_path is the slash-joined
        path so far (e.g. 'files/primary/sub-1001'). Uses '(root)' when empty."""
        if len(lines) >= max_total_lines:
            return
        # Partition immediate children
        children = sorted(node.items())
        files_here = [(n, s) for n, s in children if _is_file_node(s)]
        subdirs = [(n, s) for n, s in children if not _is_file_node(s)]
        subject_subs = [(n, s) for n, s in subdirs if _is_subject_dir(n)]
        non_subject_subs = [(n, s) for n, s in subdirs if not _is_subject_dir(n)]

        # Header: this directory's own files + extension histogram
        display_path = dir_path or "(root)"
        if files_here:
            ext_str = _ext_summary([n for n, _ in files_here])
            lines.append(f"{display_path}/  ({len(files_here)} files; {ext_str})")
        elif not subdirs:
            lines.append(f"{display_path}/  (empty)")
            return
        else:
            lines.append(f"{display_path}/")

        # List file children
        for name, leaf in files_here[:max_paths_per_dir]:
            lines.append(f"  {dir_path + '/' if dir_path else ''}{name} ({_human_size(leaf['size'])})")
        if len(files_here) > max_paths_per_dir:
            lines.append(f"  ... +{len(files_here) - max_paths_per_dir} more files at this level")

        # Subject-collapse: ≥5 subject-id-pattern subdirs → cohort summary +
        # render only the first as representative
        if len(subject_subs) >= 5:
            ids = [n for n, _ in subject_subs]
            lines.append(
                f"  └─ {len(ids)} subjects: {_summarize_subject_ids(ids)}"
            )
            ex_name, ex_sub = subject_subs[0]
            lines.append(
                f"  └─ representative subject ({ex_name}/) — other subjects "
                f"have a similar layout:"
            )
            render(ex_sub, f"{dir_path + '/' if dir_path else ''}{ex_name}")
        else:
            # Few subject dirs (or none) — render each normally
            for name, sub in subject_subs:
                if len(lines) >= max_total_lines:
                    break
                render(sub, f"{dir_path + '/' if dir_path else ''}{name}")

        # Always render non-subject subdirs normally
        for name, sub in non_subject_subs:
            if len(lines) >= max_total_lines:
                break
            render(sub, f"{dir_path + '/' if dir_path else ''}{name}")

    render(tree, "")
    if len(lines) >= max_total_lines:
        lines.append("... (truncated; use S3 list at runtime to discover unseen paths)")
    return "\n".join(lines)


def inspect(uri: str) -> dict:
    """One-shot inspection: README + file-type histogram + manifest tree +
    metadata-record previews + bucket URI/access info.

    Failures degrade gracefully — partial results are still useful."""
    readme = fetch_readme(uri)
    manifest = fetch_manifest(uri)
    bucket, _ = parse_uri(uri)
    return {
        "readme": readme or "",
        "file_types": file_type_summary(manifest),
        "file_tree": render_file_tree(manifest),
        "records": fetch_metadata_records(uri),
        "bucket_uri": uri,
        "bucket": bucket,
        "access": "oad" if is_oad_bucket(bucket) else "requester_pays",
    }
