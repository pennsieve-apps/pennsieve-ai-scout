"""Repository-agnostic helpers for dataset inspection.

The contract any repository inspector should produce (the dict returned by
its `inspect(<repo>:<id>)` function):

    {
      "readme":       <str>           # full deposit description, prose
      "file_types":   <dict[str,int]> # extension/type histogram
      "file_tree":    <str>           # rendered tree from list of files
      "records":      <dict[str,dict]># filename -> {columns, n_rows, head}
      "bucket_uri":   <str>           # primary access URL/URI for downloads
      "access":       <str>           # 'oad' / 'requester_pays' / 'https' / etc.
    }

Repository-specific inspectors (e.g. `pennsieve_inspect`, `dandi_inspect`)
implement the catalog/storage particulars; the helpers here cover what is
shared across deposits — file-tree rendering with BIDS-style cohort
collapsing, file-type histogram, human-readable size formatting.
"""
from __future__ import annotations

import re
from collections import Counter

# BIDS-style and similar subject-organization patterns. Matches whole
# directory names like `sub-1001`, `subject_42`, `participant-01`, `S03`,
# `P02`. Common across SPARC Data Standard, BIDS (the de-facto neuroimaging
# / electrophysiology standard adopted by DANDI), and many other deposits.
_SUBJECT_RE = re.compile(
    r"^(sub[-_]|subject[-_]|participant[-_]|s|p)\w+$",
    re.IGNORECASE,
)


def is_subject_dir(name: str) -> bool:
    """True if `name` matches the BIDS-style subject-directory naming."""
    return bool(_SUBJECT_RE.match(name))


def summarize_subject_ids(names: list[str], max_examples: int = 6) -> str:
    """Compact id list: 'sub-1001, sub-1002, ..., sub-2006 (21 total)'."""
    sorted_names = sorted(set(names))
    if len(sorted_names) <= max_examples:
        return ", ".join(sorted_names)
    head = ", ".join(sorted_names[:3])
    tail = ", ".join(sorted_names[-2:])
    return f"{head}, ..., {tail} ({len(sorted_names)} total)"


def human_size(n: int) -> str:
    """Format a byte count as KB / MB / GB / TB."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def build_dir_tree(files: list[dict]) -> dict:
    """Nested dict from a list of `{path, size, fileType?, packageType?}`
    entries. Leaves carry size + type; internal nodes are dicts."""
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


def is_file_node(node) -> bool:
    return isinstance(node, dict) and node.get("_file") is True


def file_type_summary(files: list[dict] | None, top_n: int = 8) -> dict[str, int]:
    """Histogram of file types from a list of `{path, fileType?, packageType?}`
    entries. Used to give the LLM a quick read on what's in the deposit."""
    if not files:
        return {}
    c: Counter[str] = Counter()
    for f in files:
        ft = f.get("fileType") or f.get("packageType") or "?"
        c[ft] += 1
    return dict(c.most_common(top_n))


def render_file_tree(files: list[dict] | None,
                     max_paths_per_dir: int = 30,
                     max_total_lines: int = 300) -> str:
    """Compact tree summary of a deposit's files, suitable for an LLM prompt.

    Walks the directory tree built from `files`. At each level, when ≥5
    subdirectories match a subject-id pattern (`sub-*`, `subject_*`, etc.),
    those directories are collapsed into a one-line cohort summary plus a
    full recursive render of one representative subject's structure. This
    lets the downstream Analyst see the actual per-subject file layout
    instead of guessing from a truncated flat listing.

    For non-subject directories, files are listed up to `max_paths_per_dir`
    with a "+N more" tail. Total output is capped at `max_total_lines`.

    `files` is a list of dicts with at least `{path, size}` entries.
    Repository-specific inspectors normalize their native manifests/asset-
    lists into this shape before calling here."""
    if not files:
        return "(no files)"

    tree = build_dir_tree(files)
    lines: list[str] = []

    def _ext_summary(names: list[str], top_n: int = 5) -> str:
        c: Counter[str] = Counter()
        for n in names:
            ext = n.rsplit(".", 1)[-1].lower() if "." in n else "(none)"
            c[ext] += 1
        return ", ".join(f"{e}:{k}" for e, k in c.most_common(top_n))

    def render(node: dict, dir_path: str) -> None:
        """Render one directory node and recurse. dir_path is the slash-joined
        path so far. Uses '(root)' when empty."""
        if len(lines) >= max_total_lines:
            return
        children = sorted(node.items())
        files_here = [(n, s) for n, s in children if is_file_node(s)]
        subdirs = [(n, s) for n, s in children if not is_file_node(s)]
        subject_subs = [(n, s) for n, s in subdirs if is_subject_dir(n)]
        non_subject_subs = [(n, s) for n, s in subdirs if not is_subject_dir(n)]

        display_path = dir_path or "(root)"
        if files_here:
            ext_str = _ext_summary([n for n, _ in files_here])
            lines.append(f"{display_path}/  ({len(files_here)} files; {ext_str})")
        elif not subdirs:
            lines.append(f"{display_path}/  (empty)")
            return
        else:
            lines.append(f"{display_path}/")

        for name, leaf in files_here[:max_paths_per_dir]:
            lines.append(
                f"  {dir_path + '/' if dir_path else ''}{name} "
                f"({human_size(leaf['size'])})"
            )
        if len(files_here) > max_paths_per_dir:
            lines.append(
                f"  ... +{len(files_here) - max_paths_per_dir} more files at this level"
            )

        if len(subject_subs) >= 5:
            ids = [n for n, _ in subject_subs]
            lines.append(f"  └─ {len(ids)} subjects: {summarize_subject_ids(ids)}")
            ex_name, ex_sub = subject_subs[0]
            lines.append(
                f"  └─ representative subject ({ex_name}/) — other subjects "
                f"have a similar layout:"
            )
            render(ex_sub, f"{dir_path + '/' if dir_path else ''}{ex_name}")
        else:
            for name, sub in subject_subs:
                if len(lines) >= max_total_lines:
                    break
                render(sub, f"{dir_path + '/' if dir_path else ''}{name}")

        for name, sub in non_subject_subs:
            if len(lines) >= max_total_lines:
                break
            render(sub, f"{dir_path + '/' if dir_path else ''}{name}")

    render(tree, "")
    if len(lines) >= max_total_lines:
        lines.append(
            "... (truncated; list at runtime to discover unseen paths)"
        )
    return "\n".join(lines)
