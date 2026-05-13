"""Markdown chunker — produces the same chunk shape as jats_chunker.

Each chunk is `{text, section, order}`. Sections are inferred from the nearest
preceding `##`/`###` heading. Pathological short paragraphs are merged with
their successor; very long paragraphs are split on sentence boundaries.

Used to ingest Marker output (PDF → Markdown) into the same RAG pipeline that
JATS XML uses.
"""
from __future__ import annotations

import re

MIN_CHARS = 200
MAX_CHARS = 4000

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def _segment(md_text: str) -> list[dict]:
    """Walk the markdown linearly, attaching the most recent heading as section."""
    out: list[dict] = []
    current_section = "body"
    current_buf: list[str] = []

    def flush() -> None:
        text = " ".join(current_buf).strip()
        text = re.sub(r"\s+", " ", text)
        if text:
            out.append({"text": text, "section": current_section, "order": len(out)})
        current_buf.clear()

    for raw_line in md_text.splitlines():
        line = raw_line.rstrip()
        m = _HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            # Top-level title becomes our section name; subheadings inherit it
            if level <= 2:
                current_section = title or current_section
            else:
                current_section = title or current_section
            continue
        if not line.strip():
            flush()
            continue
        current_buf.append(line.strip())
    flush()
    return out


def chunks_from_markdown(md_text: str) -> list[dict]:
    """Convert Marker-style markdown into section-tagged chunks."""
    if not md_text:
        return []
    raw = _segment(md_text)

    # Merge tiny adjacent paragraphs in the same section
    merged: list[dict] = []
    for c in raw:
        if merged and len(merged[-1]["text"]) < MIN_CHARS and merged[-1]["section"] == c["section"]:
            merged[-1]["text"] = merged[-1]["text"] + " " + c["text"]
        else:
            merged.append(dict(c))

    # Split oversize chunks on sentence-ish boundaries
    final: list[dict] = []
    for c in merged:
        if len(c["text"]) <= MAX_CHARS:
            final.append({"text": c["text"], "section": c["section"], "order": len(final)})
            continue
        text = c["text"]
        cur = ""
        # rough sentence split
        for piece in text.replace("? ", "?|").replace("! ", "!|").replace(". ", ".|").split("|"):
            if len(cur) + len(piece) > MAX_CHARS and cur:
                final.append({"text": cur.strip(), "section": c["section"], "order": len(final)})
                cur = piece
            else:
                cur = (cur + " " + piece) if cur else piece
        if cur.strip():
            final.append({"text": cur.strip(), "section": c["section"], "order": len(final)})

    return final
