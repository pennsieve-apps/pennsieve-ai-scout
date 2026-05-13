"""NLM JATS XML chunker — parses Europe PMC full-text into section-tagged chunks.

Output: list of dicts with text, section, paragraph index, char range.
Chunking strategy: one chunk per paragraph (`<p>`), tagged with the parent
section title (from the nearest enclosing `<sec><title>...</title></sec>`).
Paragraphs shorter than MIN_CHARS are merged with the next paragraph in the
same section to avoid pathological tiny chunks.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

MIN_CHARS = 200  # paragraphs shorter than this get merged with the next
MAX_CHARS = 4000  # split paragraphs longer than this on sentence boundaries (rough)


def _strip_namespace(elem: ET.Element) -> None:
    """Remove namespace prefixes from tag names — JATS uses xmlns that confuses xpath."""
    for el in elem.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]


def _text_of(elem: ET.Element) -> str:
    """All text under elem, recursively, with whitespace normalized."""
    parts: list[str] = []
    for t in elem.itertext():
        if t:
            parts.append(t)
    text = " ".join(p.strip() for p in parts if p.strip())
    return " ".join(text.split())  # collapse runs of whitespace


def _section_title(elem: ET.Element, root: ET.Element) -> str:
    """Walk up from elem to find the nearest enclosing <sec> with a <title>."""
    parents: dict[ET.Element, ET.Element] = {c: p for p in root.iter() for c in p}
    current = elem
    while current is not None:
        if current.tag == "sec":
            t = current.find("title")
            if t is not None:
                return _text_of(t) or "section"
        current = parents.get(current)
    return ""


def chunks_from_jats(xml_text: str) -> list[dict]:
    """Parse JATS XML and return chunks. Each chunk: {text, section, order}."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    _strip_namespace(root)

    # Use abstract + body. Skip front-matter, references, tables-only.
    chunks: list[dict] = []

    # Abstract first
    abstract = root.find(".//abstract")
    if abstract is not None:
        text = _text_of(abstract)
        if text:
            chunks.append({"text": text, "section": "Abstract", "order": len(chunks)})

    # Body sections — paragraph by paragraph
    body = root.find(".//body")
    if body is not None:
        for p in body.iter("p"):
            text = _text_of(p)
            if not text:
                continue
            section = _section_title(p, root)
            chunks.append({"text": text, "section": section or "body", "order": len(chunks)})

        # Tables: include caption + footer text as their own chunks (often the most informative numbers)
        for tw in body.iter("table-wrap"):
            cap = tw.find(".//caption")
            cap_text = _text_of(cap) if cap is not None else ""
            # Also include the table cells flattened to text — that's where the numbers live
            tab = tw.find(".//table")
            tab_text = _text_of(tab) if tab is not None else ""
            combined = " — ".join(t for t in (cap_text, tab_text) if t).strip()
            if combined:
                section = _section_title(tw, root) or "Table"
                chunks.append({
                    "text": f"[Table] {combined}",
                    "section": f"{section} (table)",
                    "order": len(chunks),
                })

        # Figures: capture caption + label as a chunk, since the caption usually
        # describes what the figure shows (sample sizes, time courses, biomarker
        # trajectories, etc. — exactly the content downstream agents want to retrieve).
        for fig in body.iter("fig"):
            label = fig.find("label")
            cap = fig.find(".//caption")
            label_text = _text_of(label) if label is not None else ""
            cap_text = _text_of(cap) if cap is not None else ""
            combined = " — ".join(t for t in (label_text, cap_text) if t).strip()
            if combined:
                section = _section_title(fig, root) or "Figure"
                chunks.append({
                    "text": f"[Figure] {combined}",
                    "section": f"{section} (figure caption)",
                    "order": len(chunks),
                })

    # Merge tiny adjacent chunks; split very large ones
    merged: list[dict] = []
    for c in chunks:
        if merged and len(merged[-1]["text"]) < MIN_CHARS and merged[-1]["section"] == c["section"]:
            merged[-1]["text"] = merged[-1]["text"] + " " + c["text"]
        else:
            merged.append(dict(c))

    final: list[dict] = []
    for c in merged:
        if len(c["text"]) <= MAX_CHARS:
            final.append({"text": c["text"], "section": c["section"], "order": len(final)})
            continue
        # crude sentence-ish split
        text = c["text"]
        cur = ""
        for piece in text.replace("? ", "?|").replace("! ", "!|").replace(". ", ".|").split("|"):
            if len(cur) + len(piece) > MAX_CHARS and cur:
                final.append({"text": cur.strip(), "section": c["section"], "order": len(final)})
                cur = piece
            else:
                cur += " " + piece
        if cur.strip():
            final.append({"text": cur.strip(), "section": c["section"], "order": len(final)})

    return final
