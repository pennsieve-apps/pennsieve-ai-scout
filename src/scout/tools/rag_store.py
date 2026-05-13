"""Persistent RAG store — caches papers, chunks, and embeddings across runs.

Layout (under runs/_knowledge/):
  papers/<sanitized-doi>.xml          — original full-text XML
  chunks/<sanitized-doi>.jsonl        — one chunk per line: {text, section, order}
  embeddings/index.jsonl              — one row per chunk: {chunk_id, doi, section, text_preview, paper_idx}
  embeddings/vectors.npy              — 2D float array, one row per chunk in index.jsonl order
  embeddings/manifest.json            — {dim, count, model_id, last_updated}

Cross-run persistent: subsequent runs see prior corpus, only embed new content.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from scout.tools import jats_chunker, markdown_chunker
from scout.tools.embeddings import Embedder
from scout.tools.reranker import Reranker


@dataclass
class Chunk:
    chunk_id: str
    doi: str
    section: str
    text: str
    score: float = 0.0  # populated on retrieval


def _sanitize(doi: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", doi.strip())[:200]


def _doi_hash(doi: str) -> str:
    return hashlib.sha1(doi.encode("utf-8")).hexdigest()[:16]


_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.\-]+")


def _tokenize_for_bm25(text: str) -> list[str]:
    """Lowercase + simple alphanum-with-hyphens-and-dots tokens. Keeps things
    like `NTRK3`, `Kv1.2`, `IL-1β` mostly intact (the β gets dropped, but
    `IL-1` survives). Adequate for BM25 over biomedical chunks."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class RAGStore:
    root: Path
    _embedder: Embedder | None = field(default=None, init=False)
    _reranker: Reranker | None = field(default=None, init=False)
    _index: list[dict] = field(default_factory=list, init=False)
    _vectors: np.ndarray | None = field(default=None, init=False)
    _bm25: object = field(default=None, init=False)
    _bm25_corpus: list[list[str]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        for sub in ("papers", "chunks", "embeddings"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self._load_index()

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = Reranker()
        return self._reranker

    # ---------- index management ----------

    def _index_path(self) -> Path:
        return self.root / "embeddings" / "index.jsonl"

    def _vectors_path(self) -> Path:
        return self.root / "embeddings" / "vectors.npy"

    def _manifest_path(self) -> Path:
        return self.root / "embeddings" / "manifest.json"

    def _load_index(self) -> None:
        idx_path = self._index_path()
        if idx_path.exists():
            self._index = [json.loads(l) for l in idx_path.read_text().splitlines() if l.strip()]
        else:
            self._index = []
        vec_path = self._vectors_path()
        if vec_path.exists() and self._index:
            self._vectors = np.load(vec_path)
        else:
            self._vectors = None
        # BM25 is rebuilt in-memory at load — fast for ~7K chunks. Lazy-built on
        # first hybrid retrieval to keep load() cheap when only metadata is needed.
        self._bm25 = None
        self._bm25_corpus = []

    def _ensure_bm25(self) -> None:
        """Build the BM25 index over chunk text on demand. Reads the per-paper
        chunks/<sanitized>.jsonl files since the in-memory index only stores
        a 200-char preview, not the full chunk text."""
        if self._bm25 is not None or not self._index:
            return
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            return  # silently no-op if rank_bm25 isn't installed
        corpus: list[list[str]] = []
        for row in self._index:
            txt = self._load_chunk_text(row["doi"], row["chunk_id"])
            corpus.append(_tokenize_for_bm25(txt))
        self._bm25_corpus = corpus
        self._bm25 = BM25Okapi(corpus)

    def _persist(self) -> None:
        idx_path = self._index_path()
        idx_path.write_text("\n".join(json.dumps(r) for r in self._index) + ("\n" if self._index else ""))
        if self._vectors is not None:
            np.save(self._vectors_path(), self._vectors)
        manifest = {
            "count": len(self._index),
            "dim": int(self._vectors.shape[1]) if self._vectors is not None else 0,
            "model_id": self.embedder.model_id,
        }
        self._manifest_path().write_text(json.dumps(manifest, indent=2))

    # ---------- paper management ----------

    def has_paper(self, doi: str) -> bool:
        h = _doi_hash(doi)
        return any(r.get("doi_hash") == h for r in self._index)

    def save_pdf(self, doi: str, pdf_bytes: bytes) -> Path | None:
        """Persist the published PDF alongside the XML, for human review.
        Idempotent — if the file exists, no-op. Returns path written or None."""
        if not pdf_bytes:
            return None
        sanitized = _sanitize(doi)
        path = self.root / "papers" / f"{sanitized}.pdf"
        if path.exists():
            return path
        path.write_bytes(pdf_bytes)
        return path

    def update_paper_meta(self, doi: str, **fields) -> None:
        """Append/update metadata for a paper (title, authors, year, etc.)
        in the manifest so humans can browse the archive."""
        manifest_path = self.root / "papers" / "manifest.jsonl"
        existing: dict[str, dict] = {}
        if manifest_path.exists():
            for line in manifest_path.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                existing[rec["doi"]] = rec
        rec = existing.get(doi, {"doi": doi})
        rec.update({k: v for k, v in fields.items() if v is not None})
        sanitized = _sanitize(doi)
        rec["xml_path"] = f"papers/{sanitized}.xml"
        pdf_path = self.root / "papers" / f"{sanitized}.pdf"
        rec["pdf_path"] = f"papers/{sanitized}.pdf" if pdf_path.exists() else None
        existing[doi] = rec
        manifest_path.write_text("\n".join(json.dumps(r) for r in existing.values()) + "\n")

    def add_paper(self, doi: str, xml_text: str, topics: list[str] | None = None) -> int:
        """Parse JATS XML, chunk, embed, and store. Returns # of chunks added.
        No-op if the paper is already in the store. `topics` are normalized
        keyword tags (lowercased) attached to every chunk for thematic
        retrieval filtering downstream."""
        return self._add_chunks(doi, jats_chunker.chunks_from_jats(xml_text),
                                source_kind="xml", source_text=xml_text, topics=topics)

    def add_paper_markdown(self, doi: str, md_text: str, topics: list[str] | None = None) -> int:
        """Parse Markdown (e.g. from Marker), chunk, embed, and store.
        Used as the fallback when JATS XML is unavailable."""
        return self._add_chunks(doi, markdown_chunker.chunks_from_markdown(md_text),
                                source_kind="md", source_text=md_text, topics=topics)

    def _add_chunks(self, doi: str, chunks: list[dict], source_kind: str,
                    source_text: str, topics: list[str] | None = None) -> int:
        if self.has_paper(doi):
            return 0
        sanitized = _sanitize(doi)
        # Cache the raw source for traceability + future re-chunking
        ext = "xml" if source_kind == "xml" else "md"
        (self.root / "papers" / f"{sanitized}.{ext}").write_text(source_text)
        if not chunks:
            return 0
        (self.root / "chunks" / f"{sanitized}.jsonl").write_text(
            "\n".join(json.dumps(c) for c in chunks)
        )

        # Embed
        texts = [c["text"] for c in chunks]
        new_vectors = np.array(self.embedder.embed(texts, input_type="search_document"), dtype=np.float32)

        # Append to index
        h = _doi_hash(doi)
        start = len(self._index)
        norm_topics = sorted({t.strip().lower() for t in (topics or []) if t and t.strip()})
        for i, c in enumerate(chunks):
            self._index.append({
                "chunk_id": f"{h}:{i}",
                "doi": doi,
                "doi_hash": h,
                "section": c.get("section", ""),
                "text_preview": c["text"][:200],
                "row": start + i,
                "topics": norm_topics,
            })

        if self._vectors is None:
            self._vectors = new_vectors
        else:
            self._vectors = np.vstack([self._vectors, new_vectors])

        self._persist()
        return len(chunks)

    # ---------- retrieval ----------

    def retrieve(self, query: str, k: int = 10, candidate_pool: int | None = None,
                 topics_filter: set[str] | None = None) -> list[Chunk]:
        """Retrieve top-k chunks for a query.

        If `candidate_pool` is provided, > k, and the reranker is enabled
        (VP_BEDROCK_RERANK_ID set), retrieves a wider cosine pool and reranks
        with Cohere Rerank 3.5. Otherwise pure cosine.

        If `topics_filter` is provided (non-empty set of normalized lowercase
        topic strings), only chunks whose stored `topics` intersect the filter
        are eligible. Untagged legacy chunks (with empty `topics`) are
        EXCLUDED when a filter is active. This is the thematic-retrieval
        contract — the caller decides which corpus slice is in scope.

        The returned `score` field reflects whichever stage produced the final
        ranking — rerank relevance score when reranked, cosine similarity otherwise.
        """
        if not self._index or self._vectors is None or len(self._index) == 0:
            return []
        qv = np.array(self.embedder.embed_query(query), dtype=np.float32)
        norms = np.linalg.norm(self._vectors, axis=1) * np.linalg.norm(qv) + 1e-12
        sims = (self._vectors @ qv) / norms

        # Apply topical mask before ranking — saves work on filtered chunks.
        mask: np.ndarray | None = None
        if topics_filter:
            tf = {t.strip().lower() for t in topics_filter if t and t.strip()}
            mask = np.array([
                bool(set(self._index[i].get("topics") or []) & tf)
                for i in range(len(self._index))
            ])
            if not mask.any():
                return []  # nothing matches the filter; caller can retry without
            sims = np.where(mask, sims, -np.inf)

        use_rerank = (
            candidate_pool is not None
            and candidate_pool > k
            and self.reranker.enabled
        )
        first_stage_n = candidate_pool if use_rerank else k

        # Hybrid candidates: union of top-N dense + top-N BM25. BM25 catches
        # exact-token matches (gene names, IDs) that dense retrieval often
        # misses in biomedical text.
        dense_idx = np.argsort(-sims)[:first_stage_n].tolist()
        self._ensure_bm25()
        bm25_idx: list[int] = []
        if self._bm25 is not None:
            try:
                bm_scores = self._bm25.get_scores(_tokenize_for_bm25(query))
                if mask is not None:
                    bm_scores = np.where(mask, bm_scores, -np.inf)
                bm25_idx = np.argsort(-bm_scores)[:first_stage_n].tolist()
            except Exception:
                bm25_idx = []
        # Dedup — keep dense order first (it's our primary signal), append
        # BM25 hits not already in the set.
        seen: set[int] = set()
        merged: list[int] = []
        for i in dense_idx + bm25_idx:
            ii = int(i)
            if ii not in seen and (mask is None or mask[ii]):
                seen.add(ii)
                merged.append(ii)
        first_stage_idx = np.array(merged, dtype=int)

        if use_rerank:
            texts: list[str] = []
            keep_idx: list[int] = []
            for i in first_stage_idx:
                row = self._index[int(i)]
                txt = self._load_chunk_text(row["doi"], row["chunk_id"])
                if txt:
                    texts.append(txt)
                    keep_idx.append(int(i))
            ranked = self.reranker.rerank(query, texts, top_n=k)
            out: list[Chunk] = []
            for pool_pos, score in ranked:
                global_idx = keep_idx[pool_pos]
                row = self._index[global_idx]
                out.append(Chunk(
                    chunk_id=row["chunk_id"],
                    doi=row["doi"],
                    section=row["section"],
                    text=texts[pool_pos],
                    score=float(score),
                ))
            return out

        out = []
        for i in first_stage_idx:
            row = self._index[int(i)]
            text = self._load_chunk_text(row["doi"], row["chunk_id"])
            out.append(Chunk(
                chunk_id=row["chunk_id"],
                doi=row["doi"],
                section=row["section"],
                text=text,
                score=float(sims[int(i)]),
            ))
        return out

    def _load_chunk_text(self, doi: str, chunk_id: str) -> str:
        sanitized = _sanitize(doi)
        path = self.root / "chunks" / f"{sanitized}.jsonl"
        if not path.exists():
            return ""
        idx = int(chunk_id.split(":")[-1])
        for line_no, line in enumerate(path.read_text().splitlines()):
            if line_no == idx:
                return json.loads(line).get("text", "")
        return ""

    def count_on_topic_papers(self, topics_filter: set[str] | None) -> int:
        """Count unique DOIs in the corpus whose chunk-level topics overlap
        the filter. Untagged papers count as 0 — they're invisible to
        thematic retrieval, so they don't contribute to coverage."""
        if not topics_filter:
            return 0
        tf = {t.strip().lower() for t in topics_filter if t and t.strip()}
        if not tf:
            return 0
        on_topic_dois: set[str] = set()
        for row in self._index:
            if set(row.get("topics") or []) & tf:
                on_topic_dois.add(row.get("doi", ""))
        on_topic_dois.discard("")
        return len(on_topic_dois)

    # ---------- stats ----------

    def stats(self) -> dict:
        papers = {r["doi"] for r in self._index}
        sections: dict[str, int] = {}
        for r in self._index:
            sec = r.get("section", "")
            sections[sec] = sections.get(sec, 0) + 1
        return {
            "papers": len(papers),
            "chunks": len(self._index),
            "model_id": self.embedder.model_id if self._embedder else None,
            "section_counts": dict(sorted(sections.items(), key=lambda x: -x[1])[:8]),
        }
