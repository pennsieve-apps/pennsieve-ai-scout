import gzip
import json
import numpy as np
import pytest

from scout.tools.rag_store import RAGStore


def artifact(root):
    manifest = {"dataset_doi": "10.1/seed", "embedding_model_id": "amazon.titan-embed-text-v2:0", "dim": 256, "chunk_count": 2}
    (root / "manifest.json").write_text(json.dumps(manifest))
    rows = [{"chunk_id": "paper:0", "paper_doi": "10.2/paper", "text": "Measured evidence", "embedding": [1.0] + [0.0] * 255},
            {"chunk_id": "metadata:0", "source_type": "dataset_files", "dataset_doi": "10.1/seed", "text": "Folder description", "embedding": [1.0] + [0.0] * 255}]
    with gzip.open(root / "chunks.jsonl.gz", "wt") as stream:
        stream.write("\n".join(json.dumps(r) for r in rows))


def test_artifact_retrieval_is_seed_scoped_and_excludes_dataset_metadata(tmp_path):
    artifact(tmp_path)
    store = RAGStore(tmp_path, read_only=True, dataset_doi="10.1/seed")
    store._embedder = type("Embedder", (), {"embed_query": lambda self, q: [1.0] + [0.0] * 255})()
    chunks = store.retrieve("query", k=1, topics_filter={"not-a-legacy-topic"})
    assert len(chunks) == 1
    assert chunks[0].doi == "10.2/paper" and chunks[0].text == "Measured evidence"
    assert np.isclose(chunks[0].score, 1)
    assert not (tmp_path / "embeddings").exists()


def test_artifact_rejects_wrong_seed_and_unsupported_model(tmp_path):
    artifact(tmp_path)
    with pytest.raises(ValueError, match="seed dataset"):
        RAGStore(tmp_path, read_only=True, dataset_doi="10.1/other")
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["embedding_model_id"] = "nomic-embed-text"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsupported query embedding model"):
        RAGStore(tmp_path, read_only=True, dataset_doi="10.1/seed")
