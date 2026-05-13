"""Bedrock reranker — Cohere Rerank 3.5 by default, gated by env var.

Set VP_BEDROCK_RERANK_ID=cohere.rerank-v3-5:0 to enable a cross-encoder
rerank pass on top of cosine retrieval. Unset => no-op (cosine only).

Probed 2026-05-05: invoke_model path works in us-east-1 with the standard
Cohere request body. See scripts/test_rerank.py.
"""
from __future__ import annotations

import json
import os

import boto3

DEFAULT_REGION = os.environ.get("AWS_REGION", "us-east-1")
COHERE_MAX_DOCS = 1000  # well above any single-call need; keep simple


class Reranker:
    """Optional cross-encoder rerank pass. Disabled when VP_BEDROCK_RERANK_ID is unset."""

    def __init__(self, model_id: str | None = None, region: str = DEFAULT_REGION):
        self.model_id = model_id if model_id is not None else os.environ.get("VP_BEDROCK_RERANK_ID")
        self._client = None
        self._region = region

    @property
    def enabled(self) -> bool:
        return bool(self.model_id)

    def _ensure_client(self) -> None:
        if self._client is None:
            self._client = boto3.client("bedrock-runtime", region_name=self._region)

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        """Score documents against query; return [(original_idx, score), ...] sorted desc.

        Length of return is min(top_n, len(documents)). Raises if disabled — callers
        should check `enabled` first."""
        if not self.enabled:
            raise RuntimeError("Reranker not enabled (VP_BEDROCK_RERANK_ID unset)")
        if not documents:
            return []
        self._ensure_client()
        body = {
            "query": query,
            "documents": documents[:COHERE_MAX_DOCS],
            "top_n": min(top_n, len(documents)),
            "api_version": 2,
        }
        resp = self._client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType="application/json",
        )
        payload = json.loads(resp["body"].read())
        return [
            (int(r["index"]), float(r.get("relevance_score", 0.0)))
            for r in payload.get("results", [])
        ]
