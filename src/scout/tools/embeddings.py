"""Bedrock embeddings client — Titan default, Cohere available via env var.

Set:
  VP_BEDROCK_EMBED_ID=amazon.titan-embed-text-v2:0  (default — universally available)
  VP_BEDROCK_EMBED_ID=cohere.embed-english-v3       (better retrieval quality if enabled)

Cohere supports batched calls (96 texts/call) and faster overall; Titan is per-call.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

import boto3

DEFAULT_MODEL = os.environ.get("VP_BEDROCK_EMBED_ID", "amazon.titan-embed-text-v2:0")
DEFAULT_REGION = os.environ.get("AWS_REGION", "us-east-1")
COHERE_BATCH = 96
TITAN_PARALLELISM = int(os.environ.get("VP_TITAN_PARALLELISM", "16"))


class Embedder:
    def __init__(self, model_id: str = DEFAULT_MODEL, region: str = DEFAULT_REGION):
        self._client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id

    def _is_cohere(self) -> bool:
        return "cohere" in self.model_id.lower()

    def _embed_titan(self, text: str) -> list[float]:
        resp = self._client.invoke_model(
            modelId=self.model_id,
            body=json.dumps({"inputText": text}),
            contentType="application/json",
        )
        payload = json.loads(resp["body"].read())
        return payload["embedding"]

    def _embed_cohere_batch(self, texts: list[str], input_type: str) -> list[list[float]]:
        resp = self._client.invoke_model(
            modelId=self.model_id,
            body=json.dumps({"texts": texts, "input_type": input_type}),
            contentType="application/json",
        )
        payload = json.loads(resp["body"].read())
        return payload["embeddings"]

    def embed(self, texts: list[str], input_type: str = "search_document") -> list[list[float]]:
        """Embed a list of texts. `input_type` is honored by Cohere (search_document
        for indexing, search_query for queries) and ignored by Titan."""
        if not texts:
            return []
        if self._is_cohere():
            out: list[list[float]] = []
            for i in range(0, len(texts), COHERE_BATCH):
                out.extend(self._embed_cohere_batch(texts[i:i + COHERE_BATCH], input_type))
            return out
        # Titan: one call per text — run in parallel to mask network RTT.
        if len(texts) == 1:
            return [self._embed_titan(texts[0])]
        with ThreadPoolExecutor(max_workers=TITAN_PARALLELISM) as ex:
            return list(ex.map(self._embed_titan, texts))

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text], input_type="search_query")[0]
