# pennsieve-ai-scout

Pennsieve compute node processor — **stage 1 of 4** of the `pennsieve-ai`
research pipeline.

Takes a Pennsieve dataset DOI, gathers the surrounding literature and dataset
context (across Pennsieve / DANDI / OpenNeuro), and produces a populated
blackboard for the downstream stages: `pennsieve-ai-hypothesize`,
`pennsieve-ai-analyze`, `pennsieve-ai-write`.

## Input

| Env var | Source | Description |
|---|---|---|
| `SEED_DOI` | user parameter | Pennsieve dataset DOI to seed the run (e.g. `10.26275/abcd-efgh`) |
| `OUTPUT_DIR` | platform-injected | Where to write `blackboard.json` and artifacts |
| `EXECUTION_RUN_ID` | platform-injected | Governor budget attribution key |
| `WORKFLOW_INSTANCE_ID` | platform-injected | Workflow execution identifier |
| `LLM_GOVERNOR_FUNCTION` | platform-injected | Lambda name for the LLM governor |

No `INPUT_DIR` consumption — Scout is the first stage.

## Output

Writes `$OUTPUT_DIR/blackboard.json` with:

- Seed dataset summary (README, file types, record tables, linked publications)
- Cross-repo similar datasets (Pennsieve + DANDI + OpenNeuro)
- Linked + citing + reference papers (Europe PMC + CrossRef)
- Dataset compatibility verdicts across the candidate datasets
- Literature claims (verified against papers seen)
- Deep evidence chunks (retrieved from the persistent RAG layer, if mounted)
- Dataset framer brief (research focus, suggested queries)

Downstream stages consume this via the standard Pennsieve `INPUT_DIR`/`OUTPUT_DIR` contract.

## Layers consumed

| Layer | Mount | Producer |
|---|---|---|
| `pennsieve-ai-rag` | `/mnt/layers/pennsieve-ai-rag/` (read-only) | `pennsieve-ai-rag-loader` workflow |

If the layer is absent, `bb.deep_evidence` is empty and downstream stages degrade gracefully (no full-text retrieval, only titles/abstracts).

## LLM access

All LLM calls route through the Pennsieve LLM Governor Lambda
(`$LLM_GOVERNOR_FUNCTION`) — see `src/scout/llm.py`. Models are addressed by
logical name (`opus`, `sonnet`, `haiku`) and resolved to Bedrock model IDs at
runtime.

## Status

**v1 — wired end-to-end, untested in Pennsieve compute.** Agents, tools, and
prompts vendored from the original `virtual-postdoc` research codebase.
Import graph fully resolves; pipeline runs locally given env vars + an
operational governor Lambda. Not yet deployed to a Pennsieve compute node.

### Known caveats to verify before first deploy

- **Embeddings + reranker call Bedrock directly** (`tools/embeddings.py`,
  `tools/reranker.py` — both `boto3.client("bedrock-runtime")`). They bypass
  the governor. Compute-node IAM role needs `bedrock:InvokeModel` permission
  for the chosen embedding/rerank model, or these calls fail. Only matters
  if the RAG layer is mounted (retrieval triggers embedding the query). When
  the governor's `/v1/embed` and `/v1/rerank` HTTP routes ship, these
  modules should migrate to use them.
- **Model IDs in `llm.py`** are placeholders. The real Bedrock model IDs
  come from the governor's SSM allow-list; override via
  `VP_MODEL_OPUS` / `VP_MODEL_SONNET` / `VP_MODEL_HAIKU` env vars.

## Pipeline

```
[seed DOI]
   │
   ▼
1. resolve DOI → Pennsieve dataset ID (Discover API)
2. dataset inspection + topic extraction
3. cross-repo similar dataset discovery (Pennsieve + DANDI + OpenNeuro)
4. pull linked / citing / reference papers (Europe PMC, CrossRef)
5. dataset_compatibility → cross-dataset usability verdicts
6. dataset_framer  → research focus + suggested queries
7. lit_scout       → Europe PMC search on suggested queries
8. verifier        → ground claims against papers_seen
9. deep_reader     → retrieve chunks from /mnt/layers/pennsieve-ai-rag/
   │
   ▼
$OUTPUT_DIR/blackboard.json
```

Deferred to v2: `collaboration_scout` (currently dataset-first runs have an
empty `lab_profile`, so this agent has nothing to do).
