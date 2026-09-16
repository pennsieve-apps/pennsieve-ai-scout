# pennsieve-ai-scout

Local hardening update: seed inspection now uses Discover public assets/downloads, records access diagnostics, and stops strict runs with no inspectable seed files. RAG readers accept dataset-scoped compressed artifacts; set `VP_REQUIRE_DEEP_EVIDENCE=1` to require full text. Docker deployment is still pending.

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
(`$LLM_GOVERNOR_FUNCTION`) — see `src/scout/llm.py`.

Models are addressed by the **role they play in the pipeline**, never by
vendor family — see `src/scout/models.py`:

| Role | Used by | Floor | Ceiling |
|---|---|---|---|
| `reasoning` | `dataset_framer` | Sonnet 4.6 | — |
| `synthesis` | `verifier`, `dataset_compatibility`, `lit_scout` | Sonnet 4.5 | Opus 4.6 |
| `bulk` | `topic_extractor` | Haiku 3.5 | Haiku 4.5 |

Selection is rank-based rather than a hardcoded preference chain: each role
takes the highest-ranked permitted model inside its floor/ceiling band, so a
model Scout has never seen can still be ordered against ones it has. The
floor is what stops a full run from silently degrading onto a model whose
output isn't worth the compute the later stages spend on it; the ceiling
keeps high-volume roles off expensive models.

**Preflight.** Bedrock has no Models API, so the governor's allow-list is
discovered by deliberately tripping its `model_not_allowed` 403 with a
sentinel model ID before any real work starts. The resulting manifest is
printed at startup and recorded on the blackboard as `model_assignments`, so
a downstream reader can tell whether a claim came from a reasoning-grade
model or a degraded one. If no permitted model meets a role's floor, the run
aborts at preflight (exit 5) with an actionable message instead of spending
the full budget. Should a model still be rejected mid-run, the allow-list is
re-learned from the 403 and the role re-resolved once.

**Overrides.** `VP_MODEL_REASONING` / `VP_MODEL_SYNTHESIS` / `VP_MODEL_BULK`
pin a role to a specific model ID, bypassing floor and ceiling — explicit
operator intent, flagged in the manifest but honored. The legacy
`VP_MODEL_OPUS` / `_SONNET` / `_HAIKU` variables still map to
`reasoning` / `synthesis` / `bulk`. Set `VP_MODEL_SELECT=1` on an
interactive terminal to choose per role from the allow-list at startup.

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
- **Thinking and effort are not yet wired.** No `thinking` parameter is
  sent, so a role served by a model that runs adaptive thinking by default
  (Opus 5) spends output tokens on reasoning without `output_config.effort`
  tuning. `bulk`'s ceiling keeps the 512-token `topic_extractor` off those
  models for now; lifting it needs the thinking/effort work first.
- **JSON is still forced with the prefill trick** rather than structured
  outputs (`output_config.format`), which is GA on Bedrock and would remove
  the prefill/no-prefill branching entirely. Blocked on confirming the
  governor's lambda-invoke transport passes `output_config` through.

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
