"""Entrypoint for the pennsieve-ai-scout processor.

Reads `SEED_DOI` from the environment, resolves it to a Pennsieve dataset ID
via the Discover API, runs the scout pipeline, and writes `blackboard.json`
to `$OUTPUT_DIR` for the downstream Hypothesis stage.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from scout.blackboard import Blackboard
from scout.llm import LLMClient
from scout.pipeline import run_scout
from scout.tools import pennsieve_discover

MAX_ITERATIONS = 3  # not used by Scout itself; carried on the blackboard for downstream stages


def main() -> int:
    seed_doi = os.environ.get("SEED_DOI", "").strip()
    if not seed_doi:
        print("ERROR: SEED_DOI environment variable is required.", file=sys.stderr)
        return 2

    output_dir = Path(os.environ.get("OUTPUT_DIR", "/mnt/efs/output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    execution_run_id = os.environ.get("EXECUTION_RUN_ID", "local-dev")
    workflow_instance_id = os.environ.get("WORKFLOW_INSTANCE_ID", "local-dev")

    print("=== pennsieve-ai-scout ===")
    print(f"  SEED_DOI={seed_doi}")
    print(f"  EXECUTION_RUN_ID={execution_run_id}")
    print(f"  WORKFLOW_INSTANCE_ID={workflow_instance_id}")
    print(f"  OUTPUT_DIR={output_dir}")

    # --- Resolve DOI → Pennsieve dataset ID ---
    print(f"\n--- resolving DOI {seed_doi} ---")
    detail = pennsieve_discover.resolve_doi(seed_doi)
    if not detail:
        print(f"ERROR: could not resolve {seed_doi} to a Pennsieve dataset. "
              f"Verify the DOI exists in Pennsieve Discover.", file=sys.stderr)
        return 3
    pennsieve_id = detail.get("id")
    if not pennsieve_id:
        print(f"ERROR: Discover record for {seed_doi} has no id field: {detail}",
              file=sys.stderr)
        return 4
    seed_dataset_id = f"pennsieve:{pennsieve_id}"
    print(f"  resolved → {seed_dataset_id}  ({detail.get('name', '')!r})")

    # --- Initialize blackboard ---
    research_question = (
        f"What productive follow-up research question can be answered using "
        f"{seed_dataset_id} ({detail.get('name', '') or 'the seed dataset'}) — "
        f"building on or extending its prior published work?"
    )
    bb = Blackboard(
        research_question=research_question,
        subsystem=seed_dataset_id,
        run_id=execution_run_id,
        max_iterations=MAX_ITERATIONS,
        lab_profile={},
    )
    bb.log("scout", "start",
           f"seed_doi={seed_doi}, seed_dataset={seed_dataset_id}, "
           f"workflow_instance={workflow_instance_id}")

    client = LLMClient()

    # --- Run the scout pipeline ---
    try:
        run_scout(bb, client, seed_dataset_id=seed_dataset_id)
    except Exception as e:
        bb.log("scout", "fatal", repr(e))
        _write_outputs(bb, client, output_dir)
        raise

    bb.log("scout", "done", "")

    # --- Write outputs to $OUTPUT_DIR ---
    _write_outputs(bb, client, output_dir)
    print(f"\nWrote: {output_dir / 'blackboard.json'}")
    return 0


def _write_outputs(bb: Blackboard, client: LLMClient, output_dir: Path) -> None:
    """Persist blackboard + audit logs to OUTPUT_DIR for the next stage."""
    bb.save(output_dir / "blackboard.json")
    nb = "\n".join(json.dumps(e) for e in bb.notebook)
    (output_dir / "notebook.jsonl").write_text(nb + "\n" if nb else "")
    ul = "\n".join(json.dumps(e) for e in client.usage_log)
    (output_dir / "usage.jsonl").write_text(ul + "\n" if ul else "")


if __name__ == "__main__":
    raise SystemExit(main())
