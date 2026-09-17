"""Entrypoint for the pennsieve-ai-scout processor.

Reads `SEED_DOI` from the environment, resolves it to a Pennsieve dataset ID
via the Discover API, runs the scout pipeline, and writes `blackboard.json`
to `$OUTPUT_DIR` for the downstream Hypothesis stage.
"""
from __future__ import annotations

import os
import sys

from pennsieve_ai_utils.blackboard import Blackboard
from pennsieve_ai_utils.llm import LLMClient
from pennsieve_ai_utils.models import ModelNotAvailable, degraded_ok
from pennsieve_ai_utils.processor import ProcessorEnv, install_sigterm_handler, write_outputs

from scout.pipeline import run_scout
from scout.tools import pennsieve_discover

STAGE = "scout"
MAX_ITERATIONS = 3  # not used by Scout itself; carried on the blackboard for downstream stages


def main() -> int:
    install_sigterm_handler()
    seed_doi = os.environ.get("SEED_DOI", "").strip()
    if not seed_doi:
        print("ERROR: SEED_DOI environment variable is required.", file=sys.stderr)
        return 2

    env = ProcessorEnv.from_environ()
    env.banner(STAGE, SEED_DOI=seed_doi)
    output_dir = env.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

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
        run_id=env.execution_run_id,
        max_iterations=MAX_ITERATIONS,
        lab_profile={},
    )
    bb.log(STAGE, "start",
           f"seed_doi={seed_doi}, seed_dataset={seed_dataset_id}, "
           f"workflow_instance={env.workflow_instance_id}")
    bb.save(output_dir / "blackboard.json")

    # --- LLM preflight: learn the allow-list, assign roles, fail loud ---
    # Done before any pipeline work so an inadequate model for a
    # reasoning-critical role costs nothing rather than a full run. Scout is
    # the only stage that uses all three roles (bulk drives topic extraction).
    client = LLMClient(stage=STAGE)
    try:
        assignments = client.preflight(strict=not degraded_ok())
    except ModelNotAvailable as e:
        print(f"ERROR: {e}", file=sys.stderr)
        bb.log(STAGE, "fatal", f"model preflight failed: {e}")
        write_outputs(bb, output_dir, client.usage_log)
        return 5
    for role, assignment in assignments.items():
        bb.record_model_assignment(stage=STAGE, role=role, assignment=assignment)
    bb.log(STAGE, "models",
           ", ".join(f"{role}={a['model_id']}" for role, a in sorted(assignments.items())))

    # --- Run the scout pipeline ---
    try:
        run_scout(bb, client, seed_dataset_id=seed_dataset_id)
    except (Exception, KeyboardInterrupt) as e:
        bb.log(STAGE, "fatal", repr(e))
        write_outputs(bb, output_dir, client.usage_log)
        raise

    if bb.degraded:
        skipped = ", ".join(d["stage"] for d in bb.degraded)
        bb.log(STAGE, "done_degraded", f"skipped stages: {skipped}")
        print(f"\nWARNING: completed with {len(bb.degraded)} skipped stage(s): "
              f"{skipped}\n  This blackboard is INCOMPLETE — output built on it is "
              f"not a full run.", file=sys.stderr)
    else:
        bb.log(STAGE, "done", "")

    write_outputs(bb, output_dir, client.usage_log)
    print(f"\nWrote: {output_dir / 'blackboard.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
