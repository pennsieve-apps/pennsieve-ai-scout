"""Dataset Framer — replaces the Director in --from-dataset mode.

Reads the seed dataset (deeply: README, manifest, record tables, linked
publications, citing papers, similar datasets), then synthesizes a
DirectorBrief grounded in *this dataset* rather than a fixed lab profile.

The brief shape is identical to the Director's, so downstream agents
(Lit Scout, Verifier, Deep Reader, HypGen, Critic, Writer) require no
changes — they read `bb.director_brief` exactly as before.
"""
from __future__ import annotations

from scout.agents._common import call_json, format_datasets, load_prompt
from scout.blackboard import Blackboard, DirectorBrief
from scout.llm import LLMClient

MODEL = "opus"
AGENT = "dataset_framer"


def _format_papers(papers: list[dict], role: str) -> str:
    chunks = []
    for i, p in enumerate(papers):
        chunks.append(
            f"[{role[0].upper()}{i}] {p.get('title','')} ({p.get('year','')}, {p.get('journal','')})\n"
            f"    DOI: {p.get('doi','')}\n"
            f"    Abstract: {(p.get('abstract','') or '(no abstract)')[:1500]}"
        )
    return "\n\n".join(chunks) if chunks else "(none)"


def run(
    bb: Blackboard,
    client: LLMClient,
    *,
    seed_dataset_id: str,
    linked_papers: list[dict],
    citing_papers: list[dict],
    reference_papers: list[dict] | None = None,
    scope_override: str | None = None,
) -> None:
    """Synthesize a dataset-anchored brief and store it on the blackboard.

    Pre-conditions: bb.dataset_summaries already contains the seed (first) and
    similar datasets, with full inspection data populated.
    """
    if not bb.dataset_summaries:
        bb.log(AGENT, "skip", "no dataset_summaries on the blackboard")
        return

    system = load_prompt("dataset_framer")
    seed = bb.dataset_summaries[0]
    similars = bb.dataset_summaries[1:]

    scope_note = (
        f"## Scope override (user-specified)\n\n{scope_override}\n\n"
        if scope_override else
        "## Scope\n\nNo cross-domain scope specified — stay strictly within this dataset's natural domain.\n\n"
    )

    user = "\n".join([
        f"# Seed dataset: {seed_dataset_id}\n",
        scope_note,
        "## Seed dataset (full metadata)\n",
        format_datasets(_make_subset(bb, [seed])),
        "",
        "## Similar datasets (also available for combined analyses)\n",
        format_datasets(_make_subset(bb, similars)) if similars else "(none found)",
        "",
        f"## Linked publications ({len(linked_papers)} — papers the depositor attached)\n",
        _format_papers(linked_papers, "linked"),
        "",
        f"## Foundational references ({len(reference_papers or [])} — most-cited works the linked publication built on)\n",
        _format_papers(reference_papers or [], "reference"),
        "",
        f"## Citing papers ({len(citing_papers)} — what's been done since)\n",
        _format_papers(citing_papers, "citing"),
    ])

    parsed = call_json(
        client,
        agent=AGENT,
        system=system,
        user=user,
        run_id=bb.run_id,
        model=MODEL,
    )

    bb.director_brief = DirectorBrief(
        field_landscape=parsed.get("field_landscape", ""),
        research_strategy=parsed.get("research_strategy", ""),
        focus_areas=parsed.get("focus_areas", []) or [],
        lab_constraints=parsed.get("lab_constraints", []) or [],
        acceptance_criteria=parsed.get("acceptance_criteria", []) or [],
        suggested_queries=parsed.get("suggested_queries", []) or [],
        review_papers=parsed.get("review_papers", []) or [],
    )
    bb.log(
        AGENT, "brief",
        f"focus_areas={len(bb.director_brief.focus_areas)}, "
        f"acceptance_criteria={len(bb.director_brief.acceptance_criteria)}, "
        f"suggested_queries={len(bb.director_brief.suggested_queries)}, "
        f"reviewed={len(bb.director_brief.review_papers)}",
    )


def _make_subset(bb: Blackboard, picks: list) -> Blackboard:
    """Tiny throwaway blackboard so we can reuse format_datasets() on a subset."""
    sub = Blackboard(research_question=bb.research_question, subsystem=bb.subsystem)
    sub.dataset_summaries = list(picks)
    return sub
