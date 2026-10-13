"""Dataset Compatibility Analyst — gatekeeps cross-dataset hypothesis generation.

Given the seed dataset and the candidate "similar" datasets the system
retrieved by tag overlap, decide for each candidate whether it is genuinely
usable for cross-dataset hypothesis testing — and if so, which specific
cross-dataset hypotheses are warranted.

Most candidates surface by shallow tag matches that don't survive
methodological scrutiny: "both have antibodies" can mean mucosal IgA in one
and anti-tumor IgG in the other. The Hypothesis Generator and Critic should
not waste tokens on those false positives. This agent's output flows into
HypGen's prompt as a structured per-candidate verdict (usable / partial /
not_usable) with explicit shared modalities, incompatibilities, and
candidate cross-dataset hypotheses; HypGen is told to prefer the suggested
hypotheses for `usable` candidates and to ignore `not_usable` ones.

One Sonnet call per run, ~$0.05.
"""
from __future__ import annotations

from scout.agents._common import call_json, load_prompt
from pennsieve_ai_utils.blackboard import Blackboard, DatasetSummary
from pennsieve_ai_utils.llm import LLMClient

MODEL = "synthesis"
AGENT = "dataset_compatibility"

_VALID_VERDICTS = {"usable", "partial", "not_usable"}


def _format_dataset(d: DatasetSummary, label: str) -> str:
    """Render a DatasetSummary for the compatibility analyst — README,
    file_tree, record_tables, measurements. Same depth as what HypGen
    sees, so the analyst's verdict reflects what HypGen would see."""
    parts = [f"### {label}: {d.id} — {d.title}"]
    if d.description:
        parts.append(f"**Description:** {d.description[:500]}")
    if d.measurements:
        parts.append(f"**Measurements:** {', '.join(d.measurements)}")
    if d.species:
        parts.append(f"**Species:** {d.species}")
    if d.n_subjects:
        parts.append(f"**n_subjects:** {d.n_subjects}")
    if d.file_types:
        parts.append(f"**File types:** {dict(list(d.file_types.items())[:8])}")
    if d.readme:
        parts.append(f"**README excerpt:**\n{d.readme[:1500]}")
    if d.record_tables:
        parts.append("**Record tables:**")
        for fn, info in list(d.record_tables.items())[:4]:
            cols = ", ".join((info.get("columns") or [])[:10])
            parts.append(f"  `{fn}`: cols=[{cols}], n_rows={info.get('n_rows', 0)}")
    if d.file_tree:
        parts.append(f"**File tree (truncated):**\n{d.file_tree[:1000]}")
    return "\n".join(parts)


def run(bb: Blackboard, client: LLMClient) -> None:
    """Compatibility check across `bb.dataset_summaries`. The first entry is
    the seed; the rest are candidate similar datasets. Result is persisted
    on `bb.cross_dataset_compatibility` for downstream agents (HypGen, Critic)
    to consume.

    No-op if there are zero or one dataset_summaries (nothing to compare)."""
    if not bb.dataset_summaries or len(bb.dataset_summaries) < 2:
        bb.log(AGENT, "skip", "fewer than 2 datasets — nothing to compare")
        bb.cross_dataset_compatibility = []
        return

    seed = bb.dataset_summaries[0]
    candidates = bb.dataset_summaries[1:]

    system = load_prompt("dataset_compatibility")
    user_parts = [
        "## Seed dataset (the reference for compatibility judgments)",
        _format_dataset(seed, "SEED"),
        "",
        f"## Candidate similar datasets ({len(candidates)} to assess)",
        "",
    ]
    for i, c in enumerate(candidates, 1):
        user_parts.append(_format_dataset(c, f"CANDIDATE {i}"))
        user_parts.append("")
    user_parts.append(
        "Assess each candidate against the seed. Most candidates surface "
        "by shallow tag overlap and should be marked `not_usable`. Surface "
        "true cross-dataset opportunities (replication, cross-species, "
        "method comparison, treatment-effect) explicitly when they exist."
    )

    # Errors are raised to the pipeline's stage wrapper, which stops the run
    # (or, with VP_DEGRADED_OK=1, records the step as degraded). Catching them
    # here made a failed call look like "no compatible datasets".
    parsed = call_json(
        client,
        agent=AGENT,
        system=system,
        user="\n\n".join(user_parts),
        run_id=bb.run_id,
        model=MODEL,
        max_tokens=4096,
    )

    raw = parsed.get("compatibility_assessments") or []
    cleaned: list[dict] = []
    for a in raw:
        if not isinstance(a, dict):
            continue
        v = a.get("verdict", "not_usable")
        if v not in _VALID_VERDICTS:
            v = "not_usable"
        cleaned.append({
            "dataset_id": a.get("dataset_id", "") or "",
            "verdict": v,
            "shared_modalities": a.get("shared_modalities") or [],
            "cohort_match": a.get("cohort_match", "") or "",
            "incompatibilities": a.get("incompatibilities") or [],
            "suggested_cross_dataset_hypotheses": a.get("suggested_cross_dataset_hypotheses") or [],
            "rationale": a.get("rationale", "") or "",
        })
    bb.cross_dataset_compatibility = cleaned

    n_usable = sum(1 for a in cleaned if a["verdict"] == "usable")
    n_partial = sum(1 for a in cleaned if a["verdict"] == "partial")
    n_not = sum(1 for a in cleaned if a["verdict"] == "not_usable")
    bb.log(
        AGENT, "complete",
        f"{len(cleaned)} candidates assessed: "
        f"{n_usable} usable, {n_partial} partial, {n_not} not_usable",
    )


def format_for_hypgen(bb: Blackboard) -> str:
    """Render the compatibility assessment for inclusion in HypGen's user
    message. Returns "" when no assessment is available (silent no-op)."""
    items = getattr(bb, "cross_dataset_compatibility", None) or []
    if not items:
        return ""
    lines = [
        "## Cross-dataset opportunity (compatibility-vetted similar datasets)",
        "",
        "The system retrieved several similar datasets by tag overlap; an "
        "explicit compatibility analyst then assessed each for genuine "
        "cross-dataset usability. Trust this assessment when designing "
        "hypotheses:",
        "",
        "- **`usable`** candidates share a real measurement modality with "
        "the seed and the cohorts/schemas are tractable. Prefer cross-"
        "dataset replication or comparison hypotheses where they apply, "
        "using the suggested cross-dataset hypotheses below as starting "
        "points.",
        "- **`partial`** candidates share a modality but require harmonization "
        "(different sampling rates, partly different cohorts, etc.). Cross-"
        "dataset hypotheses are possible but must engage explicitly with the "
        "named incompatibilities.",
        "- **`not_usable`** candidates surfaced from shallow tag overlap "
        "(e.g. both have 'antibody' but in unrelated biology). **Do NOT "
        "design cross-dataset hypotheses against these.** They are listed "
        "for transparency only; treat them as out of scope.",
        "",
    ]
    for a in items:
        lines.append(f"### {a['dataset_id']} — verdict: **{a['verdict']}**")
        if a.get("rationale"):
            lines.append(f"**Why:** {a['rationale']}")
        if a.get("shared_modalities"):
            lines.append(f"**Shared modalities:** {', '.join(a['shared_modalities'])}")
        if a.get("cohort_match"):
            lines.append(f"**Cohort match:** {a['cohort_match']}")
        if a.get("incompatibilities"):
            lines.append("**Incompatibilities:**")
            for x in a["incompatibilities"]:
                lines.append(f"- {x}")
        if a["verdict"] != "not_usable" and a.get("suggested_cross_dataset_hypotheses"):
            lines.append("**Suggested cross-dataset hypotheses (use as starting points if scientifically warranted):**")
            for h in a["suggested_cross_dataset_hypotheses"]:
                lines.append(f"- {h}")
        lines.append("")
    return "\n".join(lines)
