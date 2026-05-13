from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class Claim:
    text: str
    source: str
    evidence_strength: str  # strong | moderate | weak
    quote: str = ""  # verbatim text from the source abstract that supports the claim
    quote_match: str = "not_checked"  # exact | partial | missing | not_checked
    entailment: str = "not_checked"  # supports | contradicts | not_addressed | not_checked
    verification_note: str = ""  # human-readable note from the Verifier


@dataclass
class DatasetSummary:
    id: str
    title: str
    description: str
    measurements: list[str]
    species: str
    n_subjects: int
    repo: str = ""  # pennsieve | dandi | manual
    url: str = ""
    doi: str = ""
    # Optional fields populated by repo-specific inspection (Pennsieve currently)
    readme: str = ""  # full README text written by the depositor
    file_types: dict = field(default_factory=dict)  # {fileType: count} histogram from manifest
    file_tree: str = ""  # compact rendered tree of all paths (from manifest.json)
    record_tables: dict = field(default_factory=dict)  # {filename: {columns, n_rows, head}}
    external_publications: list[dict] = field(default_factory=list)  # linked papers (DOI + relationship)
    bucket_uri: str = ""  # s3://bucket/prefix/ — primary key for direct downloads
    access_type: str = ""  # "oad" (anonymous HTTPS works) or "requester_pays" (auth + RequestPayer)
    extracted_topics: list[str] = field(default_factory=list)  # LLM-derived topical tags from README/title — used for thematic RAG filtering. Specific (e.g. pudendal-nerve) rather than generic depositor tags (e.g. human, imaging)
    license: str = ""
    size_bytes: int = 0
    file_count: int = 0
    relevance_note: str = ""  # Data Scout's reason for selecting this dataset


@dataclass
class DirectorBrief:
    """Strategic frame produced by the Director agent. Threaded through every
    downstream agent so all reasoning is anchored in the same lab strategy."""
    field_landscape: str  # markdown synthesis: established mechanisms, debates, gaps
    research_strategy: str  # 1-2 paragraph framing
    focus_areas: list[str]  # 3-5 specific angles to prioritize
    lab_constraints: list[str]  # what to avoid
    acceptance_criteria: list[str]  # what makes a hypothesis worth pursuing for THIS lab
    suggested_queries: list[str]  # extra Europe PMC queries for Lit Scout's targeted pass
    review_papers: list[dict] = field(default_factory=list)  # provenance: papers Director read


@dataclass
class Hypothesis:
    id: str
    claim: str
    rationale: str
    supporting_evidence: list[int]  # indices into literature_claims
    predicted_observation: str
    suggested_experiment: str
    iteration: int = 1
    revises: str | None = None  # ID of the prior-iteration hypothesis this one revises/replaces
    evidence_chunks: list[dict] = field(default_factory=list)  # per-hypothesis RAG retrieval (chunk_id, doi, section, text, score)
    lay_summary: str = ""  # plain-language summary for non-specialists; populated by lay_translator after the closed loop


@dataclass
class CriticReview:
    hypothesis_id: str
    novelty: int  # 1-5
    plausibility: int  # 1-5
    feasibility: int  # 1-5
    alignment: int  # 1-5 — fit to Director's strategy/lab profile
    attacks: list[str]
    verdict: str  # accept | revise | reject
    iteration: int = 1


@dataclass
class Blackboard:
    research_question: str
    subsystem: str
    run_id: str = ""
    search_query: str = ""  # legacy single-query field, still honored
    search_queries: list[str] = field(default_factory=list)  # preferred: multi-query union
    max_iterations: int = 3
    iterations_completed: int = 0
    stop_reason: str = ""
    lab_profile: dict = field(default_factory=dict)
    director_brief: DirectorBrief | None = None
    papers_seen: list[dict] = field(default_factory=list)  # full abstracts the Lit Scout pulled — used by the Verifier
    literature_claims: list[Claim] = field(default_factory=list)
    deep_evidence: list[dict] = field(default_factory=list)  # RAG-retrieved chunks (chunk_id, doi, section, text, score)
    deep_read_stats: dict = field(default_factory=dict)  # Deep Reader summary for the report
    collaboration_candidates: list[dict] = field(default_factory=list)  # constraint-blocked hypotheses with suggested collaborators
    dataset_summaries: list[DatasetSummary] = field(default_factory=list)
    hypotheses: list[Hypothesis] = field(default_factory=list)
    critic_reviews: list[CriticReview] = field(default_factory=list)
    papers_enriched: dict = field(default_factory=dict)  # {doi: {title, authors, year, journal}} from CrossRef — authoritative source for inline citations
    prior_papers: list[dict] = field(default_factory=list)  # papers (linked from dataset + user --prior-paper) treated as off-limits prior art for hypothesis novelty: [{doi, title, abstract, authors, year, source_label}]
    cross_dataset_compatibility: list[dict] = field(default_factory=list)  # per-candidate compatibility verdict from the Dataset Compatibility Analyst: [{dataset_id, verdict, shared_modalities, cohort_match, incompatibilities, suggested_cross_dataset_hypotheses, rationale}]. Empty when only the seed exists.
    notebook: list[dict] = field(default_factory=list)

    def log(self, agent: str, action: str, content: str = "") -> None:
        entry = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "agent": agent,
            "action": action,
            "content": content,
        }
        self.notebook.append(entry)
        suffix = f" — {content}" if content else ""
        print(f"[{entry['ts']}] {agent}: {action}{suffix}")

    def hypotheses_at(self, iteration: int) -> list[Hypothesis]:
        return [h for h in self.hypotheses if h.iteration == iteration]

    def reviews_at(self, iteration: int) -> list[CriticReview]:
        return [r for r in self.critic_reviews if r.iteration == iteration]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, default=str))

    @classmethod
    def load(cls, path: Path) -> "Blackboard":
        """Reconstruct a blackboard from a saved JSON snapshot. Used by the
        Analyst (and any future post-hoc tool) that needs to read the state
        of a completed run without re-running the pipeline."""
        raw = json.loads(Path(path).read_text())

        def _to(cls_, d):
            if d is None:
                return None
            return cls_(**{k: v for k, v in d.items() if k in cls_.__dataclass_fields__})

        bb = cls(
            research_question=raw.get("research_question", ""),
            subsystem=raw.get("subsystem", ""),
        )
        for k, v in raw.items():
            if k in ("papers_seen", "deep_evidence", "deep_read_stats",
                     "collaboration_candidates", "notebook", "lab_profile",
                     "search_queries", "research_question", "subsystem",
                     "run_id", "search_query", "max_iterations",
                     "iterations_completed", "stop_reason", "papers_enriched",
                     "prior_papers", "cross_dataset_compatibility"):
                setattr(bb, k, v)
        bb.literature_claims = [_to(Claim, x) for x in raw.get("literature_claims") or []]
        bb.dataset_summaries = [_to(DatasetSummary, x) for x in raw.get("dataset_summaries") or []]
        bb.hypotheses = [_to(Hypothesis, x) for x in raw.get("hypotheses") or []]
        bb.critic_reviews = [_to(CriticReview, x) for x in raw.get("critic_reviews") or []]
        if raw.get("director_brief"):
            bb.director_brief = _to(DirectorBrief, raw["director_brief"])
        return bb
