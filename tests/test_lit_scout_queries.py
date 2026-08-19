"""Query assembly in lit_scout — the 2026-08-19 run found 0 papers because
every query ran twice (base + director were the same list) and the framer's
8-12 word natural-language queries matched nothing under Europe PMC's
AND-all-terms semantics."""
from __future__ import annotations

from scout.agents.lit_scout import _broaden, _dedupe_queries, _fallback_queries
from scout.blackboard import Blackboard, DatasetSummary


def test_dedupe_removes_the_doubled_director_queries():
    """pipeline copies suggested_queries onto search_queries, so base+director
    used to be the same eight strings twice."""
    qs = ["seizure onset zone detection"] * 2 + ["  Seizure Onset Zone detection "]
    assert _dedupe_queries(qs) == ["seizure onset zone detection"]


def test_dedupe_preserves_order():
    assert _dedupe_queries(["b query", "a query", "b query"]) == ["b query", "a query"]


def test_broaden_drops_the_qualifier_tail():
    # Verbatim from the failed run: 11 ANDed terms → 0 hits.
    q = ("intracranial EEG seizure detection benchmark automated algorithm "
         "human performance comparison")
    wide = _broaden(q)
    assert wide == "intracranial EEG seizure detection benchmark"


def test_broaden_skips_stopwords():
    q = "seizure spread in the 10 seconds after onset of stimulation"
    assert "in" not in _broaden(q).split()
    assert "the" not in _broaden(q).split()


def test_broaden_returns_empty_when_already_short():
    """A short query that found nothing won't be helped by re-running it."""
    assert _broaden("seizure detection iEEG") == ""


def test_fallback_queries_come_from_the_seed_dataset():
    bb = Blackboard(research_question="q", subsystem="pennsieve:530")
    bb.dataset_summaries = [DatasetSummary(
        id="pennsieve:530", title="Multi-Expert Seizure Annotation",
        description="", measurements=[], species="human", n_subjects=1,
        extracted_topics=["seizure-onset-zone", "intracranial-eeg"],
    )]
    fallbacks = _fallback_queries(bb)
    assert fallbacks[0] == "Multi-Expert Seizure Annotation"
    assert "seizure onset zone" in fallbacks[1]


def test_fallback_queries_empty_without_a_seed():
    assert _fallback_queries(Blackboard(research_question="q", subsystem="s")) == []
