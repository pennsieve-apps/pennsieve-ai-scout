from pennsieve_ai_utils.blackboard import Blackboard, Claim

from scout.pipeline import _papers_for_rag


def test_papers_for_rag_prefers_open_access_papers_with_supported_claims():
    bb = Blackboard("q", "pennsieve:1")
    bb.papers_seen = [
        {"doi": "10.1/oa", "is_open_access": True, "title": "Open"},
        {"doi": "10.1/closed", "is_open_access": False, "title": "Closed"},
        {"doi": "10.1/unsupported", "is_open_access": True, "title": "Unsupported"},
    ]
    bb.literature_claims = [
        Claim("a", "doi:10.1/closed", "strong", entailment="supports"),
        Claim("b", "doi:10.1/closed", "strong", entailment="supports"),
        Claim("c", "doi:10.1/OA", "strong", entailment="supports"),
        Claim("d", "doi:10.1/unsupported", "strong", entailment="not_addressed"),
        Claim("e", "pmid:123", "strong", entailment="supports"),
    ]
    papers = _papers_for_rag(bb)
    assert [p["doi"] for p in papers] == ["10.1/oa", "10.1/closed"]
    assert papers[0]["open_access"] and papers[1]["supported_claims"] == 2
    assert len(_papers_for_rag(bb, limit=1)) == 1


def test_papers_for_rag_adds_uncited_open_access_papers_as_best_effort():
    bb = Blackboard("q", "pennsieve:1")
    bb.papers_seen = [
        {"doi": "10.1/cited", "is_open_access": True, "pmcid": "PMC1"},
        {"doi": "10.1/extra", "is_open_access": True, "pmcid": "PMC2", "title": "Extra"},
        {"doi": "10.1/no-pmc", "is_open_access": True},
        {"doi": "10.1/closed", "is_open_access": False, "pmcid": "PMC3"},
    ]
    bb.literature_claims = [Claim("a", "doi:10.1/cited", "strong", entailment="supports")]
    papers = _papers_for_rag(bb)
    assert [(p["doi"], p["required"]) for p in papers] == [("10.1/cited", True), ("10.1/extra", False)]
    assert [p["doi"] for p in _papers_for_rag(bb, limit=1)] == ["10.1/cited"]
    assert [p["required"] for p in _papers_for_rag(bb, required=0)] == [False, False]


def test_a_failed_compatibility_call_is_raised_not_hidden(monkeypatch):
    import pytest
    from scout.agents import dataset_compatibility
    from pennsieve_ai_utils.blackboard import DatasetSummary

    def boom(*args, **kwargs):
        raise ConnectionError("no address associated with hostname")

    monkeypatch.setattr(dataset_compatibility, "call_json", boom)
    bb = Blackboard("q", "pennsieve:1")
    summary = dict(title="t", description="", measurements=[], species="", n_subjects=0)
    bb.dataset_summaries = [DatasetSummary(id="pennsieve:1", **summary),
                            DatasetSummary(id="pennsieve:2", **summary)]
    with pytest.raises(ConnectionError):
        dataset_compatibility.run(bb, client=None)
