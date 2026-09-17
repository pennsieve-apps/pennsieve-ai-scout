"""Degraded mode — completing a run past failed reasoning stages.

Off by default: the role floors exist so a weak run fails instead of emitting
output nobody should trust. VP_DEGRADED_OK is the explicit opt-out, for when
scout's own quality isn't what's under test (e.g. tracing downstream stages).
"""
from __future__ import annotations

import pytest

from pennsieve_ai_utils.blackboard import Blackboard
from pennsieve_ai_utils.models import degraded_ok
from scout.pipeline import _stage


@pytest.fixture
def bb():
    return Blackboard(research_question="q", subsystem="pennsieve:530")


@pytest.fixture(autouse=True)
def _no_flag(monkeypatch):
    monkeypatch.delenv("VP_DEGRADED_OK", raising=False)


def _boom():
    raise RuntimeError("governor timed out")


def test_flag_defaults_off(monkeypatch):
    assert degraded_ok() is False
    for value in ("1", "true", "yes", "YES"):
        monkeypatch.setenv("VP_DEGRADED_OK", value)
        assert degraded_ok() is True
    monkeypatch.setenv("VP_DEGRADED_OK", "0")
    assert degraded_ok() is False


def test_required_stage_failure_is_fatal_by_default(bb):
    with pytest.raises(RuntimeError, match="governor timed out"):
        _stage(bb, "dataset_framer", _boom)
    assert bb.degraded == []


def test_required_stage_is_skipped_and_recorded_when_allowed(bb, monkeypatch):
    monkeypatch.setenv("VP_DEGRADED_OK", "1")
    assert _stage(bb, "dataset_framer", _boom) is False
    assert [d["stage"] for d in bb.degraded] == ["dataset_framer"]
    assert "governor timed out" in bb.degraded[0]["error"]


def test_optional_stage_never_aborts_but_is_still_recorded(bb):
    """dataset_compatibility has always been non-fatal — but a partial
    blackboard must say what's missing rather than look complete."""
    assert _stage(bb, "dataset_compatibility", _boom, required=False) is False
    assert [d["stage"] for d in bb.degraded] == ["dataset_compatibility"]


def test_successful_stage_records_nothing(bb):
    calls = []
    assert _stage(bb, "verifier", lambda: calls.append(1)) is True
    assert calls == [1] and bb.degraded == []


def test_stage_passes_through_args(bb):
    seen = {}
    _stage(bb, "s", lambda a, b=None: seen.update(a=a, b=b), 1, b=2)
    assert seen == {"a": 1, "b": 2}


def test_degraded_survives_a_blackboard_round_trip(bb, tmp_path, monkeypatch):
    """Downstream loads this JSON — the incompleteness marker has to survive."""
    monkeypatch.setenv("VP_DEGRADED_OK", "1")
    _stage(bb, "lit_scout", _boom)
    path = tmp_path / "blackboard.json"
    bb.save(path)
    assert [d["stage"] for d in Blackboard.load(path).degraded] == ["lit_scout"]
