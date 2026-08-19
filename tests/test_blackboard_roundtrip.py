"""Blackboard JSON round-trip.

`load` reconstructs a saved blackboard for the downstream stages, so a field it
drops is a field the next stage never sees.
"""
from __future__ import annotations

from scout.blackboard import Blackboard


def test_model_assignments_survive_a_round_trip(tmp_path):
    """Regression: load()'s hardcoded field allowlist silently dropped this, so
    the model provenance never reached the downstream stages."""
    bb = Blackboard(research_question="q", subsystem="pennsieve:530")
    bb.model_assignments = {
        "reasoning": {"model_id": "us.anthropic.claude-sonnet-4-6", "rank": 246,
                      "source": "ranked", "at_floor": True, "note": ""},
    }
    path = tmp_path / "blackboard.json"
    bb.save(path)

    loaded = Blackboard.load(path)
    assert loaded.model_assignments["reasoning"]["model_id"] == "us.anthropic.claude-sonnet-4-6"
    assert loaded.model_assignments["reasoning"]["at_floor"] is True


def test_unknown_keys_in_the_json_are_ignored(tmp_path):
    """A blackboard written by a newer stage must still load here."""
    bb = Blackboard(research_question="q", subsystem="s")
    path = tmp_path / "blackboard.json"
    bb.save(path)
    path.write_text(path.read_text().replace('"subsystem": "s"',
                                             '"subsystem": "s", "invented_later": 1'))
    assert Blackboard.load(path).subsystem == "s"
