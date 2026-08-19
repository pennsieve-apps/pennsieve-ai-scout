"""Role resolution, capability ranking, and allow-list gating.

`scout.models` deliberately has no SDK dependency, so these run anywhere.
"""
from __future__ import annotations

import pytest

from scout import models as m


# Verbatim from the 403 that killed the 2026-08-04 run.
GOVERNOR_ALLOW_LIST = [
    "us.anthropic.claude-sonnet-4-6",
    "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
]

ERROR_BODY = {
    "error": "model_not_allowed",
    "message": "Model 'us.anthropic.claude-opus-4-6' is not in the allow-list",
    "allowedModels": GOVERNOR_ALLOW_LIST,
}


class FakeDenied(Exception):
    def __init__(self, body):
        super().__init__(f"Error code: 403 - {body!r}")
        self.body = body


@pytest.fixture(autouse=True)
def _no_pins(monkeypatch):
    """Clear operator pins so ranking is what's under test."""
    for role in list(m.ROLES) + list(m.LEGACY_ALIASES):
        monkeypatch.delenv(f"VP_MODEL_{role.upper()}", raising=False)
    monkeypatch.delenv("VP_MODEL_SELECT", raising=False)


# --- parsing and ranking ------------------------------------------------

@pytest.mark.parametrize(
    "model_id, expected",
    [
        ("us.anthropic.claude-opus-4-6", ("opus", (4, 6))),
        ("us.anthropic.claude-opus-5", ("opus", (5, 0))),
        ("us.anthropic.claude-sonnet-4-5-20250929-v1:0", ("sonnet", (4, 5))),
        # Version tokens precede the family in the 3.x naming scheme.
        ("us.anthropic.claude-3-5-haiku-20241022-v1:0", ("haiku", (3, 5))),
        ("anthropic.claude-fable-5", ("fable", (5, 0))),
    ],
)
def test_family_and_version_handles_both_token_orders(model_id, expected):
    assert m.family_and_version(model_id) == expected


def test_unknown_family_is_unrankable_not_zero():
    """A BYO model Scout can't reason about must never be auto-selected."""
    assert m.family_and_version("meta.llama3-70b-instruct-v1:0") is None
    assert m.rank("meta.llama3-70b-instruct-v1:0") is None


def test_rank_orders_family_first_then_version():
    r = m.rank
    assert r("us.anthropic.claude-opus-4-6") > r("us.anthropic.claude-sonnet-5")
    assert r("us.anthropic.claude-sonnet-5") > r("us.anthropic.claude-sonnet-4-6")
    assert r("us.anthropic.claude-sonnet-4-6") > r("us.anthropic.claude-sonnet-4-5")
    assert r("us.anthropic.claude-haiku-4-5") > r("us.anthropic.claude-3-5-haiku")
    assert r("us.anthropic.claude-sonnet-4-5") > r("us.anthropic.claude-haiku-4-5")


def test_family_key_makes_dated_and_undated_ids_equal():
    assert (m.family_key("us.anthropic.claude-haiku-4-5-20251001-v1:0")
            == m.family_key("us.anthropic.claude-haiku-4-5"))


# --- allow-list parsing -------------------------------------------------

def test_allow_list_from_error_body():
    assert m.parse_allowed_models(FakeDenied(ERROR_BODY)) == GOVERNOR_ALLOW_LIST


def test_allow_list_from_stringified_exception():
    """The body isn't always parsed — the repr still carries the list."""
    exc = Exception(f"Error code: 403 - {ERROR_BODY!r}")
    assert m.parse_allowed_models(exc) == GOVERNOR_ALLOW_LIST


def test_no_allow_list_returns_none():
    assert m.parse_allowed_models(Exception("Error code: 500")) is None


def test_is_model_not_allowed_discriminates():
    assert m.is_model_not_allowed(FakeDenied(ERROR_BODY))
    assert not m.is_model_not_allowed(FakeDenied({"error": "budget_exceeded"}))


# --- rank-based selection ----------------------------------------------

def test_roles_resolve_against_the_real_allow_list():
    """The 2026-08-04 allow-list: no Opus at all. Reasoning and synthesis both
    land on the strongest permitted model within their band; bulk stays on
    cheap hardware because of its ceiling."""
    r = m.ModelResolver(GOVERNOR_ALLOW_LIST)
    assert r.resolve("reasoning") == "us.anthropic.claude-sonnet-4-6"
    assert r.resolve("synthesis") == "us.anthropic.claude-sonnet-4-6"
    assert r.resolve("bulk") == "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def test_ceiling_keeps_bulk_off_expensive_models():
    """Bulk runs under a 512-token budget; it must not be handed a big model
    just because one is permitted."""
    r = m.ModelResolver(["us.anthropic.claude-opus-4-8",
                         "us.anthropic.claude-haiku-4-5-20251001-v1:0"])
    assert r.resolve("reasoning") == "us.anthropic.claude-opus-4-8"
    assert r.resolve("bulk") == "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def test_ceiling_yields_rather_than_failing_when_nothing_is_cheap():
    """Overspending beats not running: take the cheapest above the ceiling."""
    r = m.ModelResolver(["us.anthropic.claude-opus-4-8", "us.anthropic.claude-sonnet-4-6"])
    assert r.resolve("bulk") == "us.anthropic.claude-sonnet-4-6"


def test_reasoning_picks_the_strongest_permitted():
    r = m.ModelResolver(["us.anthropic.claude-sonnet-4-6",
                         "us.anthropic.claude-opus-4-6",
                         "us.anthropic.claude-opus-4-8"])
    assert r.resolve("reasoning") == "us.anthropic.claude-opus-4-8"


def test_unrankable_models_are_skipped_by_auto_selection():
    r = m.ModelResolver(["meta.llama3-70b-instruct-v1:0", "us.anthropic.claude-sonnet-4-6"])
    assert r.resolve("reasoning") == "us.anthropic.claude-sonnet-4-6"


def test_optimistic_pick_before_discovery():
    r = m.ModelResolver()
    assert r.resolve("reasoning") == m.ROLES["reasoning"].optimistic


def test_learning_invalidates_a_stale_assignment():
    r = m.ModelResolver()
    assert r.resolve("reasoning") == m.ROLES["reasoning"].optimistic  # optimistic
    r.learn(GOVERNOR_ALLOW_LIST)
    assert r.resolve("reasoning") == "us.anthropic.claude-sonnet-4-6"


def test_learning_keeps_an_assignment_that_is_still_permitted():
    r = m.ModelResolver()
    first = r.resolve("bulk")
    r.learn(GOVERNOR_ALLOW_LIST)
    assert r.resolve("bulk") == first


# --- gating -------------------------------------------------------------

def test_floor_not_met_aborts_with_an_actionable_message():
    """Only Haiku permitted — a reasoning-critical role must refuse rather
    than produce output nobody should trust."""
    r = m.ModelResolver(["us.anthropic.claude-3-5-haiku-20241022-v1:0"])
    with pytest.raises(m.ModelNotAvailable) as excinfo:
        r.resolve("reasoning")
    msg = str(excinfo.value)
    assert "VP_MODEL_REASONING" in msg      # how to override deliberately
    assert "SSM allow-list" in msg          # who to ask
    assert m.ROLES["reasoning"].rationale.split(".")[0] in msg  # why there's a floor


def test_resolve_all_fails_before_any_work():
    r = m.ModelResolver(["us.anthropic.claude-3-5-haiku-20241022-v1:0"])
    with pytest.raises(m.ModelNotAvailable):
        r.resolve_all()


def test_bulk_still_runs_when_only_weak_models_are_permitted():
    """Cheap roles degrade quietly — that's the point of a per-role floor."""
    r = m.ModelResolver(["us.anthropic.claude-3-5-haiku-20241022-v1:0"])
    assert r.resolve("bulk") == "us.anthropic.claude-3-5-haiku-20241022-v1:0"


# --- pins and legacy aliases -------------------------------------------

def test_legacy_vendor_tier_names_still_resolve():
    """Agent modules and downstream repos may still say 'opus'."""
    r = m.ModelResolver(GOVERNOR_ALLOW_LIST)
    assert r.resolve("opus") == r.resolve("reasoning")
    assert r.resolve("sonnet") == r.resolve("synthesis")
    assert r.resolve("haiku") == r.resolve("bulk")


def test_unknown_role_name_is_an_error():
    with pytest.raises(RuntimeError, match="Unknown model role"):
        m.canonical_role("gpt")


def test_legacy_env_pin_is_honored(monkeypatch):
    """A node already unblocked with VP_MODEL_OPUS must keep working."""
    monkeypatch.setenv("VP_MODEL_OPUS", "us.anthropic.claude-sonnet-4-6")
    r = m.ModelResolver(GOVERNOR_ALLOW_LIST)
    assert r.resolve("reasoning") == "us.anthropic.claude-sonnet-4-6"


def test_role_env_pin_wins_over_legacy(monkeypatch):
    monkeypatch.setenv("VP_MODEL_OPUS", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    monkeypatch.setenv("VP_MODEL_REASONING", "us.anthropic.claude-opus-4-8")
    r = m.ModelResolver(GOVERNOR_ALLOW_LIST)
    assert r.resolve("reasoning") == "us.anthropic.claude-opus-4-8"


def test_pin_below_the_floor_is_honored_but_flagged(monkeypatch):
    """Explicit operator intent overrides the floor — never silently."""
    monkeypatch.setenv("VP_MODEL_REASONING", "us.anthropic.claude-3-5-haiku-20241022-v1:0")
    r = m.ModelResolver(GOVERNOR_ALLOW_LIST)
    assert r.resolve("reasoning") == "us.anthropic.claude-3-5-haiku-20241022-v1:0"
    assignment = r.assignments()["reasoning"]
    assert assignment["source"] == "pinned"
    assert "below the role floor" in assignment["note"]


def test_pin_to_an_unrankable_byo_model_is_allowed(monkeypatch):
    """BYO path: capability can't be verified, so say so and proceed."""
    monkeypatch.setenv("VP_MODEL_SYNTHESIS", "mycorp.custom-finetune-v2")
    r = m.ModelResolver(GOVERNOR_ALLOW_LIST)
    assert r.resolve("synthesis") == "mycorp.custom-finetune-v2"
    assert "cannot" in r.assignments()["synthesis"]["note"]


# --- provenance and preflight ------------------------------------------

def test_assignments_record_provenance():
    r = m.ModelResolver(GOVERNOR_ALLOW_LIST)
    r.resolve_all()
    reasoning = r.assignments()["reasoning"]
    assert reasoning["model_id"] == "us.anthropic.claude-sonnet-4-6"
    assert reasoning["source"] == "ranked"
    assert reasoning["at_floor"] is True   # this allow-list has nothing better


def test_discover_reads_the_allow_list_from_the_probe_rejection():
    def probe(model_id):
        assert model_id == m.PROBE_MODEL
        raise FakeDenied(ERROR_BODY)
    assert m.discover_allowed_models(probe) == GOVERNOR_ALLOW_LIST


def test_discover_returns_none_when_the_backend_does_not_gate():
    """A direct Anthropic key accepts anything — no allow-list to report."""
    assert m.discover_allowed_models(lambda model_id: None) is None


def test_discover_returns_none_on_an_unrelated_error():
    def probe(model_id):
        raise Exception("Error code: 500 - internal")
    assert m.discover_allowed_models(probe) is None


def test_manifest_names_every_role_and_the_allow_list():
    r = m.ModelResolver(GOVERNOR_ALLOW_LIST)
    text = m.render_manifest(r, r.resolve_all())
    for role in m.ROLES:
        assert role in text
    assert "us.anthropic.claude-sonnet-4-6" in text
    assert "at floor" in text


# --- access_denied: permitted but not invokable -------------------------

ACCESS_DENIED_BODY = {
    "error": "access_denied",
    "message": ("Access denied for model "
                "'us.anthropic.claude-haiku-4-5-20251001-v1:0'. This is likely "
                "an IAM permission issue on the governor Lambda role."),
}


def test_access_denied_is_distinguished_from_model_not_allowed():
    denied = FakeDenied(ACCESS_DENIED_BODY)
    assert m.is_access_denied(denied)
    assert not m.is_model_not_allowed(denied)

    not_allowed = FakeDenied(ERROR_BODY)
    assert m.is_model_not_allowed(not_allowed)
    assert not m.is_access_denied(not_allowed)


def test_access_denied_carries_no_allow_list():
    """Which is why it needs its own handling — there's nothing to learn."""
    assert m.parse_allowed_models(FakeDenied(ACCESS_DENIED_BODY)) is None


def test_unusable_model_is_skipped_and_role_re_resolves():
    """The 2026-08-18 run: Haiku 4.5 is permitted but the governor's IAM role
    can't invoke it, so bulk must fall to the next permitted model."""
    r = m.ModelResolver(GOVERNOR_ALLOW_LIST)
    assert r.resolve("bulk") == "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    r.mark_unusable("us.anthropic.claude-haiku-4-5-20251001-v1:0")
    assert r.resolve("bulk") == "us.anthropic.claude-3-5-haiku-20241022-v1:0"


def test_unusable_matches_across_dated_spellings():
    r = m.ModelResolver(GOVERNOR_ALLOW_LIST)
    r.mark_unusable("us.anthropic.claude-haiku-4-5")   # undated
    assert r.resolve("bulk") == "us.anthropic.claude-3-5-haiku-20241022-v1:0"


def test_all_usable_models_exhausted_aborts_with_iam_hint():
    r = m.ModelResolver(GOVERNOR_ALLOW_LIST)
    for model_id in GOVERNOR_ALLOW_LIST:
        r.mark_unusable(model_id)
    with pytest.raises(m.ModelNotAvailable) as excinfo:
        r.resolve("bulk")
    msg = str(excinfo.value)
    assert "not invokable" in msg and "IAM" in msg


def test_unusable_optimistic_pick_without_an_allow_list_fails_clearly():
    r = m.ModelResolver()
    r.mark_unusable(m.ROLES["reasoning"].optimistic)
    with pytest.raises(m.ModelNotAvailable, match="VP_MODEL_REASONING"):
        r.resolve("reasoning")


def test_marking_unusable_leaves_other_roles_alone():
    r = m.ModelResolver(GOVERNOR_ALLOW_LIST)
    r.resolve_all()
    reasoning_before = r.assignments()["reasoning"]["model_id"]
    r.mark_unusable("us.anthropic.claude-haiku-4-5-20251001-v1:0")
    assert r.resolve("reasoning") == reasoning_before


def test_note_distinguishes_unusable_from_unlisted():
    """"not on the allow-list" is wrong for a model that is listed but blocked."""
    r = m.ModelResolver(GOVERNOR_ALLOW_LIST)
    r.mark_unusable(m.ROLES["bulk"].optimistic)
    r.resolve("bulk")
    note = r.assignments()["bulk"]["note"]
    assert "permitted but not invokable" in note
    assert "not on the allow-list" not in note


def test_manifest_flags_unusable_models():
    r = m.ModelResolver(GOVERNOR_ALLOW_LIST)
    r.mark_unusable("us.anthropic.claude-haiku-4-5-20251001-v1:0")
    text = m.render_manifest(r, r.resolve_all())
    assert "[not invokable — IAM]" in text
