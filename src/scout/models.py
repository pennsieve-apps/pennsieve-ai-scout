"""Model roles, capability ranking, and allow-list resolution.

Scout addresses models by the *role they play in the pipeline* —
`reasoning`, `synthesis`, `bulk` — never by vendor family. Which concrete
model serves a role is configuration, resolved at runtime against whatever
the serving backend actually permits.

Three concerns live here:

  * **Ranking.** `rank()` scores an arbitrary model ID from its parsed
    family and version, so a permitted model Scout has never seen can still
    be ordered against ones it has. This is what lets an operator bring
    their own Bedrock-workspace models without editing a hardcoded chain.

  * **Discovery.** The governor reports its SSM allow-list in the body of a
    `model_not_allowed` 403. `parse_allowed_models` reads it, and preflight
    deliberately trips that 403 with a sentinel ID so the full menu is known
    *before* any real work starts — Bedrock has no Models API to ask
    instead. If the probe comes back in an unexpected shape we fall back to
    learning lazily from the first real 403.

  * **Gating.** Each role declares a floor and, for cheap roles, a ceiling.
    Bulk work degrades quietly and is kept off expensive models; a
    reasoning-critical role that cannot be served adequately aborts at
    preflight instead of spending a whole pipeline run producing output
    nobody should trust.
"""
from __future__ import annotations

import os
import re
import sys
import threading
from dataclasses import dataclass, field

# --------------------------------------------------------------- ranking

# Family → base score. A version contributes at most 99 on top, so keeping
# the bases 100 apart means family dominates and version breaks ties inside
# a family.
_FAMILY_BASE: dict[str, int] = {
    "haiku": 100,
    "sonnet": 200,
    "opus": 300,
    "fable": 400,
    "mythos": 400,
}

# `-20250929`, optionally followed by a Bedrock `-v1:0` revision.
_DATE_SUFFIX = re.compile(r"-\d{8}(?:-v\d+:\d+)?$")
_NUMERIC = re.compile(r"^\d{1,2}$")


def family_key(model_id: str) -> str:
    """Date-insensitive identity for a model ID.

    Lets a dated profile ID and its undated twin compare equal, since the
    governor may list either spelling.
    """
    return _DATE_SUFFIX.sub("", model_id.strip())


def family_and_version(model_id: str) -> tuple[str, tuple[int, int]] | None:
    """Parse a model ID into `(family, (major, minor))`.

    Both token orders in circulation are handled — `claude-opus-4-6` (version
    after the family) and `claude-3-5-haiku` (version before it) — by taking
    the first two standalone numeric tokens wherever they land. Returns None
    when the family is not one Scout knows how to rank.
    """
    tokens = family_key(model_id).lower().replace(".", "-").split("-")
    family = next((t for t in tokens if t in _FAMILY_BASE), None)
    if family is None:
        return None
    nums = [int(t) for t in tokens if _NUMERIC.match(t)]
    major = nums[0] if nums else 0
    minor = nums[1] if len(nums) > 1 else 0
    return family, (major, minor)


def rank(model_id: str) -> int | None:
    """Capability score for `model_id`, or None if its family is unknown.

    Unknown families are deliberately unrankable rather than scored zero: an
    operator can still use one by pinning it explicitly, but automatic
    selection never silently reaches for a model whose capability Scout
    cannot reason about.
    """
    parsed = family_and_version(model_id)
    if parsed is None:
        return None
    family, (major, minor) = parsed
    return _FAMILY_BASE[family] + major * 10 + minor


def describe(model_id: str) -> str:
    """`model_id` plus its rank, for manifests and error messages."""
    r = rank(model_id)
    return f"{model_id} (rank {r})" if r is not None else f"{model_id} (unrankable)"


# ----------------------------------------------------------------- roles


@dataclass(frozen=True)
class Role:
    """A capability requirement of the pipeline, not a model.

    `optimistic` is what we ask for before the allow-list is known — one ID,
    not a chain; ranking handles the rest. `floor` is the weakest model whose
    output is worth the compute the rest of the pipeline spends on it.
    `ceiling` caps automatic selection so high-volume roles don't quietly
    land on an expensive model.
    """
    name: str
    optimistic: str
    floor: str
    rationale: str
    ceiling: str | None = None

    @property
    def floor_rank(self) -> int:
        r = rank(self.floor)
        assert r is not None, f"role {self.name!r}: floor must be rankable"
        return r

    @property
    def ceiling_rank(self) -> int | None:
        return rank(self.ceiling) if self.ceiling else None


ROLES: dict[str, Role] = {
    "reasoning": Role(
        name="reasoning",
        optimistic="us.anthropic.claude-opus-4-8",
        floor="us.anthropic.claude-sonnet-4-6",
        rationale=(
            "Frames the research question every downstream stage builds on. "
            "Below the floor the brief is not worth the compute the later "
            "stages spend acting on it."
        ),
    ),
    "synthesis": Role(
        name="synthesis",
        optimistic="us.anthropic.claude-sonnet-4-6",
        floor="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        ceiling="us.anthropic.claude-opus-4-6",
        rationale=(
            "Grounds claims against the sources actually seen. A weaker "
            "model yields verdicts that read confident and are not."
        ),
    ),
    "bulk": Role(
        name="bulk",
        optimistic="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        floor="us.anthropic.claude-3-5-haiku-20241022-v1:0",
        ceiling="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        rationale=(
            "Per-paper tagging and query shaping — high volume, individually "
            "low stakes, and run under a small token budget."
        ),
    ),
}

# Vendor-family tier names Scout used before roles existed. Kept so existing
# agent modules and any `VP_MODEL_OPUS`-style pin already set on a deployed
# compute node keep working unchanged.
LEGACY_ALIASES: dict[str, str] = {
    "opus": "reasoning",
    "sonnet": "synthesis",
    "haiku": "bulk",
}


def canonical_role(name: str) -> str:
    """Resolve a role name or a legacy vendor-tier alias to a role name."""
    if name in ROLES:
        return name
    if name in LEGACY_ALIASES:
        return LEGACY_ALIASES[name]
    raise RuntimeError(
        f"Unknown model role {name!r}. Valid roles: {', '.join(ROLES)} "
        f"(legacy aliases: {', '.join(LEGACY_ALIASES)})."
    )


def pinned_model(role: str) -> str | None:
    """Operator pin for `role`, from `VP_MODEL_<ROLE>`.

    The legacy vendor-family variable is honored as a fallback so a node
    already running with `VP_MODEL_OPUS` set keeps its override.
    """
    names = [f"VP_MODEL_{role.upper()}"]
    names += [f"VP_MODEL_{legacy.upper()}"
              for legacy, mapped in LEGACY_ALIASES.items() if mapped == role]
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


# ------------------------------------------------------------- allow-list


def parse_allowed_models(exc: Exception) -> list[str] | None:
    """Pull `allowedModels` out of a governor `model_not_allowed` 403.

    The governor's body is its own shape rather than Anthropic's standard
    error envelope, so read it defensively: parsed body, then the raw
    response, then the stringified exception as a last resort.
    """
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        try:
            body = exc.response.json()  # type: ignore[attr-defined]
        except Exception:
            body = None

    if isinstance(body, dict):
        models = body.get("allowedModels")
        if isinstance(models, list) and models:
            return [str(m) for m in models]

    match = re.search(r"['\"]allowedModels['\"]\s*:\s*\[(.*?)\]", str(exc), re.S)
    if match:
        found = re.findall(r"['\"]([^'\"]+)['\"]", match.group(1))
        if found:
            return found
    return None


def is_model_not_allowed(exc: Exception) -> bool:
    body = getattr(exc, "body", None)
    if isinstance(body, dict) and body.get("error") == "model_not_allowed":
        return True
    return "model_not_allowed" in str(exc)


def is_access_denied(exc: Exception) -> bool:
    """True for a governor `access_denied` 403.

    Distinct from `model_not_allowed`: the model *is* on the allow-list, but
    the governor's Lambda role can't invoke it (missing bedrock:InvokeModel).
    Permitted and usable are not the same thing, and this 403 carries no
    `allowedModels` — so it has to be handled on its own terms.
    """
    body = getattr(exc, "body", None)
    if isinstance(body, dict) and body.get("error") == "access_denied":
        return True
    return "access_denied" in str(exc)


class ModelNotAvailable(RuntimeError):
    """No permitted model clears a role's floor — raised at preflight."""


# -------------------------------------------------------------- resolver


@dataclass
class Assignment:
    """What ended up serving a role, and how that was decided."""
    role: str
    model_id: str
    rank: int | None
    source: str          # pinned | optimistic | ranked
    at_floor: bool = False
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "role": self.role,
            "model_id": self.model_id,
            "rank": self.rank,
            "source": self.source,
            "at_floor": self.at_floor,
            "note": self.note,
        }


class ModelResolver:
    """Maps a pipeline role to a concrete model ID.

    Selection is rank-based rather than a hardcoded preference chain: given
    the allow-list, each role takes the highest-ranked permitted model that
    sits within its floor/ceiling band. Shared across pipeline threads.
    """

    def __init__(self, allowed: list[str] | None = None) -> None:
        self._lock = threading.Lock()
        self._allowed: list[str] | None = list(allowed) if allowed else None
        self._assigned: dict[str, Assignment] = {}
        self._announced: set[str] = set()
        # Family keys of models the backend permits but cannot actually
        # invoke. Learned from access_denied, never from the allow-list.
        self._unusable: set[str] = set()

    # -- state ----------------------------------------------------------

    @property
    def allowed(self) -> list[str] | None:
        return list(self._allowed) if self._allowed is not None else None

    def learn(self, allowed: list[str]) -> None:
        """Record the allow-list, discarding assignments it invalidates."""
        with self._lock:
            self._allowed = list(allowed)
            permitted = {family_key(m) for m in allowed}
            for role, assignment in list(self._assigned.items()):
                if family_key(assignment.model_id) not in permitted:
                    del self._assigned[role]

    def mark_unusable(self, model_id: str) -> None:
        """Record that `model_id` is permitted but not invokable, and drop any
        assignment relying on it so the role re-resolves onto something else."""
        with self._lock:
            self._unusable.add(family_key(model_id))
            for role, assignment in list(self._assigned.items()):
                if family_key(assignment.model_id) == family_key(model_id):
                    del self._assigned[role]

    @property
    def unusable(self) -> list[str]:
        with self._lock:
            return sorted(self._unusable)

    def assignments(self) -> dict[str, dict]:
        """Provenance for the blackboard: role → how it was served."""
        with self._lock:
            return {r: a.as_dict() for r, a in sorted(self._assigned.items())}

    # -- resolution -----------------------------------------------------

    def resolve(self, role_name: str) -> str:
        """Role (or legacy alias) → the model ID to send right now."""
        role = ROLES[canonical_role(role_name)]
        with self._lock:
            existing = self._assigned.get(role.name)
            if existing:
                return existing.model_id
            assignment = self._select(role)
            self._assigned[role.name] = assignment
            if assignment.note and role.name not in self._announced:
                self._announced.add(role.name)
                print(f"  [llm] ! {assignment.note}", flush=True)
            return assignment.model_id

    def resolve_all(self) -> dict[str, Assignment]:
        """Assign every role. Raises `ModelNotAvailable` on the first role
        that cannot be served, so preflight fails before any real work."""
        for name in ROLES:
            self.resolve(name)
        with self._lock:
            return dict(self._assigned)

    def _select(self, role: Role) -> Assignment:
        """Caller holds the lock."""
        pin = pinned_model(role.name)
        if pin:
            return self._pinned_assignment(role, pin)

        if self._allowed is None:
            # Nothing discovered yet — ask for the role's first choice and
            # let a 403 teach us the rest.
            if family_key(role.optimistic) in self._unusable:
                raise ModelNotAvailable(
                    f"{role.optimistic} is the first choice for the "
                    f"{role.name!r} role but the backend cannot invoke it, and "
                    f"no allow-list has been reported to choose an alternative "
                    f"from. Set VP_MODEL_{role.name.upper()} to a model the "
                    f"backend can reach."
                )
            return Assignment(
                role=role.name,
                model_id=role.optimistic,
                rank=rank(role.optimistic),
                source="optimistic",
            )

        return self._ranked_assignment(role)

    def _pinned_assignment(self, role: Role, pin: str) -> Assignment:
        """An explicit pin is operator intent: honored even below the floor,
        but never silently."""
        pin_rank = rank(pin)
        note = ""
        if pin_rank is None:
            note = (
                f"{role.name!r} pinned to {pin}, whose capability Scout cannot "
                f"rank. Proceeding on operator intent; output quality for this "
                f"role is unverified."
            )
        elif pin_rank < role.floor_rank:
            note = (
                f"{role.name!r} pinned to {describe(pin)}, below the role floor "
                f"{describe(role.floor)}. {role.rationale}"
            )
        return Assignment(
            role=role.name,
            model_id=pin,
            rank=pin_rank,
            source="pinned",
            at_floor=pin_rank is not None and pin_rank == role.floor_rank,
            note=note,
        )

    def _ranked_assignment(self, role: Role) -> Assignment:
        """Highest-ranked permitted model inside the role's band."""
        assert self._allowed is not None
        ranked = sorted(
            ((m, rank(m)) for m in self._allowed
             if family_key(m) not in self._unusable),
            key=lambda pair: pair[1] or -1,
            reverse=True,
        )
        eligible = [(m, r) for m, r in ranked if r is not None and r >= role.floor_rank]
        if not eligible:
            raise ModelNotAvailable(self._unavailable_message(role, ranked))

        ceiling = role.ceiling_rank
        within = [(m, r) for m, r in eligible if ceiling is None or r <= ceiling]
        # Prefer the strongest inside the band; if everything permitted is
        # above the ceiling, take the cheapest of those rather than fail —
        # overspending beats not running.
        model_id, model_rank = within[0] if within else eligible[-1]

        note = ""
        if family_key(model_id) != family_key(role.optimistic):
            # Say which of the two reasons applies — "not on the allow-list" is
            # actively misleading for a model that is listed but not invokable.
            if family_key(role.optimistic) in self._unusable:
                why = f"{role.optimistic} is permitted but not invokable"
            else:
                why = f"{role.optimistic} is not on the allow-list"
            note = (
                f"{role.name!r} → {why}; using {describe(model_id)} instead. "
                f"Pin with VP_MODEL_{role.name.upper()} to skip this probe."
            )
        return Assignment(
            role=role.name,
            model_id=model_id,
            rank=model_rank,
            source="ranked",
            at_floor=model_rank == role.floor_rank,
            note=note,
        )

    def _unavailable_message(self, role: Role, ranked: list[tuple[str, int | None]]) -> str:
        best = ranked[0] if ranked else None
        best_note = (
            f"Best permitted is {describe(best[0])}." if best else "Nothing is permitted."
        )
        unusable_note = (
            f"  Permitted but not invokable by the backend "
            f"(IAM on the governor role): {', '.join(sorted(self._unusable))}\n"
            if self._unusable else ""
        )
        return (
            f"No usable model meets the floor for the {role.name!r} role "
            f"({describe(role.floor)}). {best_note}\n"
            f"  Why this role has a floor: {role.rationale}\n"
            f"  Allowed: {', '.join(self._allowed or []) or '(none reported)'}\n"
            f"{unusable_note}"
            f"  Fix: ask the platform team to add a model at or above the floor "
            f"to the governor's SSM allow-list, or override deliberately with "
            f"VP_MODEL_{role.name.upper()}=<allowed-model-id> to accept the "
            f"reduced quality for this role."
        )


# ------------------------------------------------------------- preflight

PROBE_MODEL = "__scout_preflight_probe__"


def discover_allowed_models(probe) -> list[str] | None:
    """Learn the allow-list up front by deliberately tripping a 403.

    `probe(model_id)` should issue a minimal request and let the backend's
    exception propagate. A sentinel ID is used so the rejection is certain
    and costs no inference — the governor refuses before reaching the model.
    Returns None when the backend answers in a shape without an allow-list,
    in which case resolution falls back to learning from the first real 403.
    """
    try:
        probe(PROBE_MODEL)
    except Exception as exc:  # noqa: BLE001 - any backend error may carry the list
        allowed = parse_allowed_models(exc)
        if allowed:
            return allowed
        print(
            f"  [llm] preflight probe returned no allow-list "
            f"({type(exc).__name__}); falling back to learning it on first use.",
            flush=True,
        )
        return None
    # A sentinel that is *accepted* means this backend does not gate models
    # (e.g. a direct Anthropic API key rather than the governor).
    print("  [llm] preflight: backend does not gate models; no allow-list.", flush=True)
    return None


def render_manifest(resolver: ModelResolver, assignments: dict[str, Assignment]) -> str:
    """Human-readable summary of what will serve each role this run."""
    lines = ["--- LLM preflight ---"]
    allowed = resolver.allowed
    if allowed:
        unusable = {family_key(m) for m in resolver.unusable}
        lines.append(f"  allow-list ({len(allowed)} models):")
        for model_id in sorted(allowed, key=lambda m: rank(m) or -1, reverse=True):
            flag = "  [not invokable — IAM]" if family_key(model_id) in unusable else ""
            lines.append(f"    {describe(model_id)}{flag}")
    else:
        lines.append("  allow-list: not reported (backend may not gate models)")

    lines.append("  role assignments:")
    for name, assignment in sorted(assignments.items()):
        flags = []
        if assignment.source == "pinned":
            flags.append("pinned")
        if assignment.at_floor:
            flags.append("at floor")
        suffix = f"  [{', '.join(flags)}]" if flags else ""
        lines.append(f"    {name:<10} → {describe(assignment.model_id)}{suffix}")
    return "\n".join(lines)


def select_interactively(resolver: ModelResolver) -> None:
    """Let a local operator pick a model per role from the allow-list.

    Opt-in via `VP_MODEL_SELECT=1` and only when stdin is a terminal, so a
    compute-node run in Docker is never blocked waiting on input.
    """
    if os.environ.get("VP_MODEL_SELECT", "").strip() not in ("1", "true", "yes"):
        return
    if not sys.stdin.isatty():
        print("  [llm] VP_MODEL_SELECT set but stdin is not a TTY; skipping.", flush=True)
        return
    allowed = resolver.allowed
    if not allowed:
        print("  [llm] no allow-list to choose from; skipping selection.", flush=True)
        return

    ordered = sorted(allowed, key=lambda m: rank(m) or -1, reverse=True)
    for name, role in ROLES.items():
        print(f"\n  {name} — {role.rationale}")
        for i, model_id in enumerate(ordered, 1):
            floor_note = "" if (rank(model_id) or -1) >= role.floor_rank else "  (below floor)"
            print(f"    {i}. {describe(model_id)}{floor_note}")
        choice = input(f"  select model for {name} [enter to keep auto]: ").strip()
        if not choice:
            continue
        try:
            picked = ordered[int(choice) - 1]
        except (ValueError, IndexError):
            print("    invalid selection; keeping auto.")
            continue
        os.environ[f"VP_MODEL_{name.upper()}"] = picked
        print(f"    {name} pinned to {picked}")
