from __future__ import annotations

import random

from epistemic_engine.beliefs.state import normalize
from epistemic_engine.models import QuestionAction


HYPOTHESES = {
    "config_mismatch": "Bug caused by conflicting config or env override.",
    "parser_bug": "Bug caused by a parser edge case in incoming payload handling.",
    "dependency_drift": "Bug caused by dependency drift between build and runtime.",
    "state_leak": "Bug caused by stale state surviving across requests or retries.",
    "schema_migration": "Bug caused by incomplete schema migration or backfill.",
    "race_condition": "Bug caused by missing synchronization under concurrency.",
}


MODE_IDS = list(HYPOTHESES)


ACTIONS = [
    QuestionAction(
        action_id="inspect_error_log",
        description="Read the main error log excerpt from the incident.",
        cost=0.8,
        action_type="telemetry",
    ),
    QuestionAction(
        action_id="ask_user_scope",
        description="Ask when and where the failure shows up from the user's point of view.",
        cost=0.9,
        action_type="ask_user",
    ),
    QuestionAction(
        action_id="inspect_recent_diff",
        description="Inspect the most suspicious part of the recent release diff.",
        cost=1.0,
        action_type="history",
    ),
    QuestionAction(
        action_id="inspect_config",
        description="Inspect the runtime config and env overrides.",
        cost=1.2,
        action_type="inspect_artifact",
    ),
    QuestionAction(
        action_id="inspect_lockfile",
        description="Inspect lockfile and pinned dependency versions.",
        cost=1.4,
        action_type="inspect_artifact",
    ),
    QuestionAction(
        action_id="inspect_hot_path_code",
        description="Inspect the most suspicious hot path code snippet.",
        cost=1.9,
        action_type="inspect_code",
    ),
    QuestionAction(
        action_id="inspect_migration_history",
        description="Inspect migration and backfill history for the affected tables.",
        cost=2.2,
        action_type="inspect_artifact",
    ),
    QuestionAction(
        action_id="run_targeted_regression",
        description="Run a narrow reproduction or regression report for the incident.",
        cost=2.3,
        action_type="run_test",
    ),
    QuestionAction(
        action_id="collect_thread_dump",
        description="Inspect thread dump or worker contention snapshot.",
        cost=2.4,
        action_type="probe_runtime",
    ),
]


def _distribution(primary: str, secondary: str, generic: str) -> dict[str, float]:
    return {
        primary: 0.60,
        secondary: 0.25,
        generic: 0.15,
    }


HYPOTHESIS_ACTION_LIKELIHOODS = {
    "inspect_error_log": {
        "config_mismatch": _distribution(
            "AuthConfigError: expected audience public-api, got internal-api from STAGING_AUTH_AUDIENCE.",
            "Service bootstrap aborted while loading auth settings for billing-api.",
            "Request handler bubbled up a generic startup failure after the rollout.",
        ),
        "parser_bug": _distribution(
            "ValueError: trailing comma while decoding partner NDJSON batch in parser/decoder.py:118.",
            "Parser fast path dropped the final byte of a unicode payload chunk.",
            "Request handler bubbled up a generic parse failure after the rollout.",
        ),
        "dependency_drift": _distribution(
            "ImportError: cannot import name Retry from urllib3.util.retry in billing bootstrap.",
            "ModuleNotFoundError: optional transport shim missing in runtime image.",
            "Worker startup failed after dependency initialization.",
        ),
        "state_leak": _distribution(
            "CacheStateError: request context already closed for this retry token.",
            "Poisoned in-memory cache reused stale tenant snapshot after retry.",
            "Second and later attempts fail with a generic state-related exception.",
        ),
        "schema_migration": _distribution(
            "psycopg.errors.UndefinedColumn: column accounts.legacy_tier does not exist.",
            "Serializer crashed while reading legacy_tier from hydrated account row.",
            "Database layer returned a generic schema-related failure.",
        ),
        "race_condition": _distribution(
            "TimeoutError waiting on flush barrier for worker shard-3.",
            "Concurrent writer corrupted inflight_jobs map during parallel flush.",
            "Under load the worker pool surfaces a generic timeout and retry burst.",
        ),
    },
    "ask_user_scope": {
        "config_mismatch": _distribution(
            "User report: only staging and canary fail right after secret rotation.",
            "User report: local works, staging fails after deploy with the new env pack.",
            "User report: failure started right after deploy but exact scope is fuzzy.",
        ),
        "parser_bug": _distribution(
            "User report: only marketplace payloads fail; internal fixtures still pass.",
            "User report: crashes only on CSV-to-JSON imported feeds with odd unicode symbols.",
            "User report: the endpoint fails only for one class of input data.",
        ),
        "dependency_drift": _distribution(
            "User report: prod pods fail, developer shell and CI both look fine.",
            "User report: issue appears only in fresh images after the runtime rebuild.",
            "User report: failure started after deploy but scope looks environment-specific.",
        ),
        "state_leak": _distribution(
            "User report: first attempt often works, retries or warm requests fail.",
            "User report: issue appears after a few repeated requests on the same tenant.",
            "User report: restart briefly helps before the failure comes back.",
        ),
        "schema_migration": _distribution(
            "User report: only old accounts fail; newly created accounts still work.",
            "User report: read path broke right after the migration rollout to replicas.",
            "User report: failures cluster around older data rows after deploy.",
        ),
        "race_condition": _distribution(
            "User report: issue is flaky and mostly appears during traffic spikes.",
            "User report: one worker fails under parallel load while serial replay passes.",
            "User report: retries sometimes hide the bug but load makes it obvious.",
        ),
    },
    "inspect_recent_diff": {
        "config_mismatch": _distribution(
            "Diff: staging.env changed BILLING_AUTH_AUDIENCE to internal-api while app code still expects public-api.",
            "Diff: config loader now prefers ENV shadow value over service defaults.",
            "Diff: release touched deploy wiring more than application logic.",
        ),
        "parser_bug": _distribution(
            "Diff: parser/decoder.py added a fast path for chunked NDJSON normalization.",
            "Diff: payload sanitizer now strips a trailing byte before final brace repair.",
            "Diff: release mostly touched ingest code and payload normalization paths.",
        ),
        "dependency_drift": _distribution(
            "Diff: poetry.lock bumped urllib3 and requests together with image rebuild metadata.",
            "Diff: optional transport shim moved behind a new extra requirement.",
            "Diff: release touched dependency pins more than product logic.",
        ),
        "state_leak": _distribution(
            "Diff: request cache moved from local context into a module-level singleton.",
            "Diff: retry helper now reuses warmed tenant snapshot across attempts.",
            "Diff: release touched cache lifecycle and retry orchestration paths.",
        ),
        "schema_migration": _distribution(
            "Diff: serializer now reads legacy_tier before backfill rollout completed.",
            "Diff: migration adds legacy_tier but replica-safe backfill step is missing.",
            "Diff: release touched schema-adjacent code and read model assumptions.",
        ),
        "race_condition": _distribution(
            "Diff: worker pool size jumped from 8 to 24 without adding a flush lock.",
            "Diff: inflight_jobs map moved outside a synchronized section.",
            "Diff: release touched worker scheduling and batch flush concurrency.",
        ),
    },
    "inspect_config": {
        "config_mismatch": _distribution(
            "Config: STAGING_AUTH_AUDIENCE overrides service.yml and points to internal-api.",
            "Config: partial env override applies only to canary pods and bypasses service defaults.",
            "Config: values differ across environments but only one override looks dangerous.",
        ),
        "parser_bug": _distribution(
            "Config: parser-related flags look normal; no suspicious env override found.",
            "Config: ingest limits differ a bit across environments but payload parsing flags are unchanged.",
            "Config: no smoking gun in runtime config for this incident.",
        ),
        "dependency_drift": _distribution(
            "Config: runtime image digest differs from the lockfile build digest.",
            "Config: package layer hash in runtime differs from the hash recorded by CI.",
            "Config: service config is mostly clean, but build/runtime identity does not match.",
        ),
        "state_leak": _distribution(
            "Config: cache TTL and retry knobs changed, but values alone should not explain the failure.",
            "Config: warm-cache feature flag is enabled only on the failing pool.",
            "Config: retry and cache settings look borderline but not conclusive.",
        ),
        "schema_migration": _distribution(
            "Config: read replicas are enabled before migration lag check finishes.",
            "Config: rollout flag points reads to replica pool while backfill is still pending.",
            "Config: config is mostly clean except for replica rollout timing.",
        ),
        "race_condition": _distribution(
            "Config: worker concurrency jumped and batching window shrank on the failing pool.",
            "Config: queue parallelism flag is much higher in prod than in CI.",
            "Config: concurrency-related knobs differ across environments.",
        ),
    },
    "inspect_lockfile": {
        "config_mismatch": _distribution(
            "Lockfile: dependency pins look unchanged; no suspicious transitive bump.",
            "Lockfile: only unrelated observability packages changed.",
            "Lockfile: build dependencies look boring for this incident.",
        ),
        "parser_bug": _distribution(
            "Lockfile: no parser dependency drift; failure likely lives in application code.",
            "Lockfile: only benign utility packages changed.",
            "Lockfile: package graph looks stable for the ingest path.",
        ),
        "dependency_drift": _distribution(
            "Lockfile: urllib3 pinned to 2.2.1, but runtime image still ships 1.26.x.",
            "Lockfile: optional extra for transport shim is present in lock but absent in runtime layer.",
            "Lockfile: dependency graph reveals a real build/runtime skew.",
        ),
        "state_leak": _distribution(
            "Lockfile: no meaningful dependency drift in cache or retry stack.",
            "Lockfile: package graph is noisy but nothing points to state reuse.",
            "Lockfile: transitive bumps look unrelated to the incident.",
        ),
        "schema_migration": _distribution(
            "Lockfile: ORM and migration tool versions are stable across build and runtime.",
            "Lockfile: no meaningful dependency skew around the database layer.",
            "Lockfile: package versions look boring for this failure.",
        ),
        "race_condition": _distribution(
            "Lockfile: queue and worker libs are unchanged; problem likely lives in app-level synchronization.",
            "Lockfile: runtime packages do not explain the new contention pattern.",
            "Lockfile: no meaningful dependency clue for the concurrency failure.",
        ),
    },
    "inspect_hot_path_code": {
        "config_mismatch": _distribution(
            "Code: auth bootstrap now rejects any audience not in {'public-api'} before fallback logic.",
            "Code: env override is read before service defaults and no normalization follows.",
            "Code: hot path shows a strict config guard with no compatibility shim.",
        ),
        "parser_bug": _distribution(
            "Code: decoder trims the last byte before patching closing braces for chunked NDJSON.",
            "Code: unicode repair path mutates raw bytes before JSON validation.",
            "Code: hot path shows an ingest fast path with brittle boundary handling.",
        ),
        "dependency_drift": _distribution(
            "Code: runtime import assumes Retry class layout from urllib3 2.x only.",
            "Code: optional transport shim is imported unguarded in bootstrap.",
            "Code: hot path hardcodes an API shape that changed across dependency versions.",
        ),
        "state_leak": _distribution(
            "Code: warmed tenant snapshot stored in module-level cache and reused after retry.",
            "Code: request context reset happens on success path only.",
            "Code: hot path shows stale state leaking across attempts.",
        ),
        "schema_migration": _distribution(
            "Code: serializer requires row['legacy_tier'] before compatibility fallback.",
            "Code: read model assumes migration already finished on every replica.",
            "Code: hot path bakes in a column that old rows do not have yet.",
        ),
        "race_condition": _distribution(
            "Code: inflight_jobs map is mutated outside the flush lock.",
            "Code: worker completion path checks shared state without synchronization.",
            "Code: hot path shows a classic missing lock around shared mutable state.",
        ),
    },
    "inspect_migration_history": {
        "config_mismatch": _distribution(
            "Migration log: database schema changes are clean and unrelated to the incident window.",
            "Migration log: no pending backfills or broken rollouts around the failure time.",
            "Migration log: schema history looks healthy for this incident.",
        ),
        "parser_bug": _distribution(
            "Migration log: backfills completed long before the ingest regression appeared.",
            "Migration log: no schema rollout coincides with the parser failure window.",
            "Migration log: database history does not explain the payload failure.",
        ),
        "dependency_drift": _distribution(
            "Migration log: schema state is clean; issue likely sits above the database layer.",
            "Migration log: no migration anomaly lines up with the runtime import crash.",
            "Migration log: database rollout looks unrelated to the failure.",
        ),
        "state_leak": _distribution(
            "Migration log: schema history is clean; warm-cache issue appears in app state only.",
            "Migration log: no backfill lag coincides with retry poisoning.",
            "Migration log: migration timeline looks boring for this bug.",
        ),
        "schema_migration": _distribution(
            "Migration log: 2026_03_24_add_legacy_tier.sql applied, but replica backfill job never finished.",
            "Migration log: reader rollout began before legacy_tier backfill completed on old accounts.",
            "Migration log: migration history contains a real partial rollout.",
        ),
        "race_condition": _distribution(
            "Migration log: schema rollout clean; timeout spike comes from worker contention instead.",
            "Migration log: no database rollout anomaly near the concurrency failure.",
            "Migration log: migration history does not explain the load-only timeout.",
        ),
    },
    "run_targeted_regression": {
        "config_mismatch": _distribution(
            "Regression: auth smoke passes locally but fails with staging env pack.",
            "Regression: same binary passes once config override is removed.",
            "Regression: targeted repro isolates the failing env combination.",
        ),
        "parser_bug": _distribution(
            "Regression: partner_batch.ndjson fails while golden fixture and clean unicode sample pass.",
            "Regression: unicode edge fixture fails only on the chunked ingest fast path.",
            "Regression: targeted repro isolates one payload family instead of every request.",
        ),
        "dependency_drift": _distribution(
            "Regression: unit tests pass in lockfile venv but crash inside runtime image shell.",
            "Regression: import smoke fails only in the rebuilt prod image.",
            "Regression: targeted repro isolates build/runtime skew instead of business logic.",
        ),
        "state_leak": _distribution(
            "Regression: first run passes, second run with warmed cache fails on the same tenant.",
            "Regression: clean-process replay passes but warm-retry replay fails.",
            "Regression: targeted repro needs state carry-over to surface the bug.",
        ),
        "schema_migration": _distribution(
            "Regression: old-account fixture fails until legacy_tier backfill is applied.",
            "Regression: serializer passes for new rows but fails for migrated reader on old rows.",
            "Regression: targeted repro isolates data shape created before the latest migration.",
        ),
        "race_condition": _distribution(
            "Regression: serial replay passes, parallel stress run flakes 7/50 times.",
            "Regression: bug appears only under concurrent flush pressure.",
            "Regression: targeted repro isolates load and interleaving, not static input.",
        ),
    },
    "collect_thread_dump": {
        "config_mismatch": _distribution(
            "Thread dump: workers mostly idle; no lock contention beyond startup retries.",
            "Thread dump: no meaningful contention, only repeated bootstrap failures.",
            "Thread dump: runtime snapshot is mostly quiet apart from failed startup loops.",
        ),
        "parser_bug": _distribution(
            "Thread dump: workers idle while one ingest thread burns CPU in payload repair loop.",
            "Thread dump: no shared lock contention; failure sits inside a single parse path.",
            "Thread dump: runtime looks CPU-bound in one parser branch, not globally blocked.",
        ),
        "dependency_drift": _distribution(
            "Thread dump: runtime dies before workers really start; no contention visible.",
            "Thread dump: startup threads exit early after import bootstrap failure.",
            "Thread dump: snapshot shows almost no concurrent activity before crash.",
        ),
        "state_leak": _distribution(
            "Thread dump: retry worker loops over poisoned cache entries without clearing context.",
            "Thread dump: background cleanup thread never resets stale tenant snapshot.",
            "Thread dump: runtime points to repeated state reuse rather than lock contention.",
        ),
        "schema_migration": _distribution(
            "Thread dump: threads wait on database retries, not on app-level locks.",
            "Thread dump: runtime mostly blocked on repeated query failures for old rows.",
            "Thread dump: snapshot points to query failure loops, not concurrency bugs.",
        ),
        "race_condition": _distribution(
            "Thread dump: two flush workers contend on inflight_jobs while queue depth spikes.",
            "Thread dump: worker shard-3 holds flush lock while shard-5 spins on shared map access.",
            "Thread dump: runtime snapshot shows real lock contention under load.",
        ),
    },
}


ACTION_SMOOTHING_ALPHA = {
    "inspect_error_log": 0.55,
    "ask_user_scope": 0.45,
    "inspect_recent_diff": 0.40,
    "inspect_config": 0.45,
    "inspect_lockfile": 0.35,
    "inspect_hot_path_code": 0.25,
    "inspect_migration_history": 0.22,
    "run_targeted_regression": 0.18,
    "collect_thread_dump": 0.22,
}


class ArtifactDebuggingEnvironment:
    def __init__(
        self,
        actual_hypothesis: str | None = None,
        seed: int = 7,
        max_cost: float | None = None,
        max_steps: int | None = None,
    ) -> None:
        self.rng = random.Random(seed)
        self.actual_hypothesis = actual_hypothesis or self.rng.choice(list(HYPOTHESES))
        self.actual_profile = self.actual_hypothesis
        self.max_cost = max_cost
        self.max_steps = max_steps

    def hypotheses(self) -> dict[str, str]:
        return HYPOTHESES

    def mode_ids(self) -> list[str]:
        return MODE_IDS

    def actions(self) -> list[QuestionAction]:
        return ACTIONS

    def action_by_id(self, action_id: str) -> QuestionAction:
        for action in ACTIONS:
            if action.action_id == action_id:
                return action
        raise KeyError(action_id)

    def available_actions(
        self,
        asked_actions: list[str],
        remaining_budget: float | None = None,
    ) -> list[QuestionAction]:
        asked = set(asked_actions)
        candidates = [action for action in ACTIONS if action.action_id not in asked]
        if remaining_budget is None:
            return candidates
        return [action for action in candidates if action.cost <= remaining_budget + 1e-9]

    def remaining_budget(self, state) -> float | None:
        if self.max_cost is None:
            return None
        return max(0.0, self.max_cost - state.total_cost)

    def candidate_actions(self, state) -> list[QuestionAction]:
        if self.max_steps is not None and len(state.history) >= self.max_steps:
            return []
        return self.available_actions(
            state.asked_actions,
            remaining_budget=self.remaining_budget(state),
        )

    def mode_support_to_hypotheses(self, mode_support: dict[str, float]) -> dict[str, float]:
        return normalize(
            {
                hypothesis_id: mode_support.get(hypothesis_id, 0.0) + 1e-6
                for hypothesis_id in HYPOTHESES
            }
        )

    def hypotheses_given_mode(self, mode_id: str) -> dict[str, float]:
        return normalize(
            {
                hypothesis_id: 1.0 if hypothesis_id == mode_id else 1e-6
                for hypothesis_id in HYPOTHESES
            }
        )

    def mode_likelihood(self, action_id: str, outcome: str, mode_id: str) -> float:
        return self._smoothed_distribution(action_id, mode_id).get(outcome, 0.0)

    def mode_distribution(self, action_id: str, mode_id: str) -> dict[str, float]:
        return self._smoothed_distribution(action_id, mode_id)

    def stop_reason(self, state) -> str | None:
        if self.max_steps is not None and len(state.history) >= self.max_steps:
            return "step_limit"
        remaining_budget = self.remaining_budget(state)
        unasked_actions = self.available_actions(state.asked_actions)
        if remaining_budget is not None and unasked_actions:
            affordable = self.available_actions(
                state.asked_actions,
                remaining_budget=remaining_budget,
            )
            if not affordable:
                return "budget_limit"
        if not unasked_actions:
            return "no_actions"
        return None

    def outcomes_for(self, action_id: str) -> tuple[str, ...]:
        outcomes: set[str] = set()
        for distribution in HYPOTHESIS_ACTION_LIKELIHOODS[action_id].values():
            outcomes.update(distribution)
        return tuple(sorted(outcomes))

    def likelihood(self, action_id: str, outcome: str, hypothesis_id: str) -> float:
        return self._smoothed_distribution(action_id, hypothesis_id).get(outcome, 0.0)

    def sample_observation(self, action_id: str) -> str:
        distribution = HYPOTHESIS_ACTION_LIKELIHOODS[action_id][self.actual_hypothesis]
        outcomes = list(distribution)
        weights = [distribution[outcome] for outcome in outcomes]
        return self.rng.choices(outcomes, weights=weights, k=1)[0]

    def _smoothed_distribution(
        self,
        action_id: str,
        hypothesis_id: str,
    ) -> dict[str, float]:
        raw = HYPOTHESIS_ACTION_LIKELIHOODS[action_id][hypothesis_id]
        alpha = ACTION_SMOOTHING_ALPHA[action_id]
        outcomes = self.outcomes_for(action_id)
        uniform = 1.0 / len(outcomes)
        return {
            outcome: (1.0 - alpha) * raw.get(outcome, 0.0) + alpha * uniform
            for outcome in outcomes
        }
