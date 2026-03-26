from __future__ import annotations

import random

from epistemic_engine.beliefs.state import normalize
from epistemic_engine.models import QuestionAction


HYPOTHESES = {
    "config_mismatch": "Баг вызван рассинхроном конфига или env-переменных.",
    "parser_bug": "Баг вызван ошибкой парсера входных данных.",
    "dependency_drift": "Баг вызван дрейфом зависимостей или lockfile.",
    "state_leak": "Баг вызван утечкой или загрязнением состояния между запусками.",
    "schema_migration": "Баг вызван несовместимой миграцией схемы данных.",
    "race_condition": "Баг вызван гонкой между параллельными воркерами.",
}


PROFILE_PRIORS = {
    "config_mismatch": {
        "startup_failure": 0.55,
        "auth_drift": 0.45,
    },
    "parser_bug": {
        "bad_payload": 0.60,
        "unicode_edge": 0.40,
    },
    "dependency_drift": {
        "startup_failure": 0.35,
        "import_breakage": 0.65,
    },
    "state_leak": {
        "dirty_retry": 0.65,
        "memory_bloat": 0.35,
    },
    "schema_migration": {
        "bad_payload": 0.45,
        "missing_column": 0.55,
    },
    "race_condition": {
        "flaky_worker": 0.70,
        "timeout_burst": 0.30,
    },
}

PROFILE_TRANSITIONS = {
    "config_mismatch": {
        "startup_failure": "auth_drift",
        "auth_drift": "startup_failure",
    },
    "parser_bug": {
        "bad_payload": "unicode_edge",
        "unicode_edge": "bad_payload",
    },
    "dependency_drift": {
        "startup_failure": "import_breakage",
        "import_breakage": "startup_failure",
    },
    "state_leak": {
        "dirty_retry": "memory_bloat",
        "memory_bloat": "dirty_retry",
    },
    "schema_migration": {
        "bad_payload": "missing_column",
        "missing_column": "bad_payload",
    },
    "race_condition": {
        "flaky_worker": "timeout_burst",
        "timeout_burst": "flaky_worker",
    },
}

MODE_IDS = sorted(
    {
        profile_id
        for profile_distribution in PROFILE_PRIORS.values()
        for profile_id in profile_distribution
    }
)


ACTIONS = [
    QuestionAction(
        action_id="inspect_dashboard",
        description="Посмотреть дешёвые сигналы с дашборда и алертов.",
        cost=0.5,
        action_type="telemetry",
    ),
    QuestionAction(
        action_id="inspect_error_log",
        description="Посмотреть сигнатуру ошибки в логах.",
        cost=0.8,
        action_type="telemetry",
    ),
    QuestionAction(
        action_id="inspect_recent_diff",
        description="Посмотреть недавний diff релиза.",
        cost=1.0,
        action_type="history",
    ),
    QuestionAction(
        action_id="ask_user_scope",
        description="Спросить пользователя, когда и как именно проявляется сбой.",
        cost=0.9,
        action_type="ask_user",
    ),
    QuestionAction(
        action_id="inspect_config",
        description="Проверить конфиг и env-переменные.",
        cost=1.2,
        action_type="inspect_artifact",
    ),
    QuestionAction(
        action_id="inspect_lockfile",
        description="Проверить lockfile и версии зависимостей.",
        cost=1.4,
        action_type="inspect_artifact",
    ),
    QuestionAction(
        action_id="inspect_hot_path_code",
        description="Посмотреть код горячего пути и недавние защитные ветки.",
        cost=1.9,
        action_type="inspect_code",
    ),
    QuestionAction(
        action_id="run_payload_fixture",
        description="Прогнать репродукцию на фиксированном payload fixture.",
        cost=2.1,
        action_type="run_test",
    ),
    QuestionAction(
        action_id="rerun_clean_process",
        description="Повторить сценарий в чистом процессе.",
        cost=2.0,
        action_type="run_test",
    ),
    QuestionAction(
        action_id="run_targeted_regression",
        description="Запустить узкий regression test под подозреваемый класс поломки.",
        cost=2.3,
        action_type="run_test",
    ),
    QuestionAction(
        action_id="inspect_migration_history",
        description="Проверить историю миграций и backfill-ов.",
        cost=2.2,
        action_type="inspect_artifact",
    ),
    QuestionAction(
        action_id="collect_thread_dump",
        description="Снять thread dump и посмотреть contention/retry loops.",
        cost=2.4,
        action_type="probe_runtime",
    ),
]


PROFILE_ACTION_LIKELIHOODS = {
    "inspect_dashboard": {
        "startup_failure": {
            "error_spike": 0.35,
            "deploy_spike": 0.35,
            "stable": 0.15,
            "latency_spike": 0.05,
            "retry_spike": 0.10,
        },
        "auth_drift": {
            "error_spike": 0.45,
            "deploy_spike": 0.20,
            "stable": 0.25,
            "latency_spike": 0.05,
            "retry_spike": 0.05,
        },
        "bad_payload": {
            "error_spike": 0.50,
            "deploy_spike": 0.10,
            "stable": 0.20,
            "latency_spike": 0.05,
            "retry_spike": 0.15,
        },
        "unicode_edge": {
            "error_spike": 0.35,
            "deploy_spike": 0.10,
            "stable": 0.35,
            "latency_spike": 0.05,
            "retry_spike": 0.15,
        },
        "import_breakage": {
            "error_spike": 0.40,
            "deploy_spike": 0.30,
            "stable": 0.20,
            "latency_spike": 0.05,
            "retry_spike": 0.05,
        },
        "dirty_retry": {
            "error_spike": 0.15,
            "deploy_spike": 0.00,
            "stable": 0.20,
            "latency_spike": 0.10,
            "retry_spike": 0.55,
        },
        "memory_bloat": {
            "error_spike": 0.10,
            "deploy_spike": 0.05,
            "stable": 0.15,
            "latency_spike": 0.50,
            "retry_spike": 0.20,
        },
        "missing_column": {
            "error_spike": 0.45,
            "deploy_spike": 0.15,
            "stable": 0.15,
            "latency_spike": 0.05,
            "retry_spike": 0.20,
        },
        "flaky_worker": {
            "error_spike": 0.15,
            "deploy_spike": 0.10,
            "stable": 0.15,
            "latency_spike": 0.25,
            "retry_spike": 0.35,
        },
        "timeout_burst": {
            "error_spike": 0.15,
            "deploy_spike": 0.05,
            "stable": 0.10,
            "latency_spike": 0.45,
            "retry_spike": 0.25,
        },
    },
    "inspect_error_log": {
        "startup_failure": {
            "startup_trace": 0.55,
            "auth_hint": 0.10,
            "payload_parse_hint": 0.00,
            "import_trace": 0.10,
            "state_hint": 0.00,
            "timeout_hint": 0.00,
            "schema_hint": 0.00,
            "generic": 0.25,
        },
        "auth_drift": {
            "startup_trace": 0.10,
            "auth_hint": 0.55,
            "payload_parse_hint": 0.00,
            "import_trace": 0.05,
            "state_hint": 0.05,
            "timeout_hint": 0.05,
            "schema_hint": 0.00,
            "generic": 0.20,
        },
        "bad_payload": {
            "startup_trace": 0.00,
            "auth_hint": 0.05,
            "payload_parse_hint": 0.45,
            "import_trace": 0.00,
            "state_hint": 0.05,
            "timeout_hint": 0.05,
            "schema_hint": 0.20,
            "generic": 0.20,
        },
        "unicode_edge": {
            "startup_trace": 0.00,
            "auth_hint": 0.05,
            "payload_parse_hint": 0.40,
            "import_trace": 0.05,
            "state_hint": 0.10,
            "timeout_hint": 0.05,
            "schema_hint": 0.10,
            "generic": 0.25,
        },
        "import_breakage": {
            "startup_trace": 0.15,
            "auth_hint": 0.05,
            "payload_parse_hint": 0.00,
            "import_trace": 0.60,
            "state_hint": 0.00,
            "timeout_hint": 0.05,
            "schema_hint": 0.00,
            "generic": 0.15,
        },
        "dirty_retry": {
            "startup_trace": 0.00,
            "auth_hint": 0.00,
            "payload_parse_hint": 0.05,
            "import_trace": 0.00,
            "state_hint": 0.55,
            "timeout_hint": 0.10,
            "schema_hint": 0.00,
            "generic": 0.30,
        },
        "memory_bloat": {
            "startup_trace": 0.05,
            "auth_hint": 0.00,
            "payload_parse_hint": 0.05,
            "import_trace": 0.05,
            "state_hint": 0.40,
            "timeout_hint": 0.20,
            "schema_hint": 0.00,
            "generic": 0.25,
        },
        "missing_column": {
            "startup_trace": 0.00,
            "auth_hint": 0.05,
            "payload_parse_hint": 0.20,
            "import_trace": 0.00,
            "state_hint": 0.05,
            "timeout_hint": 0.00,
            "schema_hint": 0.50,
            "generic": 0.20,
        },
        "flaky_worker": {
            "startup_trace": 0.05,
            "auth_hint": 0.00,
            "payload_parse_hint": 0.05,
            "import_trace": 0.05,
            "state_hint": 0.35,
            "timeout_hint": 0.30,
            "schema_hint": 0.00,
            "generic": 0.20,
        },
        "timeout_burst": {
            "startup_trace": 0.05,
            "auth_hint": 0.00,
            "payload_parse_hint": 0.00,
            "import_trace": 0.05,
            "state_hint": 0.15,
            "timeout_hint": 0.55,
            "schema_hint": 0.00,
            "generic": 0.20,
        },
    },
}


HYPOTHESIS_ACTION_LIKELIHOODS = {
    "ask_user_scope": {
        "config_mismatch": {
            "started_after_env_change": 0.50,
            "started_after_payload_change": 0.05,
            "started_after_deploy": 0.15,
            "mostly_after_retries": 0.05,
            "looks_flaky": 0.05,
            "unclear_report": 0.20,
        },
        "parser_bug": {
            "started_after_env_change": 0.05,
            "started_after_payload_change": 0.45,
            "started_after_deploy": 0.10,
            "mostly_after_retries": 0.05,
            "looks_flaky": 0.10,
            "unclear_report": 0.25,
        },
        "dependency_drift": {
            "started_after_env_change": 0.15,
            "started_after_payload_change": 0.05,
            "started_after_deploy": 0.45,
            "mostly_after_retries": 0.10,
            "looks_flaky": 0.05,
            "unclear_report": 0.20,
        },
        "state_leak": {
            "started_after_env_change": 0.05,
            "started_after_payload_change": 0.05,
            "started_after_deploy": 0.05,
            "mostly_after_retries": 0.45,
            "looks_flaky": 0.20,
            "unclear_report": 0.20,
        },
        "schema_migration": {
            "started_after_env_change": 0.05,
            "started_after_payload_change": 0.25,
            "started_after_deploy": 0.35,
            "mostly_after_retries": 0.05,
            "looks_flaky": 0.15,
            "unclear_report": 0.15,
        },
        "race_condition": {
            "started_after_env_change": 0.05,
            "started_after_payload_change": 0.00,
            "started_after_deploy": 0.10,
            "mostly_after_retries": 0.20,
            "looks_flaky": 0.50,
            "unclear_report": 0.15,
        },
    },
    "inspect_recent_diff": {
        "config_mismatch": {
            "config_touch": 0.60,
            "parser_touch": 0.00,
            "dependency_touch": 0.10,
            "migration_touch": 0.00,
            "runtime_touch": 0.10,
            "no_clear_diff": 0.20,
        },
        "parser_bug": {
            "config_touch": 0.00,
            "parser_touch": 0.60,
            "dependency_touch": 0.00,
            "migration_touch": 0.05,
            "runtime_touch": 0.10,
            "no_clear_diff": 0.25,
        },
        "dependency_drift": {
            "config_touch": 0.10,
            "parser_touch": 0.00,
            "dependency_touch": 0.65,
            "migration_touch": 0.00,
            "runtime_touch": 0.10,
            "no_clear_diff": 0.15,
        },
        "state_leak": {
            "config_touch": 0.00,
            "parser_touch": 0.05,
            "dependency_touch": 0.10,
            "migration_touch": 0.00,
            "runtime_touch": 0.60,
            "no_clear_diff": 0.25,
        },
        "schema_migration": {
            "config_touch": 0.10,
            "parser_touch": 0.15,
            "dependency_touch": 0.00,
            "migration_touch": 0.60,
            "runtime_touch": 0.00,
            "no_clear_diff": 0.15,
        },
        "race_condition": {
            "config_touch": 0.05,
            "parser_touch": 0.10,
            "dependency_touch": 0.05,
            "migration_touch": 0.00,
            "runtime_touch": 0.55,
            "no_clear_diff": 0.25,
        },
    },
    "inspect_hot_path_code": {
        "config_mismatch": {
            "env_branch_guard": 0.55,
            "parser_edge_case": 0.00,
            "version_shim": 0.10,
            "dirty_cache_path": 0.05,
            "migration_cast": 0.00,
            "missing_lock_guard": 0.05,
            "nothing_obvious": 0.25,
        },
        "parser_bug": {
            "env_branch_guard": 0.05,
            "parser_edge_case": 0.60,
            "version_shim": 0.00,
            "dirty_cache_path": 0.05,
            "migration_cast": 0.10,
            "missing_lock_guard": 0.00,
            "nothing_obvious": 0.20,
        },
        "dependency_drift": {
            "env_branch_guard": 0.10,
            "parser_edge_case": 0.05,
            "version_shim": 0.55,
            "dirty_cache_path": 0.05,
            "migration_cast": 0.00,
            "missing_lock_guard": 0.05,
            "nothing_obvious": 0.20,
        },
        "state_leak": {
            "env_branch_guard": 0.05,
            "parser_edge_case": 0.05,
            "version_shim": 0.05,
            "dirty_cache_path": 0.55,
            "migration_cast": 0.00,
            "missing_lock_guard": 0.10,
            "nothing_obvious": 0.20,
        },
        "schema_migration": {
            "env_branch_guard": 0.05,
            "parser_edge_case": 0.10,
            "version_shim": 0.05,
            "dirty_cache_path": 0.00,
            "migration_cast": 0.60,
            "missing_lock_guard": 0.00,
            "nothing_obvious": 0.20,
        },
        "race_condition": {
            "env_branch_guard": 0.00,
            "parser_edge_case": 0.05,
            "version_shim": 0.05,
            "dirty_cache_path": 0.10,
            "migration_cast": 0.00,
            "missing_lock_guard": 0.60,
            "nothing_obvious": 0.20,
        },
    },
    "inspect_config": {
        "config_mismatch": {
            "env_shadow": 0.60,
            "partial_override": 0.25,
            "config_clean": 0.15,
        },
        "parser_bug": {
            "env_shadow": 0.10,
            "partial_override": 0.10,
            "config_clean": 0.80,
        },
        "dependency_drift": {
            "env_shadow": 0.20,
            "partial_override": 0.15,
            "config_clean": 0.65,
        },
        "state_leak": {
            "env_shadow": 0.05,
            "partial_override": 0.05,
            "config_clean": 0.90,
        },
        "schema_migration": {
            "env_shadow": 0.15,
            "partial_override": 0.15,
            "config_clean": 0.70,
        },
        "race_condition": {
            "env_shadow": 0.05,
            "partial_override": 0.10,
            "config_clean": 0.85,
        },
    },
    "inspect_lockfile": {
        "config_mismatch": {
            "version_skew": 0.10,
            "recent_pin": 0.15,
            "locked_clean": 0.75,
        },
        "parser_bug": {
            "version_skew": 0.05,
            "recent_pin": 0.10,
            "locked_clean": 0.85,
        },
        "dependency_drift": {
            "version_skew": 0.65,
            "recent_pin": 0.20,
            "locked_clean": 0.15,
        },
        "state_leak": {
            "version_skew": 0.05,
            "recent_pin": 0.05,
            "locked_clean": 0.90,
        },
        "schema_migration": {
            "version_skew": 0.10,
            "recent_pin": 0.10,
            "locked_clean": 0.80,
        },
        "race_condition": {
            "version_skew": 0.05,
            "recent_pin": 0.10,
            "locked_clean": 0.85,
        },
    },
    "run_payload_fixture": {
        "config_mismatch": {
            "fixture_fail": 0.10,
            "schema_fail": 0.05,
            "clean_pass": 0.85,
        },
        "parser_bug": {
            "fixture_fail": 0.70,
            "schema_fail": 0.10,
            "clean_pass": 0.20,
        },
        "dependency_drift": {
            "fixture_fail": 0.15,
            "schema_fail": 0.05,
            "clean_pass": 0.80,
        },
        "state_leak": {
            "fixture_fail": 0.05,
            "schema_fail": 0.05,
            "clean_pass": 0.90,
        },
        "schema_migration": {
            "fixture_fail": 0.20,
            "schema_fail": 0.65,
            "clean_pass": 0.15,
        },
        "race_condition": {
            "fixture_fail": 0.10,
            "schema_fail": 0.05,
            "clean_pass": 0.85,
        },
    },
    "rerun_clean_process": {
        "config_mismatch": {
            "fixed": 0.10,
            "still_broken": 0.85,
            "becomes_flaky": 0.05,
        },
        "parser_bug": {
            "fixed": 0.05,
            "still_broken": 0.90,
            "becomes_flaky": 0.05,
        },
        "dependency_drift": {
            "fixed": 0.15,
            "still_broken": 0.75,
            "becomes_flaky": 0.10,
        },
        "state_leak": {
            "fixed": 0.55,
            "still_broken": 0.15,
            "becomes_flaky": 0.30,
        },
        "schema_migration": {
            "fixed": 0.05,
            "still_broken": 0.90,
            "becomes_flaky": 0.05,
        },
        "race_condition": {
            "fixed": 0.15,
            "still_broken": 0.20,
            "becomes_flaky": 0.65,
        },
    },
    "run_targeted_regression": {
        "config_mismatch": {
            "env_case_fail": 0.65,
            "parser_case_fail": 0.00,
            "import_regression": 0.05,
            "state_reset_needed": 0.05,
            "migration_case_fail": 0.00,
            "parallel_flake": 0.05,
            "passes_targeted": 0.20,
        },
        "parser_bug": {
            "env_case_fail": 0.05,
            "parser_case_fail": 0.65,
            "import_regression": 0.00,
            "state_reset_needed": 0.00,
            "migration_case_fail": 0.05,
            "parallel_flake": 0.05,
            "passes_targeted": 0.20,
        },
        "dependency_drift": {
            "env_case_fail": 0.05,
            "parser_case_fail": 0.05,
            "import_regression": 0.60,
            "state_reset_needed": 0.05,
            "migration_case_fail": 0.05,
            "parallel_flake": 0.00,
            "passes_targeted": 0.20,
        },
        "state_leak": {
            "env_case_fail": 0.05,
            "parser_case_fail": 0.05,
            "import_regression": 0.00,
            "state_reset_needed": 0.55,
            "migration_case_fail": 0.00,
            "parallel_flake": 0.15,
            "passes_targeted": 0.20,
        },
        "schema_migration": {
            "env_case_fail": 0.05,
            "parser_case_fail": 0.10,
            "import_regression": 0.00,
            "state_reset_needed": 0.00,
            "migration_case_fail": 0.60,
            "parallel_flake": 0.05,
            "passes_targeted": 0.20,
        },
        "race_condition": {
            "env_case_fail": 0.05,
            "parser_case_fail": 0.05,
            "import_regression": 0.05,
            "state_reset_needed": 0.05,
            "migration_case_fail": 0.00,
            "parallel_flake": 0.60,
            "passes_targeted": 0.20,
        },
    },
    "inspect_migration_history": {
        "config_mismatch": {
            "missing_backfill": 0.05,
            "risky_migration": 0.10,
            "clean_migrations": 0.85,
        },
        "parser_bug": {
            "missing_backfill": 0.10,
            "risky_migration": 0.10,
            "clean_migrations": 0.80,
        },
        "dependency_drift": {
            "missing_backfill": 0.05,
            "risky_migration": 0.10,
            "clean_migrations": 0.85,
        },
        "state_leak": {
            "missing_backfill": 0.05,
            "risky_migration": 0.05,
            "clean_migrations": 0.90,
        },
        "schema_migration": {
            "missing_backfill": 0.55,
            "risky_migration": 0.25,
            "clean_migrations": 0.20,
        },
        "race_condition": {
            "missing_backfill": 0.05,
            "risky_migration": 0.10,
            "clean_migrations": 0.85,
        },
    },
    "collect_thread_dump": {
        "config_mismatch": {
            "lock_contention": 0.05,
            "orphan_retry_loop": 0.05,
            "clean_threads": 0.90,
        },
        "parser_bug": {
            "lock_contention": 0.05,
            "orphan_retry_loop": 0.05,
            "clean_threads": 0.90,
        },
        "dependency_drift": {
            "lock_contention": 0.10,
            "orphan_retry_loop": 0.10,
            "clean_threads": 0.80,
        },
        "state_leak": {
            "lock_contention": 0.15,
            "orphan_retry_loop": 0.50,
            "clean_threads": 0.35,
        },
        "schema_migration": {
            "lock_contention": 0.05,
            "orphan_retry_loop": 0.05,
            "clean_threads": 0.90,
        },
        "race_condition": {
            "lock_contention": 0.60,
            "orphan_retry_loop": 0.20,
            "clean_threads": 0.20,
        },
    },
}


PROFILE_ACTION_TYPE_STRENGTHS = {
    "startup_failure": {
        "telemetry": 0.85,
        "ask_user": 0.55,
        "history": 0.85,
        "inspect_artifact": 0.95,
        "inspect_code": 0.45,
        "run_test": 0.30,
        "probe_runtime": 0.15,
    },
    "auth_drift": {
        "telemetry": 0.75,
        "ask_user": 0.90,
        "history": 0.65,
        "inspect_artifact": 0.90,
        "inspect_code": 0.40,
        "run_test": 0.35,
        "probe_runtime": 0.20,
    },
    "bad_payload": {
        "telemetry": 0.60,
        "ask_user": 0.70,
        "history": 0.40,
        "inspect_artifact": 0.35,
        "inspect_code": 0.80,
        "run_test": 0.95,
        "probe_runtime": 0.25,
    },
    "unicode_edge": {
        "telemetry": 0.55,
        "ask_user": 0.70,
        "history": 0.35,
        "inspect_artifact": 0.30,
        "inspect_code": 0.90,
        "run_test": 0.80,
        "probe_runtime": 0.25,
    },
    "import_breakage": {
        "telemetry": 0.75,
        "ask_user": 0.40,
        "history": 0.90,
        "inspect_artifact": 0.95,
        "inspect_code": 0.65,
        "run_test": 0.35,
        "probe_runtime": 0.20,
    },
    "dirty_retry": {
        "telemetry": 0.55,
        "ask_user": 0.45,
        "history": 0.25,
        "inspect_artifact": 0.20,
        "inspect_code": 0.75,
        "run_test": 0.85,
        "probe_runtime": 0.90,
    },
    "memory_bloat": {
        "telemetry": 0.60,
        "ask_user": 0.35,
        "history": 0.20,
        "inspect_artifact": 0.15,
        "inspect_code": 0.60,
        "run_test": 0.65,
        "probe_runtime": 0.95,
    },
    "missing_column": {
        "telemetry": 0.70,
        "ask_user": 0.45,
        "history": 0.85,
        "inspect_artifact": 0.90,
        "inspect_code": 0.70,
        "run_test": 0.75,
        "probe_runtime": 0.20,
    },
    "flaky_worker": {
        "telemetry": 0.60,
        "ask_user": 0.50,
        "history": 0.25,
        "inspect_artifact": 0.15,
        "inspect_code": 0.65,
        "run_test": 0.70,
        "probe_runtime": 0.95,
    },
    "timeout_burst": {
        "telemetry": 0.65,
        "ask_user": 0.45,
        "history": 0.25,
        "inspect_artifact": 0.15,
        "inspect_code": 0.55,
        "run_test": 0.80,
        "probe_runtime": 0.90,
    },
}


class DebuggingToyEnvironment:
    def __init__(
        self,
        actual_hypothesis: str | None = None,
        seed: int = 7,
        max_cost: float | None = None,
        max_steps: int | None = None,
    ) -> None:
        self.rng = random.Random(seed)
        self.actual_hypothesis = actual_hypothesis or self.rng.choice(list(HYPOTHESES))
        self.actual_profile = self._sample_profile(self.actual_hypothesis)
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
        return [
            action
            for action in candidates
            if action.cost <= remaining_budget + 1e-9
        ]

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

    def mode_support_to_hypotheses(
        self,
        mode_support: dict[str, float],
    ) -> dict[str, float]:
        scores = {hypothesis_id: 1e-6 for hypothesis_id in HYPOTHESES}
        for hypothesis_id, profile_distribution in PROFILE_PRIORS.items():
            for mode_id, support in mode_support.items():
                scores[hypothesis_id] += support * profile_distribution.get(mode_id, 0.0)
        return {
            hypothesis_id: score / sum(scores.values())
            for hypothesis_id, score in scores.items()
        }

    def hypotheses_given_mode(self, mode_id: str) -> dict[str, float]:
        scores = {
            hypothesis_id: profile_distribution.get(mode_id, 0.0)
            for hypothesis_id, profile_distribution in PROFILE_PRIORS.items()
        }
        return normalize(scores)

    def mode_likelihood(self, action_id: str, outcome: str, mode_id: str) -> float:
        if action_id in PROFILE_ACTION_LIKELIHOODS:
            return PROFILE_ACTION_LIKELIHOODS[action_id][mode_id].get(outcome, 0.0)

        hypothesis_weights = self.hypotheses_given_mode(mode_id)
        return sum(
            weight * HYPOTHESIS_ACTION_LIKELIHOODS[action_id][hypothesis_id].get(outcome, 0.0)
            for hypothesis_id, weight in hypothesis_weights.items()
        )

    def mode_distribution(self, action_id: str, mode_id: str) -> dict[str, float]:
        return {
            outcome: self.mode_likelihood(action_id, outcome, mode_id)
            for outcome in self.outcomes_for(action_id)
        }

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
        distributions = self._action_distributions(action_id)
        outcomes: set[str] = set()
        for distribution in distributions:
            outcomes.update(distribution)
        return tuple(sorted(outcomes))

    def likelihood(self, action_id: str, outcome: str, hypothesis_id: str) -> float:
        if action_id in PROFILE_ACTION_LIKELIHOODS:
            return sum(
                profile_weight
                * PROFILE_ACTION_LIKELIHOODS[action_id][profile_id].get(outcome, 0.0)
                for profile_id, profile_weight in PROFILE_PRIORS[hypothesis_id].items()
            )
        return HYPOTHESIS_ACTION_LIKELIHOODS[action_id][hypothesis_id].get(outcome, 0.0)

    def sample_observation(self, action_id: str) -> str:
        if action_id in PROFILE_ACTION_LIKELIHOODS:
            distribution = PROFILE_ACTION_LIKELIHOODS[action_id][self.actual_profile]
        else:
            distribution = HYPOTHESIS_ACTION_LIKELIHOODS[action_id][self.actual_hypothesis]
        outcomes = list(distribution)
        weights = [distribution[outcome] for outcome in outcomes]
        return self.rng.choices(outcomes, weights=weights, k=1)[0]

    def _sample_profile(self, hypothesis_id: str) -> str:
        profile_distribution = PROFILE_PRIORS[hypothesis_id]
        profiles = list(profile_distribution)
        weights = [profile_distribution[profile] for profile in profiles]
        return self.rng.choices(profiles, weights=weights, k=1)[0]

    def _action_distributions(self, action_id: str) -> list[dict[str, float]]:
        if action_id in PROFILE_ACTION_LIKELIHOODS:
            return list(PROFILE_ACTION_LIKELIHOODS[action_id].values())
        return list(HYPOTHESIS_ACTION_LIKELIHOODS[action_id].values())


class DebuggingModeShiftEnvironment(DebuggingToyEnvironment):
    def __init__(
        self,
        actual_hypothesis: str | None = None,
        seed: int = 7,
        max_cost: float | None = None,
        max_steps: int | None = None,
        shift_after_step: int | None = None,
    ) -> None:
        super().__init__(
            actual_hypothesis=actual_hypothesis,
            seed=seed,
            max_cost=max_cost,
            max_steps=max_steps,
        )
        self.initial_profile = self.actual_profile
        self.shifted_profile = PROFILE_TRANSITIONS[self.actual_hypothesis][self.initial_profile]
        self.shift_after_step = shift_after_step or self._sample_shift_step()
        self.observations_emitted = 0

    def active_profile(self) -> str:
        if self.observations_emitted < self.shift_after_step:
            return self.initial_profile
        return self.shifted_profile

    def sample_observation(self, action_id: str) -> str:
        active_profile = self.active_profile()
        if action_id in PROFILE_ACTION_LIKELIHOODS:
            distribution = PROFILE_ACTION_LIKELIHOODS[action_id][active_profile]
        else:
            distribution = HYPOTHESIS_ACTION_LIKELIHOODS[action_id][self.actual_hypothesis]
        outcomes = list(distribution)
        weights = [distribution[outcome] for outcome in outcomes]
        self.observations_emitted += 1
        return self.rng.choices(outcomes, weights=weights, k=1)[0]

    def _sample_shift_step(self) -> int:
        if self.max_steps is None:
            return 2
        max_shift_step = max(2, self.max_steps - 1)
        return self.rng.randint(2, max_shift_step)


class DebuggingQuestionValueShiftEnvironment(DebuggingModeShiftEnvironment):
    def __init__(
        self,
        actual_hypothesis: str | None = None,
        seed: int = 7,
        max_cost: float | None = None,
        max_steps: int | None = None,
        shift_after_step: int | None = None,
    ) -> None:
        super().__init__(
            actual_hypothesis=actual_hypothesis,
            seed=seed,
            max_cost=max_cost,
            max_steps=max_steps,
            shift_after_step=shift_after_step,
        )
        self.generic_action_distributions = {
            action_id: self._build_generic_action_distribution(action_id)
            for action_id in HYPOTHESIS_ACTION_LIKELIHOODS
        }

    def likelihood(self, action_id: str, outcome: str, hypothesis_id: str) -> float:
        if action_id in PROFILE_ACTION_LIKELIHOODS:
            return super().likelihood(action_id, outcome, hypothesis_id)

        return sum(
            profile_weight
            * self._profile_conditioned_distribution(
                action_id,
                hypothesis_id,
                profile_id,
            ).get(outcome, 0.0)
            for profile_id, profile_weight in PROFILE_PRIORS[hypothesis_id].items()
        )

    def mode_likelihood(self, action_id: str, outcome: str, mode_id: str) -> float:
        if action_id in PROFILE_ACTION_LIKELIHOODS:
            return super().mode_likelihood(action_id, outcome, mode_id)

        return self._mode_action_distribution(action_id, mode_id).get(outcome, 0.0)

    def sample_observation(self, action_id: str) -> str:
        active_profile = self.active_profile()
        if action_id in PROFILE_ACTION_LIKELIHOODS:
            distribution = PROFILE_ACTION_LIKELIHOODS[action_id][active_profile]
        else:
            distribution = self._profile_conditioned_distribution(
                action_id,
                self.actual_hypothesis,
                active_profile,
            )

        outcomes = list(distribution)
        weights = [distribution[outcome] for outcome in outcomes]
        self.observations_emitted += 1
        return self.rng.choices(outcomes, weights=weights, k=1)[0]

    def _action_distributions(self, action_id: str) -> list[dict[str, float]]:
        if action_id in PROFILE_ACTION_LIKELIHOODS:
            return super()._action_distributions(action_id)

        return [
            self._mode_action_distribution(action_id, mode_id)
            for mode_id in MODE_IDS
        ]

    def _profile_conditioned_distribution(
        self,
        action_id: str,
        hypothesis_id: str,
        profile_id: str,
    ) -> dict[str, float]:
        base_distribution = HYPOTHESIS_ACTION_LIKELIHOODS[action_id][hypothesis_id]
        generic_distribution = self.generic_action_distributions[action_id]
        action_type = self.action_by_id(action_id).action_type
        strength = PROFILE_ACTION_TYPE_STRENGTHS[profile_id][action_type]

        outcomes = sorted(set(base_distribution) | set(generic_distribution))
        return {
            outcome: (
                strength * base_distribution.get(outcome, 0.0)
                + (1.0 - strength) * generic_distribution.get(outcome, 0.0)
            )
            for outcome in outcomes
        }

    def _mode_action_distribution(
        self,
        action_id: str,
        mode_id: str,
    ) -> dict[str, float]:
        hypothesis_weights = self.hypotheses_given_mode(mode_id)
        scores = {
            outcome: 0.0
            for outcome in self.generic_action_distributions[action_id]
        }
        for hypothesis_id, weight in hypothesis_weights.items():
            distribution = self._profile_conditioned_distribution(
                action_id,
                hypothesis_id,
                mode_id,
            )
            for outcome, probability in distribution.items():
                scores[outcome] += weight * probability
        return normalize(scores)

    def _build_generic_action_distribution(
        self,
        action_id: str,
    ) -> dict[str, float]:
        distributions = HYPOTHESIS_ACTION_LIKELIHOODS[action_id].values()
        outcome_scores: dict[str, float] = {}
        for distribution in distributions:
            for outcome, probability in distribution.items():
                outcome_scores[outcome] = outcome_scores.get(outcome, 0.0) + probability
        return normalize(outcome_scores)


class DebuggingFalseAlarmQuestionValueEnvironment(DebuggingQuestionValueShiftEnvironment):
    def __init__(
        self,
        actual_hypothesis: str | None = None,
        seed: int = 7,
        max_cost: float | None = None,
        max_steps: int | None = None,
        shift_after_step: int | None = None,
        false_alarm_length: int = 1,
    ) -> None:
        super().__init__(
            actual_hypothesis=actual_hypothesis,
            seed=seed,
            max_cost=max_cost,
            max_steps=max_steps,
            shift_after_step=shift_after_step,
        )
        self.false_alarm_length = false_alarm_length

    def active_profile(self) -> str:
        if (
            self.shift_after_step
            <= self.observations_emitted
            < self.shift_after_step + self.false_alarm_length
        ):
            return self.shifted_profile
        return self.initial_profile

    def scenario_label(self) -> str:
        return "false_alarm"


class DebuggingAmbiguousShiftEnvironment(DebuggingQuestionValueShiftEnvironment):
    def __init__(
        self,
        actual_hypothesis: str | None = None,
        seed: int = 7,
        max_cost: float | None = None,
        max_steps: int | None = None,
        shift_after_step: int | None = None,
        shift_probability: float = 0.5,
        false_alarm_length: int = 1,
    ) -> None:
        super().__init__(
            actual_hypothesis=actual_hypothesis,
            seed=seed,
            max_cost=max_cost,
            max_steps=max_steps,
            shift_after_step=shift_after_step,
        )
        self.false_alarm_length = false_alarm_length
        self.has_true_shift = self.rng.random() < shift_probability

    def active_profile(self) -> str:
        if self.has_true_shift:
            return super().active_profile()
        if (
            self.shift_after_step
            <= self.observations_emitted
            < self.shift_after_step + self.false_alarm_length
        ):
            return self.shifted_profile
        return self.initial_profile

    def scenario_label(self) -> str:
        return "true_shift" if self.has_true_shift else "false_alarm"
