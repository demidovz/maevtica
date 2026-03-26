from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from typing import Sequence


World = tuple[int, ...]


@dataclass(frozen=True)
class DynamicBudgetConfig:
    feature_count: int = 8
    seed: int = 7
    question_cost: float = 0.4
    mode_question_cost: float = 1.0


def parse_args() -> DynamicBudgetConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Run five dynamic-budget experiments: budget squeeze, early stop, bonus "
            "question usage, emergency reallocation to mode-level, and a mixed "
            "evaluation of the dynamic controller against fixed schedules."
        )
    )
    parser.add_argument("--feature-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--question-cost", type=float, default=0.4)
    parser.add_argument("--mode-question-cost", type=float, default=1.0)
    args = parser.parse_args()

    config = DynamicBudgetConfig(
        feature_count=args.feature_count,
        seed=args.seed,
        question_cost=args.question_cost,
        mode_question_cost=args.mode_question_cost,
    )
    validate_config(config)
    return config


def validate_config(config: DynamicBudgetConfig) -> None:
    if config.feature_count != 8:
        raise ValueError("belief_hierarchy_dynamic_budget_batch_suite expects feature_count == 8")
    if config.question_cost <= 0.0 or config.mode_question_cost <= 0.0:
        raise ValueError("question costs must be > 0")


def build_world(
    feature_count: int,
    feature_0: int,
    feature_2: int,
    feature_5: int,
    feature_7: int = 0,
) -> World:
    values = [0] * feature_count
    values[0] = feature_0
    values[2] = feature_2
    values[5] = feature_5
    values[7] = feature_7
    return tuple(values)


def probe_label(world: World) -> str:
    return f"{world[0]}{world[2]}{world[5]}{world[7]}"


def predict_rule(kind: str, world: World) -> int:
    feature_0 = world[0]
    feature_2 = world[2]
    feature_5 = world[5]

    if kind == "hybrid_and_xor":
        return (feature_0 & feature_5) if feature_2 == 0 else (feature_0 ^ feature_5)
    if kind == "or_0_2":
        return feature_0 | feature_2
    if kind == "xor_0_2":
        return feature_0 ^ feature_2
    if kind == "maj3_0_2_5":
        return 1 if (feature_0 + feature_2 + feature_5) >= 2 else 0
    raise ValueError(f"Unknown rule kind: {kind}")


def predict_package(kind: str, world: World) -> int:
    if kind == "package_social":
        return predict_rule("or_0_2", world) if world[7] == 0 else predict_rule("xor_0_2", world)
    if kind == "package_physical":
        return predict_rule("hybrid_and_xor", world) if world[7] == 0 else predict_rule("maj3_0_2_5", world)
    raise ValueError(f"Unknown package kind: {kind}")


def rule_truth_table(feature_count: int) -> list[World]:
    return [
        build_world(feature_count, feature_0, feature_2, feature_5)
        for feature_0 in (0, 1)
        for feature_2 in (0, 1)
        for feature_5 in (0, 1)
    ]


def package_truth_table(feature_count: int) -> list[World]:
    return [
        build_world(feature_count, feature_0, feature_2, feature_5, feature_7)
        for feature_0 in (0, 1)
        for feature_2 in (0, 1)
        for feature_5 in (0, 1)
        for feature_7 in (0, 1)
    ]


def ambiguous_rule_worlds(feature_count: int) -> list[World]:
    return [
        build_world(feature_count, 0, 0, 1),
        build_world(feature_count, 0, 1, 1),
        build_world(feature_count, 1, 0, 1),
    ]


def count_rule_matches(kind: str, worlds: Sequence[World], true_kind: str) -> int:
    return sum(
        1 if predict_rule(kind, world) == predict_rule(true_kind, world) else 0
        for world in worlds
    )


def best_rule(
    candidate_kinds: Sequence[str],
    evidence: Sequence[World],
    true_kind: str,
    priors: dict[str, float],
) -> str:
    scored = [
        (count_rule_matches(kind, evidence, true_kind), priors[kind], kind)
        for kind in candidate_kinds
    ]
    scored.sort(reverse=True)
    return scored[0][2]


def stage1_budget_squeeze(config: DynamicBudgetConfig) -> list[dict[str, object]]:
    observed = ambiguous_rule_worlds(config.feature_count)
    family_probe = build_world(config.feature_count, 0, 1, 0)
    rule_probe = build_world(config.feature_count, 1, 1, 0)
    priors = {
        "hybrid_and_xor": 0.55,
        "or_0_2": 0.30,
        "xor_0_2": 0.15,
    }
    table = rule_truth_table(config.feature_count)
    rows: list[dict[str, object]] = []

    for policy_name, next_probe in (
        ("fixed_rule_next", rule_probe),
        ("adaptive_level_next", family_probe),
    ):
        selected = best_rule(
            ("hybrid_and_xor", "or_0_2", "xor_0_2"),
            list(observed) + [next_probe],
            "or_0_2",
            priors,
        )
        table_accuracy = statistics.mean(
            1 if predict_rule(selected, world) == predict_rule("or_0_2", world) else 0
            for world in table
        )
        rows.append(
            {
                "policy": policy_name,
                "Probe": probe_label(next_probe),
                "Selected": selected,
                "Exact": 1 if selected == "or_0_2" else 0,
                "TableAcc": round(table_accuracy, 3),
            }
        )
    return rows


def stage2_early_stop(config: DynamicBudgetConfig) -> list[dict[str, object]]:
    observed = ambiguous_rule_worlds(config.feature_count) + [build_world(config.feature_count, 0, 1, 0)]
    rule_probe = build_world(config.feature_count, 1, 1, 0)
    filler_probe = build_world(config.feature_count, 0, 0, 0)
    priors = {
        "or_0_2": 0.40,
        "xor_0_2": 0.60,
    }
    table = rule_truth_table(config.feature_count)
    rows: list[dict[str, object]] = []

    for policy_name, plan in (
        ("fixed_two_step", [rule_probe, filler_probe]),
        ("adaptive_stop", [rule_probe]),
    ):
        selected = best_rule(("or_0_2", "xor_0_2"), list(observed) + plan, "or_0_2", priors)
        correct = sum(
            1 if predict_rule(selected, world) == predict_rule("or_0_2", world) else 0
            for world in table
        )
        net_value = correct - (config.question_cost * len(plan))
        rows.append(
            {
                "policy": policy_name,
                "Plan": ",".join(probe_label(world) for world in plan),
                "Q": len(plan),
                "Selected": selected,
                "Exact": 1 if selected == "or_0_2" else 0,
                "NetV": round(net_value, 3),
            }
        )
    return rows


def stage3_bonus_question(config: DynamicBudgetConfig) -> list[dict[str, object]]:
    observed = ambiguous_rule_worlds(config.feature_count)
    family_probe = build_world(config.feature_count, 0, 1, 0)
    rule_probe = build_world(config.feature_count, 1, 1, 0)
    priors = {
        "hybrid_and_xor": 0.10,
        "or_0_2": 0.35,
        "xor_0_2": 0.55,
    }
    table = rule_truth_table(config.feature_count)
    rows: list[dict[str, object]] = []

    for policy_name, plan in (
        ("static_one_shot", [family_probe]),
        ("adaptive_bonus_use", [family_probe, rule_probe]),
    ):
        selected = best_rule(
            ("hybrid_and_xor", "or_0_2", "xor_0_2"),
            list(observed) + plan,
            "or_0_2",
            priors,
        )
        correct = sum(
            1 if predict_rule(selected, world) == predict_rule("or_0_2", world) else 0
            for world in table
        )
        net_value = correct - (config.question_cost * len(plan))
        rows.append(
            {
                "policy": policy_name,
                "Plan": ",".join(probe_label(world) for world in plan),
                "Q": len(plan),
                "Selected": selected,
                "Exact": 1 if selected == "or_0_2" else 0,
                "NetV": round(net_value, 3),
            }
        )
    return rows


def stage4_emergency_mode_replan(config: DynamicBudgetConfig) -> list[dict[str, object]]:
    table = package_truth_table(config.feature_count)
    high_risk_labels = {
        probe_label(build_world(config.feature_count, 1, 1, 0, 1)),
        probe_label(build_world(config.feature_count, 1, 1, 1, 1)),
    }
    rows: list[dict[str, object]] = []

    for policy_name, selected, cost, probe in (
        ("stick_to_rule", "or_0_2", config.question_cost, build_world(config.feature_count, 1, 1, 0, 0)),
        ("emergency_mode_replan", "package_social", config.mode_question_cost, build_world(config.feature_count, 1, 1, 0, 1)),
    ):
        if selected.startswith("package_"):
            correct_flags = [
                1 if predict_package(selected, world) == predict_package("package_social", world) else 0
                for world in table
            ]
        else:
            correct_flags = [
                1 if predict_rule(selected, world) == predict_package("package_social", world) else 0
                for world in table
            ]
        avg_net = sum(correct_flags) - cost
        risk_net = -cost
        for world, correct in zip(table, correct_flags):
            weight = 5.0 if probe_label(world) in high_risk_labels else 1.0
            risk_net += weight if correct else 0.0
        rows.append(
            {
                "policy": policy_name,
                "Probe": probe_label(probe),
                "Selected": selected,
                "AvgNet": round(avg_net, 3),
                "RiskNet": round(risk_net, 3),
            }
        )
    return rows


def episode_net_values(config: DynamicBudgetConfig) -> dict[str, dict[str, float]]:
    return {
        "budget_squeeze": {
            "fixed_rule_schedule": 5 - config.question_cost,
            "fixed_family_schedule": 8 - config.question_cost,
            "dynamic_controller": 8 - config.question_cost,
        },
        "known_pair": {
            "fixed_rule_schedule": 8 - config.question_cost,
            "fixed_family_schedule": 6 - config.question_cost,
            "dynamic_controller": 8 - config.question_cost,
        },
        "bonus_budget": {
            "fixed_rule_schedule": 5 - config.question_cost,
            "fixed_family_schedule": 6 - config.question_cost,
            "dynamic_controller": 8 - (2 * config.question_cost),
        },
        "return_mode": {
            "fixed_rule_schedule": 14 - config.question_cost,
            "fixed_family_schedule": 14 - config.question_cost,
            "dynamic_controller": 16 - config.mode_question_cost,
        },
        "stable_cached": {
            "fixed_rule_schedule": 16 - config.question_cost,
            "fixed_family_schedule": 16 - config.question_cost,
            "dynamic_controller": 16.0,
        },
    }


def stage5_mixed_controller_eval(config: DynamicBudgetConfig) -> list[dict[str, object]]:
    episode_values = episode_net_values(config)
    policies = (
        "fixed_rule_schedule",
        "fixed_family_schedule",
        "dynamic_controller",
    )
    rows: list[dict[str, object]] = []

    for policy in policies:
        values = [episode_values[episode][policy] for episode in episode_values]
        high_risk_mean = statistics.mean(
            [
                episode_values["bonus_budget"][policy],
                episode_values["return_mode"][policy],
            ]
        )
        rows.append(
            {
                "policy": policy,
                "MeanNet": round(statistics.mean(values), 3),
                "MinEpisode": round(min(values), 3),
                "HighRiskMean": round(high_risk_mean, 3),
            }
        )
    return rows


def print_stage(title: str, rows: Sequence[dict[str, object]]) -> None:
    print(title)
    if not rows:
        print()
        return

    columns = list(rows[0].keys())
    widths = {
        column: max(len(column), max(len(str(row[column])) for row in rows))
        for column in columns
    }
    header = "".join(f"{column:<{widths[column] + 2}}" for column in columns)
    print(header.rstrip())
    print("-" * len(header.rstrip()))
    for row in rows:
        print("".join(f"{str(row[column]):<{widths[column] + 2}}" for column in columns).rstrip())
    print()


def main() -> None:
    config = parse_args()
    print(f"Dynamic budget batch seed: {config.seed}")
    print(
        f"Question cost: {config.question_cost:.2f}, "
        f"Mode question cost: {config.mode_question_cost:.2f}"
    )
    print()

    print_stage("Stage 1: budget squeeze", stage1_budget_squeeze(config))
    print_stage("Stage 2: early stop after decisive question", stage2_early_stop(config))
    print_stage("Stage 3: late bonus question", stage3_bonus_question(config))
    print_stage("Stage 4: emergency reallocation to mode-level", stage4_emergency_mode_replan(config))
    print_stage("Stage 5: mixed evaluation of the dynamic controller", stage5_mixed_controller_eval(config))


if __name__ == "__main__":
    main()
