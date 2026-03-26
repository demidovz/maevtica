from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from typing import Sequence


World = tuple[int, ...]


@dataclass(frozen=True)
class HierarchyBudgetConfig:
    feature_count: int = 8
    seed: int = 7


def parse_args() -> HierarchyBudgetConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Run five budgeted hierarchical-questioning experiments: one-question level "
            "choice, two-question allocation, skipping upper levels in a known family, "
            "mode-before-rule on regime return, and risk-sensitive budget allocation."
        )
    )
    parser.add_argument("--feature-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    config = HierarchyBudgetConfig(
        feature_count=args.feature_count,
        seed=args.seed,
    )
    validate_config(config)
    return config


def validate_config(config: HierarchyBudgetConfig) -> None:
    if config.feature_count != 8:
        raise ValueError("belief_hierarchy_budget_batch_suite currently expects feature_count == 8")


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


def initial_ambiguous_worlds(feature_count: int) -> list[World]:
    return [
        build_world(feature_count, 0, 0, 1),
        build_world(feature_count, 0, 1, 1),
        build_world(feature_count, 1, 0, 1),
    ]


def score_candidate(kind: str, worlds: Sequence[World], true_kind: str) -> int:
    return sum(
        1 if predict_rule(kind, world) == predict_rule(true_kind, world) else 0
        for world in worlds
    )


def stage1_one_question_level_choice(config: HierarchyBudgetConfig) -> list[dict[str, object]]:
    observed = initial_ambiguous_worlds(config.feature_count)
    family_probe = build_world(config.feature_count, 0, 1, 0)
    rule_probe = build_world(config.feature_count, 1, 1, 0)
    table = rule_truth_table(config.feature_count)
    priors = {
        "hybrid_and_xor": 0.55,
        "or_0_2": 0.30,
        "xor_0_2": 0.15,
    }
    rows: list[dict[str, object]] = []

    for policy_name, extra_probe in (
        ("rule_level", rule_probe),
        ("family_level", family_probe),
    ):
        evidence = list(observed) + [extra_probe]
        scored = [
            (
                score_candidate(kind, evidence, "or_0_2"),
                priors[kind],
                kind,
            )
            for kind in ("hybrid_and_xor", "or_0_2", "xor_0_2")
        ]
        scored.sort(reverse=True)
        selected = scored[0][2]
        table_accuracy = statistics.mean(
            1 if predict_rule(selected, world) == predict_rule("or_0_2", world) else 0
            for world in table
        )
        rows.append(
            {
                "policy": policy_name,
                "Probe": probe_label(extra_probe),
                "Selected": selected,
                "Exact": 1 if selected == "or_0_2" else 0,
                "TableAcc": round(table_accuracy, 3),
            }
        )
    return rows


def stage2_two_question_allocation(config: HierarchyBudgetConfig) -> list[dict[str, object]]:
    observed = initial_ambiguous_worlds(config.feature_count)
    family_probe_a = build_world(config.feature_count, 0, 1, 0)
    family_probe_b = build_world(config.feature_count, 0, 1, 1)
    rule_probe = build_world(config.feature_count, 1, 1, 0)
    table = rule_truth_table(config.feature_count)
    priors = {
        "hybrid_and_xor": 0.30,
        "or_0_2": 0.15,
        "xor_0_2": 0.55,
    }
    rows: list[dict[str, object]] = []

    for policy_name, probes in (
        ("two_family", [family_probe_a, family_probe_b]),
        ("family_then_rule", [family_probe_a, rule_probe]),
    ):
        evidence = list(observed) + list(probes)
        scored = [
            (
                score_candidate(kind, evidence, "or_0_2"),
                priors[kind],
                kind,
            )
            for kind in ("hybrid_and_xor", "or_0_2", "xor_0_2")
        ]
        scored.sort(reverse=True)
        selected = scored[0][2]
        table_accuracy = statistics.mean(
            1 if predict_rule(selected, world) == predict_rule("or_0_2", world) else 0
            for world in table
        )
        rows.append(
            {
                "policy": policy_name,
                "Plan": ",".join(probe_label(probe) for probe in probes),
                "Selected": selected,
                "Exact": 1 if selected == "or_0_2" else 0,
                "TableAcc": round(table_accuracy, 3),
            }
        )
    return rows


def stage3_skip_known_upper_level(config: HierarchyBudgetConfig) -> list[dict[str, object]]:
    observed = initial_ambiguous_worlds(config.feature_count)
    family_probe = build_world(config.feature_count, 0, 1, 0)
    rule_probe = build_world(config.feature_count, 1, 1, 0)
    table = rule_truth_table(config.feature_count)
    priors = {
        "or_0_2": 0.40,
        "xor_0_2": 0.60,
    }
    rows: list[dict[str, object]] = []

    for policy_name, probe in (
        ("waste_on_family", family_probe),
        ("ask_rule_directly", rule_probe),
    ):
        evidence = list(observed) + [probe]
        scored = [
            (
                score_candidate(kind, evidence, "or_0_2"),
                priors[kind],
                kind,
            )
            for kind in ("or_0_2", "xor_0_2")
        ]
        scored.sort(reverse=True)
        selected = scored[0][2]
        table_accuracy = statistics.mean(
            1 if predict_rule(selected, world) == predict_rule("or_0_2", world) else 0
            for world in table
        )
        rows.append(
            {
                "policy": policy_name,
                "Probe": probe_label(probe),
                "Selected": selected,
                "Exact": 1 if selected == "or_0_2" else 0,
                "TableAcc": round(table_accuracy, 3),
            }
        )
    return rows


def stage4_mode_before_rule(config: HierarchyBudgetConfig) -> list[dict[str, object]]:
    table = package_truth_table(config.feature_count)
    rule_probe = build_world(config.feature_count, 1, 1, 0, 0)
    mode_probe = build_world(config.feature_count, 1, 1, 0, 1)
    rows: list[dict[str, object]] = []

    for policy_name, selected, probe in (
        ("rule_first", "or_0_2", rule_probe),
        ("mode_first", "package_social", mode_probe),
    ):
        if selected.startswith("package_"):
            table_accuracy = statistics.mean(
                1 if predict_package(selected, world) == predict_package("package_social", world) else 0
                for world in table
            )
        else:
            table_accuracy = statistics.mean(
                1 if predict_rule(selected, world) == predict_package("package_social", world) else 0
                for world in table
            )
        rows.append(
            {
                "policy": policy_name,
                "Probe": probe_label(probe),
                "Selected": selected,
                "ExactPkg": 1 if selected == "package_social" else 0,
                "TableAcc": round(table_accuracy, 3),
            }
        )
    return rows


def stage5_risk_sensitive_budget(config: HierarchyBudgetConfig) -> list[dict[str, object]]:
    table = package_truth_table(config.feature_count)
    high_risk_labels = {
        probe_label(build_world(config.feature_count, 1, 1, 0, 1)),
        probe_label(build_world(config.feature_count, 1, 1, 1, 1)),
    }
    options = [
        {
            "policy": "avg_budget",
            "Probe": "cheap_rule",
            "Cost": 0.2,
            "Selected": "or_0_2",
        },
        {
            "policy": "risk_budget",
            "Probe": "expensive_mode",
            "Cost": 2.6,
            "Selected": "package_social",
        },
    ]
    rows: list[dict[str, object]] = []

    for option in options:
        correct_flags: list[int] = []
        risk_utility = 0.0
        for world in table:
            label = probe_label(world)
            if option["Selected"].startswith("package_"):
                correct = 1 if predict_package(option["Selected"], world) == predict_package("package_social", world) else 0
            else:
                correct = 1 if predict_rule(option["Selected"], world) == predict_package("package_social", world) else 0
            correct_flags.append(correct)
            if label in high_risk_labels:
                risk_utility += 5.0 if correct else 0.0
            else:
                risk_utility += 1.0 if correct else 0.0

        average_net = sum(correct_flags) - option["Cost"]
        risk_net = risk_utility - option["Cost"]
        rows.append(
            {
                "policy": option["policy"],
                "Probe": option["Probe"],
                "Selected": option["Selected"],
                "AvgNet": round(average_net, 3),
                "RiskNet": round(risk_net, 3),
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
    print(f"Hierarchy budget batch seed: {config.seed}")
    print()

    print_stage("Stage 1: one-question level choice", stage1_one_question_level_choice(config))
    print_stage("Stage 2: two-question allocation", stage2_two_question_allocation(config))
    print_stage("Stage 3: skip upper level in a known family", stage3_skip_known_upper_level(config))
    print_stage("Stage 4: mode-before-rule on regime return", stage4_mode_before_rule(config))
    print_stage("Stage 5: risk-sensitive budget allocation", stage5_risk_sensitive_budget(config))


if __name__ == "__main__":
    main()
