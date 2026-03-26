from __future__ import annotations

import argparse
import math
import random
import statistics
from dataclasses import dataclass
from typing import Sequence


World = tuple[int, ...]


@dataclass(frozen=True)
class HierarchyBatchConfig:
    feature_count: int = 8
    seed: int = 7
    consistency: float = 0.8


@dataclass(frozen=True)
class RuleCandidate:
    name: str
    family: str
    subfamily: str
    kind: str


@dataclass(frozen=True)
class PackageCandidate:
    name: str
    kind: str


def parse_args() -> HierarchyBatchConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Run five hierarchical routing experiments: family, subfamily, exact rule, "
            "archive package, and world mode routing."
        )
    )
    parser.add_argument("--feature-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--consistency", type=float, default=0.8)
    args = parser.parse_args()

    config = HierarchyBatchConfig(
        feature_count=args.feature_count,
        seed=args.seed,
        consistency=args.consistency,
    )
    validate_config(config)
    return config


def validate_config(config: HierarchyBatchConfig) -> None:
    if config.feature_count != 8:
        raise ValueError("belief_hierarchy_batch_suite currently expects feature_count == 8")
    if not 0.5 < config.consistency < 1.0:
        raise ValueError("consistency must be in (0.5, 1.0)")


def build_world(
    feature_count: int,
    feature_0: int,
    feature_2: int,
    feature_5: int,
    feature_6: int = 0,
    feature_7: int = 0,
) -> World:
    values = [0] * feature_count
    values[0] = feature_0
    values[2] = feature_2
    values[5] = feature_5
    values[6] = feature_6
    values[7] = feature_7
    return tuple(values)


def probe_label(world: World) -> str:
    return f"{world[0]}{world[2]}{world[5]}{world[7]}"


def base_predict(kind: str, world: World) -> int:
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
    if kind == "nand3_0_2_5":
        return 1 - (feature_0 & feature_2 & feature_5)
    raise ValueError(f"Unknown rule kind: {kind}")


def package_predict(kind: str, world: World) -> int:
    if kind == "package_social":
        return base_predict("or_0_2", world) if world[7] == 0 else base_predict("xor_0_2", world)
    if kind == "package_physical":
        return base_predict("hybrid_and_xor", world) if world[7] == 0 else base_predict("maj3_0_2_5", world)
    raise ValueError(f"Unknown package kind: {kind}")


def softmax(scores: Sequence[float]) -> list[float]:
    max_score = max(scores)
    exponentials = [math.exp(score - max_score) for score in scores]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def family_rank(family: str) -> int:
    order = {
        "hybrid": 0,
        "novel_pair": 1,
        "novel_triple": 2,
    }
    return order[family]


def top_rule_index(scores: Sequence[float], candidates: Sequence[RuleCandidate]) -> int:
    return max(
        range(len(scores)),
        key=lambda index: (scores[index], -family_rank(candidates[index].family), candidates[index].name),
    )


def top_rule(scores: Sequence[float], candidates: Sequence[RuleCandidate]) -> RuleCandidate:
    return candidates[top_rule_index(scores, candidates)]


def update_rule_scores(
    scores: list[float],
    candidates: Sequence[RuleCandidate],
    world: World,
    actual: int,
    consistency: float,
) -> None:
    for index, candidate in enumerate(candidates):
        likelihood = consistency if base_predict(candidate.kind, world) == actual else (1.0 - consistency)
        scores[index] += math.log(likelihood)


def initial_rule_scores(candidates: Sequence[RuleCandidate], priors: dict[str, float]) -> list[float]:
    return [math.log(priors[candidate.name]) for candidate in candidates]


def rules_truth_table(feature_count: int) -> list[World]:
    return [
        build_world(feature_count, feature_0, feature_2, feature_5)
        for feature_0 in (0, 1)
        for feature_2 in (0, 1)
        for feature_5 in (0, 1)
    ]


def packages_truth_table(feature_count: int) -> list[World]:
    return [
        build_world(feature_count, feature_0, feature_2, feature_5, feature_7=feature_7)
        for feature_0 in (0, 1)
        for feature_2 in (0, 1)
        for feature_5 in (0, 1)
        for feature_7 in (0, 1)
    ]


def ambiguous_rule_observations(feature_count: int) -> list[World]:
    return [
        build_world(feature_count, 0, 0, 1),
        build_world(feature_count, 0, 1, 1),
        build_world(feature_count, 1, 0, 1),
    ]


def package_observations(feature_count: int) -> list[World]:
    return [
        build_world(feature_count, 0, 0, 1, feature_7=0),
        build_world(feature_count, 0, 1, 1, feature_7=0),
        build_world(feature_count, 1, 0, 1, feature_7=0),
    ]


def stage1_family_routing(config: HierarchyBatchConfig) -> list[dict[str, object]]:
    candidates = [
        RuleCandidate("gate_f2(and->xor)", "hybrid", "hybrid", "hybrid_and_xor"),
        RuleCandidate("or_0_2", "novel_pair", "pair", "or_0_2"),
        RuleCandidate("maj3_0_2_5", "novel_triple", "triple", "maj3_0_2_5"),
    ]
    priors = {
        "gate_f2(and->xor)": 0.70,
        "or_0_2": 0.20,
        "maj3_0_2_5": 0.10,
    }
    observations = ambiguous_rule_observations(config.feature_count)
    family_probe = build_world(config.feature_count, 0, 1, 0)
    truth_worlds = rules_truth_table(config.feature_count)
    rows: list[dict[str, object]] = []

    for policy_name in ("flat_rule", "family_router"):
        scores = initial_rule_scores(candidates, priors)
        for world in observations:
            update_rule_scores(scores, candidates, world, base_predict("or_0_2", world), config.consistency)
        if policy_name == "family_router":
            update_rule_scores(scores, candidates, family_probe, base_predict("or_0_2", family_probe), config.consistency)
        selected = top_rule(scores, candidates)
        table_accuracy = statistics.mean(
            1 if base_predict(selected.kind, world) == base_predict("or_0_2", world) else 0
            for world in truth_worlds
        )
        rows.append(
            {
                "policy": policy_name,
                "Probe": probe_label(family_probe) if policy_name == "family_router" else "-",
                "Family": selected.family,
                "Rule": selected.name,
                "Exact": 1 if selected.name == "or_0_2" else 0,
                "TableAcc": round(table_accuracy, 3),
            }
        )
    return rows


def stage2_subfamily_routing(config: HierarchyBatchConfig) -> list[dict[str, object]]:
    candidates = [
        RuleCandidate("or_0_2", "novel_pair", "pair", "or_0_2"),
        RuleCandidate("maj3_0_2_5", "novel_triple", "triple", "maj3_0_2_5"),
    ]
    priors = {
        "or_0_2": 0.35,
        "maj3_0_2_5": 0.65,
    }
    observations = ambiguous_rule_observations(config.feature_count)
    subfamily_probe = build_world(config.feature_count, 0, 1, 0)
    truth_worlds = rules_truth_table(config.feature_count)
    rows: list[dict[str, object]] = []

    for policy_name in ("jump_top_subfamily", "subfamily_router"):
        scores = initial_rule_scores(candidates, priors)
        for world in observations:
            update_rule_scores(scores, candidates, world, base_predict("or_0_2", world), config.consistency)
        if policy_name == "subfamily_router":
            update_rule_scores(scores, candidates, subfamily_probe, base_predict("or_0_2", subfamily_probe), config.consistency)
        selected = top_rule(scores, candidates)
        table_accuracy = statistics.mean(
            1 if base_predict(selected.kind, world) == base_predict("or_0_2", world) else 0
            for world in truth_worlds
        )
        rows.append(
            {
                "policy": policy_name,
                "Probe": probe_label(subfamily_probe) if policy_name == "subfamily_router" else "-",
                "Subfamily": selected.subfamily,
                "Rule": selected.name,
                "Exact": 1 if selected.name == "or_0_2" else 0,
                "TableAcc": round(table_accuracy, 3),
            }
        )
    return rows


def stage3_exact_rule_routing(config: HierarchyBatchConfig) -> list[dict[str, object]]:
    candidates = [
        RuleCandidate("or_0_2", "novel_pair", "pair", "or_0_2"),
        RuleCandidate("xor_0_2", "novel_pair", "pair", "xor_0_2"),
    ]
    priors = {
        "or_0_2": 0.40,
        "xor_0_2": 0.60,
    }
    observations = ambiguous_rule_observations(config.feature_count)
    rule_probe = build_world(config.feature_count, 1, 1, 0)
    truth_worlds = rules_truth_table(config.feature_count)
    rows: list[dict[str, object]] = []

    for policy_name in ("jump_top_rule", "rule_router"):
        scores = initial_rule_scores(candidates, priors)
        for world in observations:
            update_rule_scores(scores, candidates, world, base_predict("or_0_2", world), config.consistency)
        if policy_name == "rule_router":
            update_rule_scores(scores, candidates, rule_probe, base_predict("or_0_2", rule_probe), config.consistency)
        selected = top_rule(scores, candidates)
        table_accuracy = statistics.mean(
            1 if base_predict(selected.kind, world) == base_predict("or_0_2", world) else 0
            for world in truth_worlds
        )
        rows.append(
            {
                "policy": policy_name,
                "Probe": probe_label(rule_probe) if policy_name == "rule_router" else "-",
                "Rule": selected.name,
                "Exact": 1 if selected.name == "or_0_2" else 0,
                "TableAcc": round(table_accuracy, 3),
            }
        )
    return rows


def stage4_archive_package_routing(config: HierarchyBatchConfig) -> list[dict[str, object]]:
    package_candidates = [
        PackageCandidate("package_social", "package_social"),
        PackageCandidate("package_physical", "package_physical"),
    ]
    observations = package_observations(config.feature_count)
    package_probe = build_world(config.feature_count, 1, 1, 0, feature_7=1)
    truth_worlds = packages_truth_table(config.feature_count)
    rows: list[dict[str, object]] = []

    for policy_name in ("single_rule_router", "package_router"):
        if policy_name == "single_rule_router":
            single_rule_accuracy = statistics.mean(
                1 if base_predict("or_0_2", world) == package_predict("package_social", world) else 0
                for world in truth_worlds
            )
            rows.append(
                {
                    "policy": policy_name,
                    "Probe": "-",
                    "Selected": "or_0_2",
                    "ExactPkg": 0,
                    "TableAcc": round(single_rule_accuracy, 3),
                }
            )
            continue

        best_package = max(
            package_candidates,
            key=lambda candidate: statistics.mean(
                1
                if package_predict(candidate.kind, world) == package_predict("package_social", world)
                else 0
                for world in observations + [package_probe]
            ),
        )
        table_accuracy = statistics.mean(
            1 if package_predict(best_package.kind, world) == package_predict("package_social", world) else 0
            for world in truth_worlds
        )
        rows.append(
            {
                "policy": policy_name,
                "Probe": probe_label(package_probe),
                "Selected": best_package.name,
                "ExactPkg": 1 if best_package.name == "package_social" else 0,
                "TableAcc": round(table_accuracy, 3),
            }
        )
    return rows


def stage5_world_mode_routing(config: HierarchyBatchConfig) -> list[dict[str, object]]:
    return_worlds = packages_truth_table(config.feature_count)
    rng = random.Random(config.seed)
    rng.shuffle(return_worlds)
    mismatch_prefix = [
        build_world(config.feature_count, 1, 1, 0, feature_7=1),
        build_world(config.feature_count, 1, 1, 1, feature_7=1),
        build_world(config.feature_count, 0, 1, 1, feature_7=1),
        build_world(config.feature_count, 1, 0, 1, feature_7=1),
    ]
    prefix_labels = {probe_label(world) for world in mismatch_prefix}
    return_worlds = mismatch_prefix + [
        world for world in return_worlds if probe_label(world) not in prefix_labels
    ]

    cue_to_package = {
        0: "package_social",
        1: "package_physical",
    }
    rows: list[dict[str, object]] = []

    for policy_name in ("latest_package", "mode_router"):
        if policy_name == "latest_package":
            active_package = "package_physical"
            first_four: list[int] = []
            overall: list[int] = []
            switched = False
            for index, world in enumerate(return_worlds):
                cue_world = list(world)
                cue_world[6] = 0
                world_with_cue = tuple(cue_world)
                actual = package_predict("package_social", world_with_cue)
                prediction = package_predict(active_package, world_with_cue)
                correct = 1 if prediction == actual else 0
                overall.append(correct)
                if index < 4:
                    first_four.append(correct)
                if not switched and correct == 0:
                    active_package = "package_social"
                    switched = True
            rows.append(
                {
                    "policy": policy_name,
                    "Selected": active_package,
                    "First4": round(statistics.mean(first_four), 3),
                    "ReturnAll": round(statistics.mean(overall), 3),
                }
            )
        else:
            active_package = cue_to_package[0]
            first_four = []
            overall = []
            for index, world in enumerate(return_worlds):
                cue_world = list(world)
                cue_world[6] = 0
                world_with_cue = tuple(cue_world)
                actual = package_predict("package_social", world_with_cue)
                prediction = package_predict(active_package, world_with_cue)
                correct = 1 if prediction == actual else 0
                overall.append(correct)
                if index < 4:
                    first_four.append(correct)
            rows.append(
                {
                    "policy": policy_name,
                    "Selected": active_package,
                    "First4": round(statistics.mean(first_four), 3),
                    "ReturnAll": round(statistics.mean(overall), 3),
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
    print(f"Hierarchy batch seed: {config.seed}")
    print(f"Consistency: {config.consistency:.2f}")
    print()

    print_stage("Stage 1: family routing", stage1_family_routing(config))
    print_stage("Stage 2: subfamily routing", stage2_subfamily_routing(config))
    print_stage("Stage 3: exact rule routing", stage3_exact_rule_routing(config))
    print_stage("Stage 4: archive package routing", stage4_archive_package_routing(config))
    print_stage("Stage 5: world mode routing", stage5_world_mode_routing(config))


if __name__ == "__main__":
    main()
