from __future__ import annotations

import argparse
import math
import random
import statistics
from dataclasses import dataclass
from typing import Sequence


World = tuple[int, ...]


@dataclass(frozen=True)
class BatchConfig:
    feature_count: int = 8
    seed: int = 7
    consistency: float = 0.8


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    kind: str


def parse_args() -> BatchConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Run five consecutive class-switch experiments: value-gated diagnostics, "
            "probe selection, suppression of unnecessary diagnostics, risk-sensitive "
            "pre-probing, and routing inside the novel family."
        )
    )
    parser.add_argument("--feature-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--consistency", type=float, default=0.8)
    args = parser.parse_args()

    config = BatchConfig(
        feature_count=args.feature_count,
        seed=args.seed,
        consistency=args.consistency,
    )
    validate_config(config)
    return config


def validate_config(config: BatchConfig) -> None:
    if config.feature_count != 8:
        raise ValueError("belief_class_switch_batch_suite currently expects feature_count == 8")
    if not 0.5 < config.consistency < 1.0:
        raise ValueError("consistency must be in (0.5, 1.0)")


def build_world(feature_count: int, feature_0: int, feature_2: int, feature_5: int) -> World:
    values = [0] * feature_count
    values[0] = feature_0
    values[2] = feature_2
    values[5] = feature_5
    return tuple(values)


def truth_table(feature_count: int) -> list[World]:
    return [
        build_world(feature_count, feature_0, feature_2, feature_5)
        for feature_0 in (0, 1)
        for feature_2 in (0, 1)
        for feature_5 in (0, 1)
    ]


def probe_label(world: World) -> str:
    return f"{world[0]}{world[2]}{world[5]}"


def predict(kind: str, world: World) -> int:
    feature_0 = world[0]
    feature_2 = world[2]
    feature_5 = world[5]

    if kind == "hybrid_and_xor":
        if feature_2 == 0:
            return feature_0 & feature_5
        return feature_0 ^ feature_5
    if kind == "or_0_2":
        return feature_0 | feature_2
    if kind == "maj3_0_2_5":
        return 1 if (feature_0 + feature_2 + feature_5) >= 2 else 0
    if kind == "nand3_0_2_5":
        return 1 - (feature_0 & feature_2 & feature_5)
    raise ValueError(f"Unknown kind: {kind}")


def family_rank(family: str) -> int:
    if family == "hybrid":
        return 0
    if family == "novel":
        return 1
    raise ValueError(f"Unknown family: {family}")


def softmax(scores: Sequence[float]) -> list[float]:
    max_score = max(scores)
    exponentials = [math.exp(score - max_score) for score in scores]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def top_index(scores: Sequence[float], candidates: Sequence[Candidate]) -> int:
    return max(
        range(len(scores)),
        key=lambda index: (scores[index], -family_rank(candidates[index].family), candidates[index].name),
    )


def top_candidate(scores: Sequence[float], candidates: Sequence[Candidate]) -> Candidate:
    return candidates[top_index(scores, candidates)]


def top_novel_candidate(scores: Sequence[float], candidates: Sequence[Candidate]) -> Candidate:
    novel_indices = [index for index, item in enumerate(candidates) if item.family == "novel"]
    best_index = max(
        novel_indices,
        key=lambda index: (scores[index], candidates[index].name),
    )
    return candidates[best_index]


def update_scores(
    scores: list[float],
    candidates: Sequence[Candidate],
    world: World,
    observed_action: int,
    consistency: float,
) -> None:
    for index, candidate in enumerate(candidates):
        likelihood = consistency if predict(candidate.kind, world) == observed_action else (1.0 - consistency)
        scores[index] += math.log(likelihood)


def initial_scores(candidates: Sequence[Candidate], priors: dict[str, float]) -> list[float]:
    return [math.log(priors[candidate.name]) for candidate in candidates]


def choose_best_probe(
    scores: Sequence[float],
    candidates: Sequence[Candidate],
    available_worlds: Sequence[World],
    mode: str,
) -> World:
    if not available_worlds:
        raise ValueError("No available worlds for probing")

    if mode == "blind":
        return sorted(available_worlds, key=probe_label)[0]

    if mode == "hybrid_vs_novel":
        current = top_candidate(scores, candidates)
        novel = top_novel_candidate(scores, candidates)
        best_world = available_worlds[0]
        best_score = -1.0
        for world in sorted(available_worlds, key=probe_label):
            separation = abs(predict(current.kind, world) - predict(novel.kind, world))
            if separation > best_score:
                best_score = separation
                best_world = world
        return best_world

    if mode == "novel_vs_novel":
        novel_indices = [index for index, item in enumerate(candidates) if item.family == "novel"]
        sorted_novels = sorted(
            novel_indices,
            key=lambda index: (scores[index], candidates[index].name),
            reverse=True,
        )
        best = candidates[sorted_novels[0]]
        second = candidates[sorted_novels[1]]
        best_world = available_worlds[0]
        best_score = -1.0
        for world in sorted(available_worlds, key=probe_label):
            separation = abs(predict(best.kind, world) - predict(second.kind, world))
            if separation > best_score:
                best_score = separation
                best_world = world
        return best_world

    raise ValueError(f"Unknown probe mode: {mode}")


def worlds_without(observed: Sequence[World], feature_count: int) -> list[World]:
    seen = {probe_label(world) for world in observed}
    return [world for world in truth_table(feature_count) if probe_label(world) not in seen]


def ambiguous_worlds(feature_count: int) -> list[World]:
    return [
        build_world(feature_count, 0, 0, 1),
        build_world(feature_count, 0, 1, 1),
        build_world(feature_count, 1, 0, 1),
    ]


def hybrid_rule(world: World) -> int:
    return predict("hybrid_and_xor", world)


def novel_or_rule(world: World) -> int:
    return predict("or_0_2", world)


def stage1_value_gated_diagnostics(config: BatchConfig) -> list[dict[str, object]]:
    candidates = [
        Candidate(name="gate_f2(and->xor)", family="hybrid", kind="hybrid_and_xor"),
        Candidate(name="or_0_2", family="novel", kind="or_0_2"),
    ]
    priors = {
        "gate_f2(and->xor)": 0.85,
        "or_0_2": 0.15,
    }
    initial_worlds = ambiguous_worlds(config.feature_count)
    warmup_worlds = initial_worlds * 2
    reveal_worlds = [build_world(config.feature_count, 1, 1, 1)] * 4
    hold_worlds = truth_table(config.feature_count) * 2
    diagnostic_probe = build_world(config.feature_count, 0, 1, 0)
    rows: list[dict[str, object]] = []

    for label, question_cost in (("low_cost", 0.40), ("high_cost", 1.30)):
        for policy_name in ("never", "always", "value"):
            scores = initial_scores(candidates, priors)
            for world in initial_worlds:
                update_scores(scores, candidates, world, novel_or_rule(world), config.consistency)
            selected = top_candidate(scores, candidates)
            observed = list(initial_worlds)
            questions_asked = 0
            correct_count = 0
            total_worlds = warmup_worlds + reveal_worlds + hold_worlds

            for step_index, world in enumerate(total_worlds):
                actual = novel_or_rule(world)
                correct_count += 1 if predict(selected.kind, world) == actual else 0
                update_scores(scores, candidates, world, actual, config.consistency)
                observed.append(world)

                if (
                    step_index >= len(warmup_worlds)
                    and step_index < len(warmup_worlds) + len(reveal_worlds)
                    and questions_asked == 0
                    and predict(selected.kind, world) != actual
                ):
                    should_ask = False
                    if policy_name == "always":
                        should_ask = True
                    elif policy_name == "value":
                        expected_saved_errors = 1.0
                        should_ask = expected_saved_errors > question_cost
                    if should_ask:
                        questions_asked = 1
                        update_scores(
                            scores,
                            candidates,
                            diagnostic_probe,
                            novel_or_rule(diagnostic_probe),
                            config.consistency,
                        )
                        observed.append(diagnostic_probe)

                selected = top_candidate(scores, candidates)

            net_value = correct_count - (question_cost * questions_asked)
            rows.append(
                {
                    "condition": label,
                    "policy": policy_name,
                    "QCost": question_cost,
                    "Ask": questions_asked,
                    "Correct": correct_count,
                    "NetV": round(net_value, 3),
                }
            )
    return rows


def stage2_probe_selection(config: BatchConfig) -> list[dict[str, object]]:
    candidates = [
        Candidate(name="gate_f2(and->xor)", family="hybrid", kind="hybrid_and_xor"),
        Candidate(name="or_0_2", family="novel", kind="or_0_2"),
    ]
    priors = {
        "gate_f2(and->xor)": 0.85,
        "or_0_2": 0.15,
    }
    initial_worlds = ambiguous_worlds(config.feature_count)
    warmup_worlds = initial_worlds * 2
    reveal_worlds = [build_world(config.feature_count, 1, 1, 1)] * 4
    hold_worlds = truth_table(config.feature_count) * 2
    rows: list[dict[str, object]] = []

    for policy_name, probe_mode in (
        ("no_probe", None),
        ("blind_probe", "blind"),
        ("best_probe", "hybrid_vs_novel"),
    ):
        scores = initial_scores(candidates, priors)
        for world in initial_worlds:
            update_scores(scores, candidates, world, novel_or_rule(world), config.consistency)
        selected = top_candidate(scores, candidates)
        observed = list(initial_worlds)
        questions_asked = 0
        first_novel_step: int | None = None
        accuracies: list[int] = []
        probe_used = "-"

        for step_index, world in enumerate(warmup_worlds + reveal_worlds + hold_worlds):
            actual = novel_or_rule(world)
            accuracies.append(1 if predict(selected.kind, world) == actual else 0)
            update_scores(scores, candidates, world, actual, config.consistency)
            observed.append(world)

            if (
                probe_mode is not None
                and step_index >= len(warmup_worlds)
                and step_index < len(warmup_worlds) + len(reveal_worlds)
                and questions_asked == 0
                and predict(selected.kind, world) != actual
            ):
                probe = choose_best_probe(
                    scores,
                    candidates,
                    worlds_without(observed, config.feature_count),
                    probe_mode,
                )
                questions_asked = 1
                update_scores(scores, candidates, probe, novel_or_rule(probe), config.consistency)
                observed.append(probe)
                probe_used = probe_label(probe)

            selected = top_candidate(scores, candidates)
            if first_novel_step is None and selected.family == "novel":
                first_novel_step = step_index

        rows.append(
            {
                "policy": policy_name,
                "Probe": probe_used if questions_asked else "-",
                "NovelAt": first_novel_step if first_novel_step is not None else "-",
                "Reveal": round(
                    statistics.mean(
                        accuracies[len(warmup_worlds): len(warmup_worlds) + len(reveal_worlds)]
                    ),
                    3,
                ),
                "Overall": round(statistics.mean(accuracies), 3),
            }
        )
    return rows


def stage3_suppress_unnecessary_diagnostics(config: BatchConfig) -> list[dict[str, object]]:
    candidates = [
        Candidate(name="gate_f2(and->xor)", family="hybrid", kind="hybrid_and_xor"),
        Candidate(name="or_0_2", family="novel", kind="or_0_2"),
    ]
    priors = {
        "gate_f2(and->xor)": 0.85,
        "or_0_2": 0.15,
    }
    initial_worlds = ambiguous_worlds(config.feature_count)
    action_worlds = truth_table(config.feature_count) * 3
    noisy_world = build_world(config.feature_count, 1, 1, 1)
    diagnostic_probe = build_world(config.feature_count, 0, 1, 0)
    question_cost = 0.40
    rows: list[dict[str, object]] = []

    for policy_name in ("always_diagnose", "suppress_singleton"):
        scores = initial_scores(candidates, priors)
        for world in initial_worlds:
            update_scores(scores, candidates, world, hybrid_rule(world), config.consistency)
        selected = top_candidate(scores, candidates)
        questions_asked = 0
        mismatch_count = 0
        correct_count = 0

        actual = hybrid_rule(noisy_world)
        observed_action = 1 - actual
        correct_count += 1 if predict(selected.kind, noisy_world) == actual else 0
        update_scores(scores, candidates, noisy_world, observed_action, config.consistency)
        if predict(selected.kind, noisy_world) != observed_action:
            mismatch_count += 1

        if policy_name == "always_diagnose" and mismatch_count >= 1:
            questions_asked = 1
            update_scores(scores, candidates, diagnostic_probe, hybrid_rule(diagnostic_probe), config.consistency)
        elif policy_name == "suppress_singleton" and mismatch_count >= 2:
            questions_asked = 1
            update_scores(scores, candidates, diagnostic_probe, hybrid_rule(diagnostic_probe), config.consistency)

        selected = top_candidate(scores, candidates)
        hold_accuracies = []
        for world in action_worlds:
            actual = hybrid_rule(world)
            hold_accuracies.append(1 if predict(selected.kind, world) == actual else 0)

        net_value = correct_count + sum(hold_accuracies) - (question_cost * questions_asked)
        rows.append(
            {
                "policy": policy_name,
                "Ask": questions_asked,
                "Final": selected.name,
                "Hold": round(statistics.mean(hold_accuracies), 3),
                "NetV": round(net_value, 3),
            }
        )
    return rows


def stage4_risk_sensitive_preprobe(config: BatchConfig) -> list[dict[str, object]]:
    candidates = [
        Candidate(name="gate_f2(and->xor)", family="hybrid", kind="hybrid_and_xor"),
        Candidate(name="or_0_2", family="novel", kind="or_0_2"),
    ]
    priors = {
        "gate_f2(and->xor)": 0.75,
        "or_0_2": 0.25,
    }
    initial_worlds = ambiguous_worlds(config.feature_count)
    high_risk_world = build_world(config.feature_count, 1, 1, 1)
    hold_worlds = truth_table(config.feature_count) * 2
    diagnostic_probe = build_world(config.feature_count, 0, 1, 0)
    question_cost = 1.20
    high_risk_value = 3.0
    rows: list[dict[str, object]] = []

    for policy_name in ("no_preprobe", "uniform_value_preprobe", "risk_sensitive_preprobe"):
        scores = initial_scores(candidates, priors)
        for world in initial_worlds:
            update_scores(scores, candidates, world, novel_or_rule(world), config.consistency)
        selected = top_candidate(scores, candidates)
        questions_asked = 0

        if policy_name != "no_preprobe":
            should_ask = False
            if policy_name == "uniform_value_preprobe":
                should_ask = 1.0 > question_cost
            elif policy_name == "risk_sensitive_preprobe":
                should_ask = high_risk_value > question_cost
            if should_ask:
                questions_asked = 1
                update_scores(scores, candidates, diagnostic_probe, novel_or_rule(diagnostic_probe), config.consistency)
                selected = top_candidate(scores, candidates)

        reveal_actual = novel_or_rule(high_risk_world)
        reveal_correct = 1 if predict(selected.kind, high_risk_world) == reveal_actual else 0
        weighted_gain = high_risk_value if reveal_correct else 0.0
        update_scores(scores, candidates, high_risk_world, reveal_actual, config.consistency)
        selected = top_candidate(scores, candidates)

        hold_scores = [
            1 if predict(selected.kind, world) == novel_or_rule(world) else 0
            for world in hold_worlds
        ]
        net_utility = weighted_gain + sum(hold_scores) - (question_cost * questions_asked)
        rows.append(
            {
                "policy": policy_name,
                "Ask": questions_asked,
                "RevealGain": weighted_gain,
                "Final": selected.name,
                "Hold": round(statistics.mean(hold_scores), 3),
                "NetU": round(net_utility, 3),
            }
        )
    return rows


def stage5_route_inside_novel_family(config: BatchConfig) -> list[dict[str, object]]:
    candidates = [
        Candidate(name="gate_f2(and->xor)", family="hybrid", kind="hybrid_and_xor"),
        Candidate(name="or_0_2", family="novel", kind="or_0_2"),
        Candidate(name="maj3_0_2_5", family="novel", kind="maj3_0_2_5"),
    ]
    priors = {
        "gate_f2(and->xor)": 0.60,
        "or_0_2": 0.10,
        "maj3_0_2_5": 0.30,
    }
    initial_worlds = ambiguous_worlds(config.feature_count)
    reveal_world = build_world(config.feature_count, 1, 1, 1)
    hold_worlds = truth_table(config.feature_count) * 3
    rows: list[dict[str, object]] = []

    for policy_name in ("jump_top_novel", "route_top_novels"):
        scores = initial_scores(candidates, priors)
        for world in initial_worlds:
            update_scores(scores, candidates, world, novel_or_rule(world), config.consistency)
        update_scores(scores, candidates, reveal_world, novel_or_rule(reveal_world), config.consistency)
        observed = list(initial_worlds) + [reveal_world]

        current_top = top_candidate(scores, candidates)
        if policy_name == "jump_top_novel":
            selected = top_novel_candidate(scores, candidates)
            probe_used = "-"
        else:
            probe = choose_best_probe(
                scores,
                candidates,
                worlds_without(observed, config.feature_count),
                "novel_vs_novel",
            )
            update_scores(scores, candidates, probe, novel_or_rule(probe), config.consistency)
            selected = top_novel_candidate(scores, candidates)
            probe_used = probe_label(probe)

        hold_accuracies = [
            1 if predict(selected.kind, world) == novel_or_rule(world) else 0
            for world in hold_worlds
        ]
        rows.append(
            {
                "policy": policy_name,
                "TopAfterReveal": current_top.name,
                "Probe": probe_used,
                "Final": selected.name,
                "Hold": round(statistics.mean(hold_accuracies), 3),
                "Overall": round((1 + sum(hold_accuracies)) / (1 + len(hold_accuracies)), 3),
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
    print(f"Batch suite seed: {config.seed}")
    print(f"Consistency: {config.consistency:.2f}")
    print()

    print_stage(
        "Stage 1: value-gated diagnostics",
        stage1_value_gated_diagnostics(config),
    )
    print_stage(
        "Stage 2: selecting the best diagnostic probe",
        stage2_probe_selection(config),
    )
    print_stage(
        "Stage 3: suppressing unnecessary diagnostics in a stable world",
        stage3_suppress_unnecessary_diagnostics(config),
    )
    print_stage(
        "Stage 4: risk-sensitive pre-probing",
        stage4_risk_sensitive_preprobe(config),
    )
    print_stage(
        "Stage 5: routing inside the novel family",
        stage5_route_inside_novel_family(config),
    )


if __name__ == "__main__":
    main()
