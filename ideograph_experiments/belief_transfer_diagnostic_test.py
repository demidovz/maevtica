from __future__ import annotations

import argparse
import math
import random
import statistics
from dataclasses import dataclass
from typing import Sequence


World = tuple[int, ...]


@dataclass(frozen=True)
class BeliefTransferDiagnosticConfig:
    feature_count: int = 8
    pretrain_episodes: int = 150
    base_conflict_episodes: int = 120
    target_conflict_episodes: int = 40
    seed: int = 7
    consistency: float = 0.92
    spawn_after: int = 12
    spawn_window: int = 8
    spawn_threshold: float = 0.72
    transfer_bias: float = 6.0
    reactive_probe_episodes: int = 6
    diagnostic_questions: int = 1
    applicability_prior_alpha: float = 1.0
    applicability_prior_beta: float = 1.0
    applicability_threshold: float = 0.65


@dataclass(frozen=True)
class Hypothesis:
    name: str
    kind: str
    feature_a: int
    feature_b: int | None = None

    def predict(self, world: World) -> int:
        if self.kind == "feature":
            return world[self.feature_a]
        if self.kind == "or":
            return world[self.feature_a] | world[self.feature_b]
        if self.kind == "and":
            return world[self.feature_a] & world[self.feature_b]
        if self.kind == "xor":
            return world[self.feature_a] ^ world[self.feature_b]
        if self.kind == "xnor":
            return 1 - (world[self.feature_a] ^ world[self.feature_b])
        raise ValueError(f"Unknown hypothesis kind: {self.kind}")


@dataclass(frozen=True)
class PretrainResult:
    source_name: str
    top_hypothesis: str
    top_mass: float


@dataclass(frozen=True)
class BaseConflictResult:
    overall_accuracy: float
    early_accuracy: float
    late_accuracy: float
    final_top_hypothesis: str
    final_top_mass: float
    schema_kind: str
    spawn_episode: int | None


@dataclass(frozen=True)
class TargetResult:
    target_name: str
    policy_name: str
    overall_accuracy: float
    first_ten_accuracy: float
    last_ten_accuracy: float
    first_action_accuracy: float
    final_top_hypothesis: str
    spawn_episode: int | None
    gate_status: str
    gate_phase: str
    gate_decision_index: int | None
    diagnostic_questions: int
    applicability_mean: float | None
    action_error_before_gate: bool


def parse_args() -> BeliefTransferDiagnosticConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether diagnostic questions can reject a bad transferred schema "
            "before the first action error."
        )
    )
    parser.add_argument("--feature-count", type=int, default=8)
    parser.add_argument("--pretrain-episodes", type=int, default=150)
    parser.add_argument("--base-conflict-episodes", type=int, default=120)
    parser.add_argument("--target-conflict-episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--consistency", type=float, default=0.92)
    parser.add_argument("--spawn-after", type=int, default=12)
    parser.add_argument("--spawn-window", type=int, default=8)
    parser.add_argument("--spawn-threshold", type=float, default=0.72)
    parser.add_argument("--transfer-bias", type=float, default=6.0)
    parser.add_argument("--reactive-probe-episodes", type=int, default=6)
    parser.add_argument("--diagnostic-questions", type=int, default=1)
    parser.add_argument("--applicability-prior-alpha", type=float, default=1.0)
    parser.add_argument("--applicability-prior-beta", type=float, default=1.0)
    parser.add_argument("--applicability-threshold", type=float, default=0.65)
    args = parser.parse_args()

    config = BeliefTransferDiagnosticConfig(
        feature_count=args.feature_count,
        pretrain_episodes=args.pretrain_episodes,
        base_conflict_episodes=args.base_conflict_episodes,
        target_conflict_episodes=args.target_conflict_episodes,
        seed=args.seed,
        consistency=args.consistency,
        spawn_after=args.spawn_after,
        spawn_window=args.spawn_window,
        spawn_threshold=args.spawn_threshold,
        transfer_bias=args.transfer_bias,
        reactive_probe_episodes=args.reactive_probe_episodes,
        diagnostic_questions=args.diagnostic_questions,
        applicability_prior_alpha=args.applicability_prior_alpha,
        applicability_prior_beta=args.applicability_prior_beta,
        applicability_threshold=args.applicability_threshold,
    )
    validate_config(config)
    return config


def validate_config(config: BeliefTransferDiagnosticConfig) -> None:
    if config.feature_count != 8:
        raise ValueError("belief_transfer_diagnostic_test currently expects feature_count == 8")
    if config.pretrain_episodes <= 0:
        raise ValueError("pretrain_episodes must be > 0")
    if config.base_conflict_episodes <= 0 or config.target_conflict_episodes <= 0:
        raise ValueError("conflict episode counts must be > 0")
    if not 0.5 < config.consistency < 1.0:
        raise ValueError("consistency must be in (0.5, 1.0)")
    if config.spawn_after < 0:
        raise ValueError("spawn_after must be >= 0")
    if config.spawn_window <= 0:
        raise ValueError("spawn_window must be > 0")
    if not 0.0 < config.spawn_threshold < 1.0:
        raise ValueError("spawn_threshold must be in (0, 1)")
    if config.transfer_bias <= 0:
        raise ValueError("transfer_bias must be > 0")
    if config.reactive_probe_episodes <= 0:
        raise ValueError("reactive_probe_episodes must be > 0")
    if config.diagnostic_questions <= 0:
        raise ValueError("diagnostic_questions must be > 0")
    if config.applicability_prior_alpha <= 0 or config.applicability_prior_beta <= 0:
        raise ValueError("applicability priors must be > 0")
    if not 0.0 < config.applicability_threshold < 1.0:
        raise ValueError("applicability_threshold must be in (0, 1)")


def softmax(scores: Sequence[float]) -> list[float]:
    max_score = max(scores)
    exponentials = [math.exp(score - max_score) for score in scores]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def build_feature_hypotheses(feature_count: int) -> list[Hypothesis]:
    return [
        Hypothesis(name=f"feat_{feature_index}", kind="feature", feature_a=feature_index)
        for feature_index in range(feature_count)
    ]


def build_pair_composites(feature_a: int, feature_b: int) -> list[Hypothesis]:
    return [
        Hypothesis(name=f"or_{feature_a}_{feature_b}", kind="or", feature_a=feature_a, feature_b=feature_b),
        Hypothesis(name=f"and_{feature_a}_{feature_b}", kind="and", feature_a=feature_a, feature_b=feature_b),
        Hypothesis(name=f"xor_{feature_a}_{feature_b}", kind="xor", feature_a=feature_a, feature_b=feature_b),
        Hypothesis(name=f"xnor_{feature_a}_{feature_b}", kind="xnor", feature_a=feature_a, feature_b=feature_b),
    ]


def sample_worlds(rng: random.Random, feature_count: int, count: int) -> list[World]:
    return [
        tuple(1 if rng.random() < 0.5 else 0 for _ in range(feature_count))
        for _ in range(count)
    ]


def update_scores(
    scores: list[float],
    hypotheses: Sequence[Hypothesis],
    world: World,
    true_action: int,
    consistency: float,
) -> None:
    for index, hypothesis in enumerate(hypotheses):
        likelihood = consistency if hypothesis.predict(world) == true_action else (1.0 - consistency)
        scores[index] += math.log(likelihood)


def target_value(operator: str, world: World, feature_a: int, feature_b: int) -> int:
    if operator == "xor":
        return world[feature_a] ^ world[feature_b]
    if operator == "and":
        return world[feature_a] & world[feature_b]
    if operator == "or":
        return world[feature_a] | world[feature_b]
    if operator == "xnor":
        return 1 - (world[feature_a] ^ world[feature_b])
    raise ValueError(f"Unknown operator: {operator}")


def top_hypothesis(scores: Sequence[float], hypotheses: Sequence[Hypothesis]) -> tuple[str, float]:
    belief = softmax(scores)
    top_index = max(range(len(scores)), key=scores.__getitem__)
    return hypotheses[top_index].name, belief[top_index]


def pretrain_source(
    config: BeliefTransferDiagnosticConfig,
    worlds: Sequence[World],
    target_feature: int,
    source_name: str,
) -> tuple[PretrainResult, Hypothesis]:
    hypotheses = build_feature_hypotheses(config.feature_count)
    scores = [0.0] * len(hypotheses)

    for world in worlds:
        update_scores(scores, hypotheses, world, world[target_feature], config.consistency)

    top_name, top_mass = top_hypothesis(scores, hypotheses)
    top_index = max(range(len(scores)), key=scores.__getitem__)
    return (
        PretrainResult(
            source_name=source_name,
            top_hypothesis=top_name,
            top_mass=top_mass,
        ),
        hypotheses[top_index],
    )


def run_base_conflict(
    config: BeliefTransferDiagnosticConfig,
    source_hypotheses: Sequence[Hypothesis],
    worlds: Sequence[World],
) -> BaseConflictResult:
    hypotheses = list(source_hypotheses)
    scores = [0.0] * len(hypotheses)
    accuracies: list[int] = []
    recent_accuracies: list[int] = []
    spawn_episode: int | None = None

    for episode_index, world in enumerate(worlds):
        true_action = target_value("xor", world, 0, 4)
        belief = softmax(scores)
        probability_action_one = sum(
            weight * hypothesis.predict(world)
            for weight, hypothesis in zip(belief, hypotheses)
        )
        predicted_action = 1 if probability_action_one >= 0.5 else 0
        is_correct = 1 if predicted_action == true_action else 0
        accuracies.append(is_correct)
        recent_accuracies.append(is_correct)
        if len(recent_accuracies) > config.spawn_window:
            recent_accuracies.pop(0)

        if (
            spawn_episode is None
            and episode_index >= config.spawn_after
            and statistics.mean(recent_accuracies) < config.spawn_threshold
        ):
            hypotheses.extend(build_pair_composites(0, 4))
            scores.extend([0.0] * 4)
            spawn_episode = episode_index

        update_scores(scores, hypotheses, world, true_action, config.consistency)

    final_top_name, final_top_mass = top_hypothesis(scores, hypotheses)
    top_index = max(range(len(scores)), key=scores.__getitem__)
    schema_kind = hypotheses[top_index].kind

    return BaseConflictResult(
        overall_accuracy=statistics.mean(accuracies),
        early_accuracy=statistics.mean(accuracies[:20]),
        late_accuracy=statistics.mean(accuracies[-20:]),
        final_top_hypothesis=final_top_name,
        final_top_mass=final_top_mass,
        schema_kind=schema_kind,
        spawn_episode=spawn_episode,
    )


def evaluate_gate(
    config: BeliefTransferDiagnosticConfig,
    matches: int,
    total: int,
) -> tuple[bool, float]:
    alpha = config.applicability_prior_alpha + matches
    beta = config.applicability_prior_beta + (total - matches)
    posterior_mean = alpha / (alpha + beta)
    keep_transfer = posterior_mean >= config.applicability_threshold
    return keep_transfer, posterior_mean


def remove_hypothesis(
    hypotheses: list[Hypothesis],
    scores: list[float],
    name: str,
) -> None:
    for index, hypothesis in enumerate(hypotheses):
        if hypothesis.name == name:
            hypotheses.pop(index)
            scores.pop(index)
            return


def build_diagnostic_bank(feature_count: int) -> list[World]:
    bank: list[World] = []
    for left, right in ((1, 1), (1, 0), (0, 1), (0, 0)):
        values = [0] * feature_count
        values[0] = left
        values[5] = right
        bank.append(tuple(values))
    return bank


def select_diagnostic_probe(
    transfer_schema: str,
    probe_bank: Sequence[World],
    asked_indices: set[int],
) -> tuple[int, World]:
    transfer = Hypothesis(
        name=f"{transfer_schema}_0_5",
        kind=transfer_schema,
        feature_a=0,
        feature_b=5,
    )
    alternatives = [
        candidate
        for candidate in build_pair_composites(0, 5)
        if candidate.kind != transfer_schema
    ]

    best_index = -1
    best_score = -1
    for probe_index, probe in enumerate(probe_bank):
        if probe_index in asked_indices:
            continue
        transfer_prediction = transfer.predict(probe)
        disagreement = sum(
            1
            for candidate in alternatives
            if candidate.predict(probe) != transfer_prediction
        )
        if disagreement > best_score:
            best_index = probe_index
            best_score = disagreement

    if best_index < 0:
        raise ValueError("No diagnostic probe available")
    return best_index, probe_bank[best_index]


def run_target_conflict(
    config: BeliefTransferDiagnosticConfig,
    source_hypotheses: Sequence[Hypothesis],
    worlds: Sequence[World],
    operator: str,
    policy_name: str,
    transfer_schema: str | None,
) -> TargetResult:
    hypotheses = list(source_hypotheses)
    scores = [0.0] * len(hypotheses)
    accuracies: list[int] = []
    recent_accuracies: list[int] = []
    spawn_episode: int | None = None
    gate_status = "inactive"
    gate_phase = "-"
    gate_decision_index: int | None = None
    applicability_mean: float | None = None
    diagnostic_questions = 0
    action_error_before_gate = False
    transfer_name: str | None = None
    transfer_matches = 0

    if transfer_schema is not None:
        transferred = Hypothesis(
            name=f"{transfer_schema}_0_5",
            kind=transfer_schema,
            feature_a=0,
            feature_b=5,
        )
        hypotheses.append(transferred)
        scores.append(config.transfer_bias)
        transfer_name = transferred.name
        spawn_episode = 0

        if policy_name == "reactive_gating":
            gate_status = "probing"
            gate_phase = "action"

        if policy_name == "diagnostic_gating":
            gate_status = "probing"
            gate_phase = "diagnostic"
            probe_bank = build_diagnostic_bank(config.feature_count)
            asked_indices: set[int] = set()
            for question_index in range(config.diagnostic_questions):
                probe_index, probe = select_diagnostic_probe(transfer_schema, probe_bank, asked_indices)
                asked_indices.add(probe_index)
                diagnostic_questions += 1
                true_action = target_value(operator, probe, 0, 5)
                transfer_matches += 1 if transferred.predict(probe) == true_action else 0
                update_scores(scores, hypotheses, probe, true_action, config.consistency)
                keep_transfer, applicability_mean = evaluate_gate(
                    config,
                    matches=transfer_matches,
                    total=diagnostic_questions,
                )
                if not keep_transfer:
                    gate_status = "rejected"
                    gate_decision_index = question_index
                    remove_hypothesis(hypotheses, scores, transfer_name)
                    transfer_name = None
                    if spawn_episode == 0:
                        spawn_episode = None
                    break

            if gate_status == "probing":
                gate_status = "accepted"
                gate_decision_index = diagnostic_questions - 1

    for episode_index, world in enumerate(worlds):
        true_action = target_value(operator, world, 0, 5)
        belief = softmax(scores)
        probability_action_one = sum(
            weight * hypothesis.predict(world)
            for weight, hypothesis in zip(belief, hypotheses)
        )
        predicted_action = 1 if probability_action_one >= 0.5 else 0
        is_correct = 1 if predicted_action == true_action else 0

        transfer_active = transfer_name is not None and any(
            hypothesis.name == transfer_name for hypothesis in hypotheses
        )
        gate_pending = policy_name == "reactive_gating" and gate_status == "probing"
        if transfer_active and gate_pending and not is_correct:
            action_error_before_gate = True

        accuracies.append(is_correct)
        recent_accuracies.append(is_correct)
        if len(recent_accuracies) > config.spawn_window:
            recent_accuracies.pop(0)

        if policy_name == "reactive_gating" and transfer_active and gate_status == "probing":
            transferred = next(hypothesis for hypothesis in hypotheses if hypothesis.name == transfer_name)
            transfer_matches += 1 if transferred.predict(world) == true_action else 0
            keep_transfer, applicability_mean = evaluate_gate(
                config,
                matches=transfer_matches,
                total=episode_index + 1,
            )
            should_reject_now = not keep_transfer
            should_accept_now = keep_transfer and (episode_index + 1) >= config.reactive_probe_episodes
            if should_reject_now or should_accept_now:
                gate_decision_index = episode_index
                if should_accept_now:
                    gate_status = "accepted"
                else:
                    gate_status = "rejected"
                    remove_hypothesis(hypotheses, scores, transfer_name)
                    transfer_name = None
                    if spawn_episode == 0:
                        spawn_episode = None

        if (
            episode_index >= config.spawn_after
            and statistics.mean(recent_accuracies) < config.spawn_threshold
        ):
            existing_names = {hypothesis.name for hypothesis in hypotheses}
            to_add = [
                candidate
                for candidate in build_pair_composites(0, 5)
                if candidate.name not in existing_names
            ]
            if to_add:
                hypotheses.extend(to_add)
                scores.extend([0.0] * len(to_add))
                if policy_name == "scratch_synthesis":
                    spawn_episode = episode_index
                elif policy_name in {"naive_transfer", "reactive_gating", "diagnostic_gating"} and spawn_episode is None:
                    spawn_episode = episode_index

        update_scores(scores, hypotheses, world, true_action, config.consistency)

    final_top_name, _ = top_hypothesis(scores, hypotheses)

    if policy_name == "naive_transfer" and transfer_name is not None:
        gate_status = "forced_accept"
        gate_phase = "action"
    if policy_name == "reactive_gating" and gate_status == "probing":
        gate_status = "accepted"
        gate_decision_index = len(worlds) - 1

    return TargetResult(
        target_name=operator,
        policy_name=policy_name,
        overall_accuracy=statistics.mean(accuracies),
        first_ten_accuracy=statistics.mean(accuracies[:10]),
        last_ten_accuracy=statistics.mean(accuracies[-10:]),
        first_action_accuracy=accuracies[0],
        final_top_hypothesis=final_top_name,
        spawn_episode=spawn_episode,
        gate_status=gate_status,
        gate_phase=gate_phase,
        gate_decision_index=gate_decision_index,
        diagnostic_questions=diagnostic_questions,
        applicability_mean=applicability_mean,
        action_error_before_gate=action_error_before_gate,
    )


def print_report(
    config: BeliefTransferDiagnosticConfig,
    pretrain_results: Sequence[PretrainResult],
    base_conflict_result: BaseConflictResult,
    target_results: Sequence[TargetResult],
) -> None:
    print("Experiment: diagnostic questions for transfer applicability")
    print("Source beliefs:")
    print("A -> feat_0, B -> feat_4, C -> feat_5")
    print("Base conflict:")
    print("xor(feat_0, feat_4)")
    print("Positive target:")
    print("xor(feat_0, feat_5)")
    print("Negative target:")
    print("and(feat_0, feat_5)")
    print(f"Seed: {config.seed}")
    print()

    print("Pretrain results:")
    pretrain_header = f"{'Source':<10}{'Top belief':<16}{'Top mass':>10}"
    print(pretrain_header)
    print("-" * len(pretrain_header))
    for result in pretrain_results:
        print(
            f"{result.source_name:<10}"
            f"{result.top_hypothesis:<16}"
            f"{result.top_mass:>10.3f}"
        )

    print()
    print("Base conflict synthesis:")
    print(
        f"overall={base_conflict_result.overall_accuracy:.3f}, "
        f"early={base_conflict_result.early_accuracy:.3f}, "
        f"late={base_conflict_result.late_accuracy:.3f}, "
        f"top={base_conflict_result.final_top_hypothesis}, "
        f"schema={base_conflict_result.schema_kind}, "
        f"spawn={base_conflict_result.spawn_episode}"
    )

    print()
    print("Target conflict results:")
    header = (
        f"{'Target':<10}"
        f"{'Policy':<20}"
        f"{'Overall':>10}"
        f"{'First10':>10}"
        f"{'First1':>8}"
        f"{'Last10':>10}"
        f"{'Top belief':>18}"
        f"{'Gate':>12}"
        f"{'Phase':>12}"
        f"{'Idx':>6}"
        f"{'DiagQ':>8}"
        f"{'MetaP':>8}"
        f"{'Err<Gate':>10}"
    )
    print(header)
    print("-" * len(header))
    for result in target_results:
        gate_index = "-" if result.gate_decision_index is None else str(result.gate_decision_index)
        meta_text = "-" if result.applicability_mean is None else f"{result.applicability_mean:.3f}"
        err_text = "yes" if result.action_error_before_gate else "no"
        print(
            f"{result.target_name:<10}"
            f"{result.policy_name:<20}"
            f"{result.overall_accuracy:>10.3f}"
            f"{result.first_ten_accuracy:>10.3f}"
            f"{result.first_action_accuracy:>8.3f}"
            f"{result.last_ten_accuracy:>10.3f}"
            f"{result.final_top_hypothesis:>18}"
            f"{result.gate_status:>12}"
            f"{result.gate_phase:>12}"
            f"{gate_index:>6}"
            f"{result.diagnostic_questions:>8}"
            f"{meta_text:>8}"
            f"{err_text:>10}"
        )

    target_groups: dict[str, list[TargetResult]] = {}
    for result in target_results:
        target_groups.setdefault(result.target_name, []).append(result)

    print()
    print("Diagnostic transfer summary:")
    for target_name, results in sorted(target_groups.items()):
        naive = next(result for result in results if result.policy_name == "naive_transfer")
        reactive = next(result for result in results if result.policy_name == "reactive_gating")
        diagnostic = next(result for result in results if result.policy_name == "diagnostic_gating")
        print(
            f"{target_name}: "
            f"reactive_err_before_gate={reactive.action_error_before_gate}, "
            f"diagnostic_err_before_gate={diagnostic.action_error_before_gate}, "
            f"diagnostic_gate={diagnostic.gate_phase}@{diagnostic.gate_decision_index}, "
            f"diagnostic_vs_naive_first10={diagnostic.first_ten_accuracy - naive.first_ten_accuracy:+.3f}"
        )


def main() -> None:
    config = parse_args()
    rng = random.Random(config.seed)

    pretrain_worlds_a = sample_worlds(rng, config.feature_count, config.pretrain_episodes)
    pretrain_worlds_b = sample_worlds(rng, config.feature_count, config.pretrain_episodes)
    pretrain_worlds_c = sample_worlds(rng, config.feature_count, config.pretrain_episodes)
    base_conflict_worlds = sample_worlds(rng, config.feature_count, config.base_conflict_episodes)
    positive_target_worlds = sample_worlds(rng, config.feature_count, config.target_conflict_episodes)
    negative_target_worlds = sample_worlds(rng, config.feature_count, config.target_conflict_episodes)

    pretrain_a, belief_a = pretrain_source(config, pretrain_worlds_a, 0, "source_A")
    pretrain_b, belief_b = pretrain_source(config, pretrain_worlds_b, 4, "source_B")
    pretrain_c, belief_c = pretrain_source(config, pretrain_worlds_c, 5, "source_C")

    base_conflict_result = run_base_conflict(
        config,
        [belief_a, belief_b],
        base_conflict_worlds,
    )

    target_results = [
        run_target_conflict(
            config,
            [belief_a, belief_c],
            positive_target_worlds,
            operator="xor",
            policy_name="scratch_synthesis",
            transfer_schema=None,
        ),
        run_target_conflict(
            config,
            [belief_a, belief_c],
            positive_target_worlds,
            operator="xor",
            policy_name="naive_transfer",
            transfer_schema=base_conflict_result.schema_kind,
        ),
        run_target_conflict(
            config,
            [belief_a, belief_c],
            positive_target_worlds,
            operator="xor",
            policy_name="reactive_gating",
            transfer_schema=base_conflict_result.schema_kind,
        ),
        run_target_conflict(
            config,
            [belief_a, belief_c],
            positive_target_worlds,
            operator="xor",
            policy_name="diagnostic_gating",
            transfer_schema=base_conflict_result.schema_kind,
        ),
        run_target_conflict(
            config,
            [belief_a, belief_c],
            negative_target_worlds,
            operator="and",
            policy_name="scratch_synthesis",
            transfer_schema=None,
        ),
        run_target_conflict(
            config,
            [belief_a, belief_c],
            negative_target_worlds,
            operator="and",
            policy_name="naive_transfer",
            transfer_schema=base_conflict_result.schema_kind,
        ),
        run_target_conflict(
            config,
            [belief_a, belief_c],
            negative_target_worlds,
            operator="and",
            policy_name="reactive_gating",
            transfer_schema=base_conflict_result.schema_kind,
        ),
        run_target_conflict(
            config,
            [belief_a, belief_c],
            negative_target_worlds,
            operator="and",
            policy_name="diagnostic_gating",
            transfer_schema=base_conflict_result.schema_kind,
        ),
    ]

    print_report(config, [pretrain_a, pretrain_b, pretrain_c], base_conflict_result, target_results)


if __name__ == "__main__":
    main()
