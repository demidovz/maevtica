from __future__ import annotations

import argparse
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass


CAUSE_A = 0
CAUSE_B = 1
RED = "red"
BLUE = "blue"


@dataclass(frozen=True)
class CausalLatentInterventionConfig:
    seed: int = 7
    train_episodes: int = 4000
    eval_episodes: int = 2000
    probe_cost: float = 0.18
    context_correlation: float = 0.85


@dataclass(frozen=True)
class EpisodeSpec:
    phase: int
    cause: int
    context: str
    appearance: str = "masked"


@dataclass(frozen=True)
class EvalMetrics:
    policy_name: str
    phase_name: str
    accuracy: float
    mean_utility: float
    probe_rate: float


def parse_args() -> CausalLatentInterventionConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Compare observational latent compression against causal latent inference "
            "that uses interventions to recover hidden causes."
        )
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-episodes", type=int, default=4000)
    parser.add_argument("--eval-episodes", type=int, default=2000)
    parser.add_argument("--probe-cost", type=float, default=0.18)
    parser.add_argument("--context-correlation", type=float, default=0.85)
    args = parser.parse_args()
    return CausalLatentInterventionConfig(
        seed=args.seed,
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        probe_cost=args.probe_cost,
        context_correlation=args.context_correlation,
    )


def sample_episode(
    rng: random.Random,
    phase: int,
    config: CausalLatentInterventionConfig,
) -> EpisodeSpec:
    cause = rng.choice((CAUSE_A, CAUSE_B))
    if phase == 1:
        preferred_prob = config.context_correlation
    else:
        preferred_prob = 1.0 - config.context_correlation

    if cause == CAUSE_A:
        context = RED if rng.random() < preferred_prob else BLUE
    else:
        context = BLUE if rng.random() < preferred_prob else RED
    return EpisodeSpec(phase=phase, cause=cause, context=context)


def intervention_response(spec: EpisodeSpec) -> int:
    return 0 if spec.cause == CAUSE_A else 1


def train_surface_proxy(
    rng: random.Random,
    config: CausalLatentInterventionConfig,
) -> dict[str, int]:
    counts: dict[str, Counter[int]] = defaultdict(Counter)
    for _ in range(config.train_episodes):
        spec = sample_episode(rng, phase=1, config=config)
        counts[spec.context][spec.cause] += 1
    return {
        context: counter.most_common(1)[0][0]
        for context, counter in counts.items()
    }


def train_causal_latent(
    rng: random.Random,
    config: CausalLatentInterventionConfig,
) -> dict[int, int]:
    counts: dict[int, Counter[int]] = defaultdict(Counter)
    for _ in range(config.train_episodes):
        spec = sample_episode(rng, phase=1, config=config)
        response = intervention_response(spec)
        counts[response][spec.cause] += 1
    return {
        response: counter.most_common(1)[0][0]
        for response, counter in counts.items()
    }


def evaluate_surface_proxy(
    mapping: dict[str, int],
    phase: int,
    rng: random.Random,
    config: CausalLatentInterventionConfig,
) -> EvalMetrics:
    correct = 0
    utilities: list[float] = []
    for _ in range(config.eval_episodes):
        spec = sample_episode(rng, phase=phase, config=config)
        predicted = mapping[spec.context]
        is_correct = predicted == spec.cause
        correct += int(is_correct)
        utilities.append(1.0 if is_correct else 0.0)
    return EvalMetrics(
        policy_name="surface_proxy_memory",
        phase_name=f"phase_{phase}",
        accuracy=correct / config.eval_episodes,
        mean_utility=statistics.mean(utilities),
        probe_rate=0.0,
    )


def evaluate_causal_latent(
    mapping: dict[int, int],
    phase: int,
    rng: random.Random,
    config: CausalLatentInterventionConfig,
) -> EvalMetrics:
    correct = 0
    utilities: list[float] = []
    for _ in range(config.eval_episodes):
        spec = sample_episode(rng, phase=phase, config=config)
        response = intervention_response(spec)
        predicted = mapping[response]
        is_correct = predicted == spec.cause
        correct += int(is_correct)
        utilities.append((1.0 if is_correct else 0.0) - config.probe_cost)
    return EvalMetrics(
        policy_name="causal_intervention_latent",
        phase_name=f"phase_{phase}",
        accuracy=correct / config.eval_episodes,
        mean_utility=statistics.mean(utilities),
        probe_rate=1.0,
    )


def causal_alignment_surface(
    rng: random.Random,
    config: CausalLatentInterventionConfig,
) -> float:
    counts: dict[int, Counter[str]] = defaultdict(Counter)
    total_by_cause: Counter[int] = Counter()
    for phase in (1, 2):
        for _ in range(config.eval_episodes):
            spec = sample_episode(rng, phase=phase, config=config)
            latent_id = spec.context
            counts[spec.cause][latent_id] += 1
            total_by_cause[spec.cause] += 1
    coverage = []
    for cause in (CAUSE_A, CAUSE_B):
        dominant = counts[cause].most_common(1)[0][1]
        coverage.append(dominant / total_by_cause[cause])
    return statistics.mean(coverage)


def causal_alignment_intervention(
    rng: random.Random,
    config: CausalLatentInterventionConfig,
) -> float:
    counts: dict[int, Counter[str]] = defaultdict(Counter)
    total_by_cause: Counter[int] = Counter()
    for phase in (1, 2):
        for _ in range(config.eval_episodes):
            spec = sample_episode(rng, phase=phase, config=config)
            latent_id = f"resp_{intervention_response(spec)}"
            counts[spec.cause][latent_id] += 1
            total_by_cause[spec.cause] += 1
    coverage = []
    for cause in (CAUSE_A, CAUSE_B):
        dominant = counts[cause].most_common(1)[0][1]
        coverage.append(dominant / total_by_cause[cause])
    return statistics.mean(coverage)


def mixed_mean_utility(
    phase1: EvalMetrics,
    phase2: EvalMetrics,
) -> float:
    return statistics.mean([phase1.mean_utility, phase2.mean_utility])


def print_report(
    config: CausalLatentInterventionConfig,
    surface_phase1: EvalMetrics,
    surface_phase2: EvalMetrics,
    causal_phase1: EvalMetrics,
    causal_phase2: EvalMetrics,
    surface_align: float,
    causal_align: float,
) -> None:
    print("Experiment: causal latent states vs observational surface proxies")
    print("Visible appearance is masked; only nuisance context is observable before intervention")
    print("Phase 1: context is correlated with cause")
    print("Phase 2: the context correlation flips, but the causal probe stays stable")
    print(
        f"Probe cost: {config.probe_cost:.2f} | "
        f"Train episodes: {config.train_episodes} | Eval episodes: {config.eval_episodes}"
    )
    print(f"Seed: {config.seed}")
    print()

    header = (
        f"{'Policy':<28}"
        f"{'Phase':<10}"
        f"{'Acc':>10}"
        f"{'MeanU':>10}"
        f"{'Probe':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in (
        surface_phase1,
        surface_phase2,
        causal_phase1,
        causal_phase2,
    ):
        print(
            f"{row.policy_name:<28}"
            f"{row.phase_name:<10}"
            f"{row.accuracy:>10.3f}"
            f"{row.mean_utility:>10.3f}"
            f"{row.probe_rate:>10.3f}"
        )

    print()
    print(
        "Pooled causal alignment across both phases:"
        f" surface_proxy_memory={surface_align:.3f},"
        f" causal_intervention_latent={causal_align:.3f}"
    )
    print(
        "Mixed mean utility across both phases:"
        f" surface_proxy_memory={mixed_mean_utility(surface_phase1, surface_phase2):.3f},"
        f" causal_intervention_latent={mixed_mean_utility(causal_phase1, causal_phase2):.3f}"
    )


def main() -> None:
    config = parse_args()
    train_rng = random.Random(config.seed)

    surface_mapping = train_surface_proxy(train_rng, config)
    causal_mapping = train_causal_latent(train_rng, config)

    surface_phase1 = evaluate_surface_proxy(
        mapping=surface_mapping,
        phase=1,
        rng=random.Random(config.seed + 101),
        config=config,
    )
    surface_phase2 = evaluate_surface_proxy(
        mapping=surface_mapping,
        phase=2,
        rng=random.Random(config.seed + 202),
        config=config,
    )
    causal_phase1 = evaluate_causal_latent(
        mapping=causal_mapping,
        phase=1,
        rng=random.Random(config.seed + 101),
        config=config,
    )
    causal_phase2 = evaluate_causal_latent(
        mapping=causal_mapping,
        phase=2,
        rng=random.Random(config.seed + 202),
        config=config,
    )

    surface_align = causal_alignment_surface(
        rng=random.Random(config.seed + 303),
        config=config,
    )
    causal_align = causal_alignment_intervention(
        rng=random.Random(config.seed + 404),
        config=config,
    )

    print_report(
        config=config,
        surface_phase1=surface_phase1,
        surface_phase2=surface_phase2,
        causal_phase1=causal_phase1,
        causal_phase2=causal_phase2,
        surface_align=surface_align,
        causal_align=causal_align,
    )


if __name__ == "__main__":
    main()
