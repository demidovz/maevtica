from __future__ import annotations

import argparse
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass


MODE_TABLES = {
    "color": {"00": 0, "01": 0, "10": 1, "11": 1},
    "shape": {"00": 0, "01": 1, "10": 0, "11": 1},
    "xor": {"00": 0, "01": 1, "10": 1, "11": 0},
}
CONTEXT_PREFIX = ("01", "10", "11", "00")
CONTEXTS = tuple(MODE_TABLES["color"].keys())


@dataclass(frozen=True)
class LongHorizonIntegratedConfig:
    seed: int = 7
    block_length: int = 16
    probe_cost: float = 0.18
    high_risk_cost: float = 4.0
    low_risk_cost: float = 1.0
    random_high_risk_prob: float = 0.10
    shift_hint_episodes: int = 2


@dataclass(frozen=True)
class Episode:
    episode_index: int
    block_index: int
    block_offset: int
    mode_name: str
    context: str
    true_label: int
    risk_cost: float
    shift_hint: bool
    return_block: bool


@dataclass(frozen=True)
class PolicyMetrics:
    policy_name: str
    mean_utility: float
    accuracy: float
    catastrophic_error_rate: float
    probe_rate: float
    return_first4_accuracy: float
    avg_reactivation_lag: str
    archive_hit_rate: str
    final_archives: int


def parse_args() -> LongHorizonIntegratedConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether an integrated agent can use causal probes, archived latent "
            "modes, and reactivation to survive a long horizon with recurring regimes."
        )
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--block-length", type=int, default=16)
    parser.add_argument("--probe-cost", type=float, default=0.18)
    parser.add_argument("--high-risk-cost", type=float, default=4.0)
    parser.add_argument("--low-risk-cost", type=float, default=1.0)
    parser.add_argument("--random-high-risk-prob", type=float, default=0.10)
    parser.add_argument("--shift-hint-episodes", type=int, default=2)
    args = parser.parse_args()
    return LongHorizonIntegratedConfig(
        seed=args.seed,
        block_length=args.block_length,
        probe_cost=args.probe_cost,
        high_risk_cost=args.high_risk_cost,
        low_risk_cost=args.low_risk_cost,
        random_high_risk_prob=args.random_high_risk_prob,
        shift_hint_episodes=args.shift_hint_episodes,
    )


def build_episodes(
    rng: random.Random,
    config: LongHorizonIntegratedConfig,
) -> list[Episode]:
    schedule = ("color", "shape", "color", "xor", "shape", "xor")
    seen_modes: set[str] = set()
    episodes: list[Episode] = []

    for block_index, mode_name in enumerate(schedule):
        return_block = mode_name in seen_modes
        for block_offset in range(config.block_length):
            if block_offset < len(CONTEXT_PREFIX):
                context = CONTEXT_PREFIX[block_offset]
            else:
                context = rng.choice(CONTEXTS)
            shift_hint = block_offset < config.shift_hint_episodes
            risk_cost = (
                config.high_risk_cost
                if block_offset == 0 or rng.random() < config.random_high_risk_prob
                else config.low_risk_cost
            )
            episodes.append(
                Episode(
                    episode_index=len(episodes),
                    block_index=block_index,
                    block_offset=block_offset,
                    mode_name=mode_name,
                    context=context,
                    true_label=MODE_TABLES[mode_name][context],
                    risk_cost=risk_cost,
                    shift_hint=shift_hint,
                    return_block=return_block,
                )
            )
        seen_modes.add(mode_name)
    return episodes


def guess_from_table(table: dict[str, int]) -> int:
    if not table:
        return 0
    ones = sum(table.values())
    zeros = len(table) - ones
    return 1 if ones > zeros else 0


def format_optional(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def compute_metrics(
    policy_name: str,
    utilities: list[float],
    correct_flags: list[int],
    catastrophic_flags: list[int],
    high_risk_total: int,
    probe_flags: list[int],
    return_first4_correct: list[int],
    reactivation_lags: list[int],
    archive_hits: int,
    return_blocks: int,
    final_archives: int,
) -> PolicyMetrics:
    catastrophic_error_rate = 0.0 if high_risk_total == 0 else sum(catastrophic_flags) / high_risk_total
    archive_hit_rate = None if return_blocks == 0 else archive_hits / return_blocks
    avg_reactivation_lag = None if not reactivation_lags else statistics.mean(reactivation_lags)
    return PolicyMetrics(
        policy_name=policy_name,
        mean_utility=statistics.mean(utilities),
        accuracy=statistics.mean(correct_flags),
        catastrophic_error_rate=catastrophic_error_rate,
        probe_rate=statistics.mean(probe_flags),
        return_first4_accuracy=statistics.mean(return_first4_correct),
        avg_reactivation_lag=format_optional(avg_reactivation_lag),
        archive_hit_rate=format_optional(archive_hit_rate),
        final_archives=final_archives,
    )


def run_flat_surface_memory(episodes: list[Episode]) -> PolicyMetrics:
    counts: dict[str, Counter[int]] = defaultdict(Counter)
    utilities: list[float] = []
    correct_flags: list[int] = []
    catastrophic_flags: list[int] = []
    probe_flags: list[int] = []
    return_first4_correct: list[int] = []
    high_risk_total = 0
    return_blocks = len({episode.block_index for episode in episodes if episode.return_block})

    for episode in episodes:
        prediction = counts[episode.context].most_common(1)[0][0] if counts[episode.context] else 0
        correct = int(prediction == episode.true_label)
        utility = 1.0 if correct else -episode.risk_cost
        counts[episode.context][episode.true_label] += 1

        utilities.append(utility)
        correct_flags.append(correct)
        probe_flags.append(0)
        if episode.risk_cost == 4.0:
            high_risk_total += 1
            catastrophic_flags.append(1 - correct)
        if episode.return_block and episode.block_offset < 4:
            return_first4_correct.append(correct)

    return compute_metrics(
        policy_name="flat_surface_memory",
        utilities=utilities,
        correct_flags=correct_flags,
        catastrophic_flags=catastrophic_flags,
        high_risk_total=high_risk_total,
        probe_flags=probe_flags,
        return_first4_correct=return_first4_correct,
        reactivation_lags=[],
        archive_hits=0,
        return_blocks=return_blocks,
        final_archives=0,
    )


def run_always_probe(episodes: list[Episode], config: LongHorizonIntegratedConfig) -> PolicyMetrics:
    utilities: list[float] = []
    correct_flags: list[int] = []
    catastrophic_flags: list[int] = []
    probe_flags: list[int] = []
    return_first4_correct: list[int] = []
    high_risk_total = 0
    return_blocks = len({episode.block_index for episode in episodes if episode.return_block})

    for episode in episodes:
        utilities.append(1.0 - config.probe_cost)
        correct_flags.append(1)
        probe_flags.append(1)
        if episode.risk_cost == 4.0:
            high_risk_total += 1
            catastrophic_flags.append(0)
        if episode.return_block and episode.block_offset < 4:
            return_first4_correct.append(1)

    return compute_metrics(
        policy_name="always_probe_causal",
        utilities=utilities,
        correct_flags=correct_flags,
        catastrophic_flags=catastrophic_flags,
        high_risk_total=high_risk_total,
        probe_flags=probe_flags,
        return_first4_correct=return_first4_correct,
        reactivation_lags=[],
        archive_hits=0,
        return_blocks=return_blocks,
        final_archives=0,
    )


def run_adaptive_no_archive(
    episodes: list[Episode],
    config: LongHorizonIntegratedConfig,
) -> PolicyMetrics:
    active_table: dict[str, int] = {}
    utilities: list[float] = []
    correct_flags: list[int] = []
    catastrophic_flags: list[int] = []
    probe_flags: list[int] = []
    return_first4_correct: list[int] = []
    high_risk_total = 0
    return_blocks = len({episode.block_index for episode in episodes if episode.return_block})

    for episode in episodes:
        if episode.block_offset == 0 and episode.shift_hint:
            active_table = {}

        if episode.shift_hint:
            probed = True
            prediction = episode.true_label
        elif episode.context in active_table:
            probed = False
            prediction = active_table[episode.context]
        elif episode.risk_cost > config.low_risk_cost:
            probed = True
            prediction = episode.true_label
        else:
            probed = False
            prediction = guess_from_table(active_table)

        correct = int(prediction == episode.true_label)
        if probed:
            utility = 1.0 - config.probe_cost
        else:
            utility = 1.0 if correct else -episode.risk_cost

        active_table[episode.context] = episode.true_label
        if not probed and not correct and episode.context in active_table:
            active_table[episode.context] = episode.true_label

        utilities.append(utility)
        correct_flags.append(correct)
        probe_flags.append(int(probed))
        if episode.risk_cost == 4.0:
            high_risk_total += 1
            catastrophic_flags.append(1 - correct)
        if episode.return_block and episode.block_offset < 4:
            return_first4_correct.append(correct)

    return compute_metrics(
        policy_name="adaptive_no_archive",
        utilities=utilities,
        correct_flags=correct_flags,
        catastrophic_flags=catastrophic_flags,
        high_risk_total=high_risk_total,
        probe_flags=probe_flags,
        return_first4_correct=return_first4_correct,
        reactivation_lags=[],
        archive_hits=0,
        return_blocks=return_blocks,
        final_archives=0,
    )


def consistent_modes(observations: dict[str, int], candidates: dict[str, dict[str, int]]) -> list[str]:
    matches: list[str] = []
    for mode_name, table in candidates.items():
        if all(table[context] == label for context, label in observations.items()):
            matches.append(mode_name)
    return matches


def run_integrated_archive_agent(
    episodes: list[Episode],
    config: LongHorizonIntegratedConfig,
) -> PolicyMetrics:
    archives: dict[str, dict[str, int]] = {}
    active_mode: str | None = None
    recent_obs: dict[str, int] = {}
    utilities: list[float] = []
    correct_flags: list[int] = []
    catastrophic_flags: list[int] = []
    probe_flags: list[int] = []
    return_first4_correct: list[int] = []
    high_risk_total = 0
    reactivation_lags: list[int] = []
    archive_hits = 0
    return_blocks = len({episode.block_index for episode in episodes if episode.return_block})
    block_reactivated = False

    for episode in episodes:
        if episode.block_offset == 0:
            active_mode = None
            recent_obs = {}
            block_reactivated = False

        if active_mode is None and recent_obs:
            archive_matches = consistent_modes(recent_obs, archives)
            if len(archive_matches) == 1:
                active_mode = archive_matches[0]
            else:
                all_matches = consistent_modes(recent_obs, MODE_TABLES)
                if len(all_matches) == 1:
                    active_mode = all_matches[0]
                    archives.setdefault(active_mode, MODE_TABLES[active_mode])

        if active_mode is not None:
            probed = False
            prediction = MODE_TABLES[active_mode][episode.context]
        else:
            probed = True
            prediction = episode.true_label

        correct = int(prediction == episode.true_label)
        utility = (1.0 - config.probe_cost) if probed else (1.0 if correct else -episode.risk_cost)

        if probed:
            recent_obs[episode.context] = episode.true_label
            archive_matches = consistent_modes(recent_obs, archives)
            if len(archive_matches) == 1:
                active_mode = archive_matches[0]
            else:
                all_matches = consistent_modes(recent_obs, MODE_TABLES)
                if len(all_matches) == 1:
                    active_mode = all_matches[0]
                    archives.setdefault(active_mode, MODE_TABLES[active_mode])
        elif not correct:
            active_mode = None
            recent_obs = {episode.context: episode.true_label}

        if (
            episode.return_block
            and not block_reactivated
            and active_mode == episode.mode_name
            and episode.mode_name in archives
        ):
            reactivation_lags.append(episode.block_offset)
            archive_hits += int(episode.block_offset <= 1)
            block_reactivated = True

        utilities.append(utility)
        correct_flags.append(correct)
        probe_flags.append(int(probed))
        if episode.risk_cost == 4.0:
            high_risk_total += 1
            catastrophic_flags.append(1 - correct)
        if episode.return_block and episode.block_offset < 4:
            return_first4_correct.append(correct)

    return compute_metrics(
        policy_name="integrated_archive_agent",
        utilities=utilities,
        correct_flags=correct_flags,
        catastrophic_flags=catastrophic_flags,
        high_risk_total=high_risk_total,
        probe_flags=probe_flags,
        return_first4_correct=return_first4_correct,
        reactivation_lags=reactivation_lags,
        archive_hits=archive_hits,
        return_blocks=return_blocks,
        final_archives=len(archives),
    )


def print_report(
    config: LongHorizonIntegratedConfig,
    episodes: list[Episode],
    metrics: list[PolicyMetrics],
) -> None:
    total_episodes = len(episodes)
    return_blocks = sorted({episode.block_index for episode in episodes if episode.return_block})
    high_risk_total = sum(1 for episode in episodes if episode.risk_cost == config.high_risk_cost)

    print("Experiment: long-horizon integrated causal agent")
    print("World modes recur over time: color -> shape -> color -> xor -> shape -> xor")
    print("The agent sees context, risk, and a short drift hint at block start; probing reveals the true label at a cost")
    print(
        f"Episodes: {total_episodes} | Block length: {config.block_length} | "
        f"Return blocks: {len(return_blocks)} | High-risk episodes: {high_risk_total}"
    )
    print(
        f"Probe cost: {config.probe_cost:.2f} | "
        f"Low-risk error: {config.low_risk_cost:.1f} | High-risk error: {config.high_risk_cost:.1f}"
    )
    print(f"Seed: {config.seed}")
    print()

    header = (
        f"{'Policy':<24}"
        f"{'MeanU':>10}"
        f"{'Acc':>10}"
        f"{'CatErr':>10}"
        f"{'Probe':>10}"
        f"{'Return4':>10}"
        f"{'ReactLag':>10}"
        f"{'ArchHit':>10}"
        f"{'Arch':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in metrics:
        print(
            f"{row.policy_name:<24}"
            f"{row.mean_utility:>10.3f}"
            f"{row.accuracy:>10.3f}"
            f"{row.catastrophic_error_rate:>10.3f}"
            f"{row.probe_rate:>10.3f}"
            f"{row.return_first4_accuracy:>10.3f}"
            f"{row.avg_reactivation_lag:>10}"
            f"{row.archive_hit_rate:>10}"
            f"{row.final_archives:>8}"
        )


def main() -> None:
    config = parse_args()
    episodes = build_episodes(random.Random(config.seed), config)
    metrics = [
        run_flat_surface_memory(episodes),
        run_always_probe(episodes, config),
        run_adaptive_no_archive(episodes, config),
        run_integrated_archive_agent(episodes, config),
    ]
    print_report(config, episodes, metrics)


if __name__ == "__main__":
    main()
