from __future__ import annotations

import random
import statistics
from dataclasses import dataclass

import four_iteration_hidden_shift_test as iteration_suite
import learned_hidden_shift_batch_suite as flat_suite
import long_horizon_hidden_shift_test as long_suite


@dataclass(frozen=True)
class ReturnSignalMetrics:
    baseline: float
    similar: float
    dissimilar: float
    drifted: float
    high_risk: float
    mixed_mean: float
    mixed_floor: float
    long_mean: float


class ReturnSignalBeliefState(iteration_suite.RecurrentBeliefState):
    def __init__(self, config, context_probs, probe_cost, low_risk_cost) -> None:
        super().__init__(config, context_probs, probe_cost, low_risk_cost)
        self.last_seen = {mode_name: None for mode_name in flat_suite.MODE_NAMES}
        self.top_streak = 0
        self.current_top = None

    def update(self, episode, *, probed: bool) -> None:
        posterior = self.posterior(episode.context)
        top_mode = max(posterior, key=posterior.get)
        if top_mode == self.current_top:
            self.top_streak += 1
        else:
            self.current_top = top_mode
            self.top_streak = 1
        super().update(episode, probed=probed)
        for mode_name in self.last_seen:
            if self.last_seen[mode_name] is not None:
                self.last_seen[mode_name] += 1
        if posterior[top_mode] >= self.config.conf_hi and self.top_streak >= 2:
            self.last_seen[top_mode] = 0


def return_override_action(values, belief: ReturnSignalBeliefState, episode) -> str:
    state = belief.state_key(episode)
    greedy = iteration_suite.greedy_action(values, state)
    posterior = belief.posterior(episode.context)
    top_mode = max(posterior, key=posterior.get)
    revisit = belief.last_seen[top_mode] is not None and belief.last_seen[top_mode] >= 4
    high_confidence = posterior[top_mode] >= 0.8
    high_risk = episode.risk_cost > belief.low_risk_cost
    if greedy == "act" and revisit and high_confidence and high_risk:
        return "probe"
    return greedy


def evaluate_iter5(config: iteration_suite.IterationConfig) -> ReturnSignalMetrics:
    scenario_means = {scenario_name: [] for scenario_name in iteration_suite.SCENARIO_ORDER}
    long_means: list[float] = []

    for seed in range(config.seed_start, config.seed_start + config.seed_count):
        values = iteration_suite.train_recurrent_variant(config, seed, "mixed_plus_long")

        for offset, scenario_name in enumerate(iteration_suite.SCENARIO_ORDER):
            scenario = iteration_suite.scenario_by_name(scenario_name, config, seed)
            belief = ReturnSignalBeliefState(
                config,
                scenario.context_probs,
                scenario.probe_cost,
                scenario.low_risk_cost,
            )
            episodes = flat_suite.build_episodes(
                random.Random(seed + 100 + offset),
                scenario,
                config.block_length,
            )
            rewards: list[float] = []
            for episode in episodes:
                action = return_override_action(values, belief, episode)
                reward, _correct = iteration_suite.reward_for_action(action, belief, episode)
                rewards.append(reward)
                belief.update(episode, probed=(action == "probe"))
            scenario_means[scenario_name].append(statistics.mean(rewards))

        reference = long_suite.HiddenShiftConfig(seed=seed)
        belief = ReturnSignalBeliefState(
            config,
            long_suite.MODE_CONTEXT_PROBS,
            reference.probe_cost,
            reference.low_risk_cost,
        )
        episodes = long_suite.build_episodes(random.Random(seed), reference)
        rewards = []
        for episode in episodes:
            action = return_override_action(values, belief, episode)
            reward, _correct = iteration_suite.reward_for_action(action, belief, episode)
            rewards.append(reward)
            belief.update(episode, probed=(action == "probe"))
        long_means.append(statistics.mean(rewards))

    means = {
        scenario_name: statistics.mean(scores)
        for scenario_name, scores in scenario_means.items()
    }
    return ReturnSignalMetrics(
        baseline=means["baseline"],
        similar=means["similar"],
        dissimilar=means["dissimilar"],
        drifted=means["drifted"],
        high_risk=means["high_risk"],
        mixed_mean=statistics.mean(means.values()),
        mixed_floor=min(means.values()),
        long_mean=statistics.mean(long_means),
    )


def main() -> None:
    config = iteration_suite.IterationConfig()
    iter4 = iteration_suite.evaluate_variant(
        config,
        iteration_suite.Variant(
            variant_id="iter4_recurrent_long_finetune",
            hypothesis="reference",
            trainer="mixed_plus_long",
        ),
    )
    iter5 = evaluate_iter5(config)
    reference = iteration_suite.evaluate_risk_aware_reference(config)

    rows = [
        {
            "variant": "iter4_recurrent_long_finetune",
            "mixed_mean": round(iter4.mixed_mean, 3),
            "mixed_floor": round(iter4.mixed_floor, 3),
            "long_mean": round(iter4.long_mean, 3),
        },
        {
            "variant": "iter5_return_mode_signal",
            "mixed_mean": round(iter5.mixed_mean, 3),
            "mixed_floor": round(iter5.mixed_floor, 3),
            "long_mean": round(iter5.long_mean, 3),
        },
    ]
    iteration_suite.print_table("Iter4 vs Iter5", rows)
    print(f"hand-crafted risk_aware_hidden_shift long_mean={reference:.3f}")


if __name__ == "__main__":
    main()
