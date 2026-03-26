from __future__ import annotations

import statistics

from epistemic_engine.models import BenchmarkResult, EpisodeMetrics


def summarize(policy_name: str, episodes: list[EpisodeMetrics]) -> BenchmarkResult:
    return BenchmarkResult(
        policy_name=policy_name,
        accuracy=statistics.mean(episode.correct for episode in episodes),
        mean_steps=statistics.mean(episode.steps for episode in episodes),
        mean_cost=statistics.mean(episode.total_cost for episode in episodes),
        mean_final_confidence=statistics.mean(
            episode.final_confidence for episode in episodes
        ),
        mean_utility=statistics.mean(episode.utility for episode in episodes),
        budget_stop_rate=statistics.mean(episode.budget_stop for episode in episodes),
        step_stop_rate=statistics.mean(episode.step_stop for episode in episodes),
    )
