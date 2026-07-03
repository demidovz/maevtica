from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path

from epistemic_explorer.evolution_models import (
    EcosystemSummary,
    EpochMetrics,
    ResearchStrategy,
    StrategyActionResult,
    StrategyGenome,
)


DOMAINS = ["theory", "experiment", "counterexample", "integration", "meta"]


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class EvolutionaryEcosystem:
    def __init__(
        self,
        population_size: int,
        epochs: int,
        seed: int,
        out_dir: Path,
        mutation_rate: float = 0.18,
    ) -> None:
        self.population_size = population_size
        self.epochs = epochs
        self.seed = seed
        self.rng = random.Random(seed)
        self.out_dir = out_dir
        self.mutation_rate = mutation_rate
        self.strategies: list[ResearchStrategy] = []
        self.epoch_metrics: list[EpochMetrics] = []
        self.events: list[dict[str, object]] = []
        self.total_epistemic_value = 0.0
        self.births = 0
        self.deaths = 0

    def initialize(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.strategies = [self._new_strategy(index) for index in range(1, self.population_size + 1)]
        self.births = len(self.strategies)
        self._write_json("initial_population.json", [strategy.to_dict() for strategy in self.strategies])

    def run(self) -> EcosystemSummary:
        self.initialize()
        for epoch in range(1, self.epochs + 1):
            self.run_epoch(epoch)
        summary = self.summary()
        self._write_json("epoch_metrics.json", [metrics.to_dict() for metrics in self.epoch_metrics])
        self._write_json("final_population.json", [strategy.to_dict() for strategy in self.strategies])
        self._write_json("events.json", self.events)
        self._write_json("summary.json", summary.to_dict())
        return summary

    def run_epoch(self, epoch: int) -> None:
        alive_before = [strategy for strategy in self.strategies if strategy.alive]
        results = [self._act(strategy, epoch) for strategy in alive_before]
        for strategy, result in zip(alive_before, results):
            self._apply_result(strategy, result)
        deaths = self._kill_weak_strategies(epoch)
        births = self._reproduce(epoch)
        metrics = self._epoch_metrics(epoch, results, births, deaths)
        self.epoch_metrics.append(metrics)
        self.events.append(
            {
                "epoch": epoch,
                "type": "epoch_completed",
                "metrics": metrics.to_dict(),
            }
        )

    def _act(self, strategy: ResearchStrategy, epoch: int) -> StrategyActionResult:
        genome = strategy.genome
        domain_weights = {
            "theory": genome.formalization_bias + 0.5 * genome.reasoning_depth,
            "experiment": genome.simulation_bias + genome.quick_check_bias,
            "counterexample": genome.criticality + genome.counterexample_bias,
            "integration": genome.analogy_bias + genome.merge_bias,
            "meta": genome.meta_bias + 0.5 * genome.collaboration_bias,
        }
        domain = max(domain_weights, key=domain_weights.get)
        cost = (
            strategy.existence_cost
            + 0.8 * genome.reasoning_depth
            + 0.7 * genome.simulation_bias
            + 0.5 * genome.formalization_bias
            + 0.4 * genome.counterexample_bias
        )
        hypotheses_created = 1 if self.rng.random() < 0.35 + 0.25 * genome.analogy_bias else 0
        falsification_power = 0.2 + 0.45 * genome.criticality + 0.35 * genome.counterexample_bias
        hypotheses_falsified = 1 if self.rng.random() < falsification_power else 0
        strong_test_power = 0.15 + 0.35 * genome.simulation_bias + 0.25 * genome.formalization_bias
        hypotheses_survived = 1 if self.rng.random() < strong_test_power * (1.0 - 0.35 * genome.risk_tolerance) else 0
        experiment_power = 0.1 + 0.55 * genome.simulation_bias + 0.2 * genome.quick_check_bias
        discriminating_experiments = 1 if self.rng.random() < experiment_power else 0
        valuable_questions = 1 if self.rng.random() < 0.15 + 0.35 * genome.meta_bias + 0.2 * genome.reasoning_depth else 0
        bridge_power = 0.05 + 0.45 * genome.analogy_bias + 0.25 * genome.merge_bias + 0.15 * genome.collaboration_bias
        bridges_created = 1 if self.rng.random() < bridge_power else 0
        compression_delta = (
            0.4 * genome.merge_bias
            + 0.2 * genome.formalization_bias
            - 0.25 * genome.split_bias
            - 0.15 * hypotheses_created
        )
        dead_end_risk = (
            0.3 * genome.risk_tolerance
            + 0.2 * genome.reasoning_depth
            + 0.2 * genome.analogy_bias
            - 0.25 * genome.criticality
            - 0.2 * genome.quick_check_bias
        )
        dead_end_cost = cost if self.rng.random() < clamp(dead_end_risk) else 0.0

        value = (
            2.0 * hypotheses_falsified
            + 2.5 * hypotheses_survived
            + 3.0 * discriminating_experiments
            + 1.5 * valuable_questions
            + 2.0 * bridges_created
            + 2.0 * max(0.0, compression_delta)
            - dead_end_cost
        )
        energy_delta = value - cost
        success_events = []
        failure_events = []
        if hypotheses_falsified:
            success_events.append("counterexample_found")
        if hypotheses_survived:
            success_events.append("hypothesis_survived_strong_test")
        if discriminating_experiments:
            success_events.append("discriminating_experiment")
        if bridges_created:
            success_events.append("bridge_created")
        if compression_delta > 0.2:
            success_events.append("graph_compressed")
        if dead_end_cost > 0:
            failure_events.append("dead_end")
        if value <= 0:
            failure_events.append("no_epistemic_value")

        self.total_epistemic_value += value
        return StrategyActionResult(
            strategy_id=strategy.strategy_id,
            epoch=epoch,
            domain=domain,
            cost=cost,
            energy_delta=energy_delta,
            hypotheses_created=hypotheses_created,
            hypotheses_falsified=hypotheses_falsified,
            hypotheses_survived_strong_test=hypotheses_survived,
            discriminating_experiments=discriminating_experiments,
            valuable_questions=valuable_questions,
            bridges_created=bridges_created,
            compression_delta=compression_delta,
            dead_end_cost=dead_end_cost,
            success_events=success_events,
            failure_events=failure_events,
        )

    def _apply_result(self, strategy: ResearchStrategy, result: StrategyActionResult) -> None:
        strategy.age += 1
        strategy.energy += result.energy_delta
        strategy.successes.extend(result.success_events)
        strategy.failures.extend(result.failure_events)
        strategy.species_hint = classify_species(strategy.genome)
        self.events.append({"epoch": result.epoch, "type": "strategy_result", "result": result.to_dict()})

    def _kill_weak_strategies(self, epoch: int) -> int:
        deaths = 0
        for strategy in self.strategies:
            if strategy.alive and strategy.energy <= 0:
                strategy.alive = False
                deaths += 1
                self.events.append({"epoch": epoch, "type": "strategy_died", "strategy_id": strategy.strategy_id})
        alive = [strategy for strategy in self.strategies if strategy.alive]
        if len(alive) <= max(2, self.population_size // 4):
            self.deaths += deaths
            return deaths
        ranked = sorted(alive, key=lambda item: item.energy)
        cull_count = max(0, int(len(alive) * 0.12))
        for strategy in ranked[:cull_count]:
            if strategy.energy < 8.0:
                strategy.alive = False
                deaths += 1
                self.events.append({"epoch": epoch, "type": "strategy_culled", "strategy_id": strategy.strategy_id})
        self.deaths += deaths
        return deaths

    def _reproduce(self, epoch: int) -> int:
        alive = [strategy for strategy in self.strategies if strategy.alive]
        if not alive:
            parent = self._new_strategy(len(self.strategies) + 1)
            self.strategies.append(parent)
            self.births += 1
            return 1
        births = 0
        target = self.population_size
        ranked = sorted(alive, key=lambda item: item.energy + 0.5 * len(item.successes), reverse=True)
        while len([strategy for strategy in self.strategies if strategy.alive]) < target:
            parent = ranked[births % max(1, min(5, len(ranked)))]
            child = self._mutate(parent, epoch)
            self.strategies.append(child)
            births += 1
            self.births += 1
            self.events.append(
                {
                    "epoch": epoch,
                    "type": "strategy_born",
                    "strategy_id": child.strategy_id,
                    "parent_id": parent.strategy_id,
                }
            )
        return births

    def _epoch_metrics(self, epoch: int, results: list[StrategyActionResult], births: int, deaths: int) -> EpochMetrics:
        alive = [strategy for strategy in self.strategies if strategy.alive]
        falsified = sum(result.hypotheses_falsified for result in results)
        created = sum(result.hypotheses_created for result in results)
        experiments = sum(result.discriminating_experiments for result in results)
        question_value = sum(result.valuable_questions for result in results) / max(1, len(results))
        bridge_score = sum(result.bridges_created for result in results) / max(1, len(results))
        compression_score = sum(result.compression_delta for result in results) / max(1, len(results))
        dead_end_rate = sum(result.dead_end_cost for result in results) / max(1e-9, sum(result.cost for result in results))
        species_counts = Counter(strategy.species_hint for strategy in alive)
        dominant_share = max(species_counts.values(), default=0) / max(1, len(alive))
        diversity = shannon_entropy(species_counts) / math.log(max(2, len(species_counts) or 2))
        meta_notes = self._meta_notes(alive, dominant_share, diversity, dead_end_rate)
        return EpochMetrics(
            epoch=epoch,
            population_size=self.population_size,
            alive_count=len(alive),
            births=births,
            deaths=deaths,
            selection_pressure=falsified / max(1, created + falsified),
            experiment_yield=experiments / max(1, len(results)),
            question_value=question_value,
            bridge_score=bridge_score,
            compression_score=compression_score,
            research_diversity=diversity,
            dead_end_rate=dead_end_rate,
            mean_energy=sum(strategy.energy for strategy in alive) / max(1, len(alive)),
            dominant_species_share=dominant_share,
            meta_notes=meta_notes,
        )

    def _meta_notes(
        self,
        alive: list[ResearchStrategy],
        dominant_share: float,
        diversity: float,
        dead_end_rate: float,
    ) -> list[str]:
        notes = []
        if dominant_share > 0.65:
            notes.append("One research species dominates; watch for premature convergence.")
        if diversity < 0.45:
            notes.append("Research diversity is low; mutation or niche pressure may be too weak.")
        if dead_end_rate > 0.35:
            notes.append("Dead-end resource burn is high; selection pressure should increase.")
        if alive and sum(strategy.age for strategy in alive) / len(alive) < 2:
            notes.append("Population is turning over quickly; stable species may not have emerged yet.")
        return notes

    def summary(self) -> EcosystemSummary:
        alive = [strategy for strategy in self.strategies if strategy.alive]
        species_counts = Counter(strategy.species_hint for strategy in alive)
        return EcosystemSummary(
            population_target=self.population_size,
            epochs=self.epochs,
            seed=self.seed,
            final_alive=len(alive),
            total_births=self.births,
            total_deaths=self.deaths,
            mean_selection_pressure=mean(metric.selection_pressure for metric in self.epoch_metrics),
            mean_experiment_yield=mean(metric.experiment_yield for metric in self.epoch_metrics),
            mean_question_value=mean(metric.question_value for metric in self.epoch_metrics),
            mean_bridge_score=mean(metric.bridge_score for metric in self.epoch_metrics),
            mean_compression_score=mean(metric.compression_score for metric in self.epoch_metrics),
            mean_research_diversity=mean(metric.research_diversity for metric in self.epoch_metrics),
            mean_dead_end_rate=mean(metric.dead_end_rate for metric in self.epoch_metrics),
            final_species_counts=dict(sorted(species_counts.items())),
            total_epistemic_value=self.total_epistemic_value,
        )

    def _new_strategy(self, index: int) -> ResearchStrategy:
        genome = StrategyGenome(
            criticality=self.rng.random(),
            formalization_bias=self.rng.random(),
            simulation_bias=self.rng.random(),
            analogy_bias=self.rng.random(),
            counterexample_bias=self.rng.random(),
            merge_bias=self.rng.random(),
            split_bias=self.rng.random(),
            reasoning_depth=self.rng.random(),
            quick_check_bias=self.rng.random(),
            risk_tolerance=self.rng.random(),
            collaboration_bias=self.rng.random(),
            meta_bias=self.rng.random(),
        )
        strategy = ResearchStrategy(
            strategy_id=f"s-{index:05d}",
            genome=genome,
            existence_cost=0.8 + 0.6 * self.rng.random(),
            energy=8.0 + 6.0 * self.rng.random(),
        )
        strategy.species_hint = classify_species(strategy.genome)
        return strategy

    def _mutate(self, parent: ResearchStrategy, epoch: int) -> ResearchStrategy:
        values = parent.genome.to_dict()
        mutations = []
        for key, value in values.items():
            if self.rng.random() < self.mutation_rate:
                delta = self.rng.gauss(0.0, 0.16)
                values[key] = clamp(value + delta)
                mutations.append(f"{key}:{delta:+.2f}")
        if not mutations:
            key = self.rng.choice(list(values))
            delta = self.rng.gauss(0.0, 0.1)
            values[key] = clamp(values[key] + delta)
            mutations.append(f"{key}:{delta:+.2f}")
        strategy = ResearchStrategy(
            strategy_id=f"s-{len(self.strategies) + 1:05d}",
            genome=StrategyGenome(**values),
            parent_id=parent.strategy_id,
            mutation_history=parent.mutation_history[-5:] + [f"epoch={epoch} " + ", ".join(mutations)],
            existence_cost=parent.existence_cost * (0.95 + 0.1 * self.rng.random()),
            energy=max(6.0, parent.energy * 0.45),
        )
        strategy.species_hint = classify_species(strategy.genome)
        return strategy

    def _write_json(self, name: str, data: object) -> None:
        path = self.out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def classify_species(genome: StrategyGenome) -> str:
    scores = {
        "theorist": genome.formalization_bias + genome.reasoning_depth,
        "experimenter": genome.simulation_bias + genome.quick_check_bias,
        "destroyer": genome.criticality + genome.counterexample_bias,
        "integrator": genome.analogy_bias + genome.merge_bias + genome.collaboration_bias,
        "splitter": genome.split_bias + genome.criticality,
        "meta_analyst": genome.meta_bias + genome.collaboration_bias,
    }
    return max(scores, key=scores.get)


def shannon_entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log(probability)
    return entropy


def mean(values) -> float:
    values = list(values)
    return sum(values) / max(1, len(values))

