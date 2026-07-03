from __future__ import annotations

import math
from itertools import combinations

from epistemic_engine.abstractions.models import Abstraction, Atom, ToyObject
from epistemic_engine.abstractions.phase_diagram import AblationSpec, PhaseRunMetrics, classify_regime


class LocalDynamicsAgent:
    name = "local_dynamics"
    max_condition_size = 3

    def __init__(self, ablation: AblationSpec) -> None:
        self.ablation = ablation
        self.observed: list[ToyObject] = []
        self.abstractions: dict[str, Abstraction] = {}
        self._next_id = 1
        self.operations = 0
        self.hierarchy_emergence_time = 0

    def observe(self, toy_object: ToyObject, step: int) -> None:
        self.observed.append(toy_object)
        self._reuse(toy_object, step)
        if self.ablation.operation_budget is not None and self.operations >= self.ablation.operation_budget:
            return
        self._local_update(toy_object, step)

    def _local_update(self, toy_object: ToyObject, step: int) -> None:
        raise NotImplementedError

    def _reuse(self, toy_object: ToyObject, step: int) -> None:
        for abstraction in self.abstractions.values():
            if not abstraction.alive or not abstraction.conditions <= toy_object.atoms:
                continue
            abstraction.use_count += 1
            if abstraction.expected_effects <= toy_object.effects:
                abstraction.survived_observations += 1
                abstraction.object_ids.add(toy_object.object_id)
            else:
                abstraction.failure_count += 1
                abstraction.destroyed_at = step

    def _create(
        self,
        condition: frozenset[Atom],
        effects: frozenset[str],
        step: int,
    ) -> Abstraction | None:
        if not effects:
            return None
        if self.ablation.abstraction_budget is not None:
            alive = [item for item in self.abstractions.values() if item.alive]
            if len(alive) >= self.ablation.abstraction_budget:
                weakest = min(alive, key=lambda item: (item.life_reuse, item.use_count, -len(item.conditions)))
                weakest.destroyed_at = step
        for abstraction in self.abstractions.values():
            if abstraction.alive and abstraction.conditions == condition:
                return None
        parents = self._parents_for(condition)
        depth = 1 + max((self.abstractions[parent].depth for parent in parents), default=0)
        if depth > self.ablation.max_depth:
            return None
        abstraction = Abstraction(
            abstraction_id=f"{self.name}_{self._next_id}",
            parents=parents,
            conditions=condition,
            expected_effects=effects,
            object_ids={obj.object_id for obj in self._matching(condition)},
            created_at=step,
            depth=depth,
        )
        self._next_id += 1
        self.abstractions[abstraction.abstraction_id] = abstraction
        for parent in parents:
            self.abstractions[parent].children.add(abstraction.abstraction_id)
            self.abstractions[parent].composition_count += 1
        if depth > 1 and self.hierarchy_emergence_time == 0:
            self.hierarchy_emergence_time = step
        self.operations += 1
        return abstraction

    def _candidate_conditions(self, toy_object: ToyObject | None = None) -> set[frozenset[Atom]]:
        atoms = sorted(self._frequent_atoms() if toy_object is None else toy_object.atoms)
        max_size = self.max_condition_size
        if not self.ablation.allow_composition:
            max_size = min(max_size, 1)
        candidates: set[frozenset[Atom]] = set()
        for size in range(1, max_size + 1):
            for combo in combinations(atoms, size):
                keys = [key for key, _ in combo]
                if len(set(keys)) != len(keys):
                    continue
                condition = frozenset(combo)
                if any(abstraction.alive and abstraction.conditions == condition for abstraction in self.abstractions.values()):
                    continue
                candidates.add(condition)
                if len(candidates) >= 160:
                    return candidates
        return candidates

    def _frequent_atoms(self) -> list[Atom]:
        counts: dict[Atom, int] = {}
        for obj in self.observed:
            for atom in obj.atoms:
                counts[atom] = counts.get(atom, 0) + 1
        return [
            atom
            for atom, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ][:10]

    def _parents_for(self, condition: frozenset[Atom]) -> tuple[str, ...]:
        if len(condition) <= 1 or not self.ablation.allow_composition:
            return ()
        parents = [
            abstraction
            for abstraction in self.abstractions.values()
            if abstraction.alive and abstraction.conditions < condition
        ]
        parents.sort(key=lambda item: (-len(item.conditions), -item.life_reuse))
        selected: list[str] = []
        covered: set[Atom] = set()
        for parent in parents:
            if len(selected) >= 2:
                break
            selected.append(parent.abstraction_id)
            covered |= set(parent.conditions)
            if covered >= condition:
                break
        return tuple(selected) if covered >= condition else ()

    def _matching(self, condition: frozenset[Atom]) -> list[ToyObject]:
        return [obj for obj in self.observed if condition <= obj.atoms]

    def _shared_effects(self, objects: list[ToyObject]) -> frozenset[str]:
        if not objects:
            return frozenset()
        shared = set(objects[0].effects)
        for obj in objects[1:]:
            shared &= obj.effects
        return frozenset(shared)

    def _validity(self, condition: frozenset[Atom], effects: frozenset[str]) -> float:
        objects = self._matching(condition)
        if not objects:
            return 0.0
        return sum(1 for obj in objects if effects <= obj.effects) / len(objects)

    def _effect_universe(self) -> tuple[str, ...]:
        universe = sorted({effect for obj in self.observed for effect in obj.effects})
        return tuple(universe)

    def metrics(self, transfer: list[ToyObject]) -> PhaseRunMetrics:
        abstractions = list(self.abstractions.values())
        alive = [item for item in abstractions if item.alive]
        valid_transfer = 0
        raw_transfer = 0
        for obj in transfer:
            for abstraction in alive:
                if abstraction.conditions <= obj.atoms:
                    raw_transfer += 1
                    if abstraction.expected_effects <= obj.effects:
                        valid_transfer += 1
        semantic_edges = 0
        total_edges = 0
        for child in abstractions:
            for parent_id in child.parents:
                if parent_id not in self.abstractions:
                    continue
                total_edges += 1
                parent = self.abstractions[parent_id]
                if parent.conditions <= child.conditions and parent.expected_effects <= child.expected_effects:
                    semantic_edges += 1
        depth = max((item.depth for item in abstractions), default=0)
        deaths = sum(1 for item in abstractions if not item.alive)
        regime = classify_regime(
            abstraction_count=len(abstractions),
            alive_count=len(alive),
            depth=depth,
            width=self._dag_width(alive),
            mean_reuse=_mean([item.use_count for item in abstractions]),
            deaths=deaths,
        )
        return PhaseRunMetrics(
            objective=self.name,
            world="",
            ablation=self.ablation.name,
            seed=0,
            abstraction_count=len(abstractions),
            alive_count=len(alive),
            graph_depth=depth,
            dag_width=self._dag_width(alive),
            mean_lifetime=_mean([item.lifetime for item in abstractions]),
            mean_exposure_lifetime=_mean([item.survived_observations for item in abstractions]),
            mean_reuse=_mean([item.use_count for item in abstractions]),
            valid_reuse=sum(item.survived_observations for item in abstractions),
            raw_transfer_reuse=raw_transfer,
            valid_transfer_reuse=valid_transfer,
            transfer_correctness=valid_transfer / raw_transfer if raw_transfer else 0.0,
            graph_compression_ratio=len(self.observed) / max(1, len(alive)),
            semantic_subsumption_ratio=semantic_edges / total_edges if total_edges else 0.0,
            branching_factor=_mean([len(item.children) for item in abstractions]),
            hierarchy_emergence_time=self.hierarchy_emergence_time,
            births=len(abstractions),
            deaths=deaths,
            regime=regime,
        )

    @staticmethod
    def _dag_width(alive: list[Abstraction]) -> int:
        by_depth: dict[int, int] = {}
        for abstraction in alive:
            by_depth[abstraction.depth] = by_depth.get(abstraction.depth, 0) + 1
        return max(by_depth.values(), default=0)


class MergeDrivenDynamics(LocalDynamicsAgent):
    name = "merge_driven"
    max_condition_size = 3

    def _local_update(self, toy_object: ToyObject, step: int) -> None:
        scored = []
        for condition in self._candidate_conditions():
            objects = self._matching(condition)
            if len(objects) < 2:
                continue
            effects = self._shared_effects(objects)
            if not effects:
                continue
            scored.append((len(objects) / len(condition), condition, effects))
        if scored:
            _, condition, effects = max(scored, key=lambda item: (item[0], -len(item[1]), sorted(item[1])))
            self._create(condition, effects, step)


class PredictionDrivenDynamics(LocalDynamicsAgent):
    name = "prediction_driven"
    max_condition_size = 4

    def _local_update(self, toy_object: ToyObject, step: int) -> None:
        prediction = set()
        for abstraction in self.abstractions.values():
            if abstraction.alive and abstraction.conditions <= toy_object.atoms:
                prediction |= set(abstraction.expected_effects)
        missing = toy_object.effects - prediction
        false_positive = prediction - toy_object.effects
        if not missing and not false_positive:
            return
        scored = []
        for condition in self._candidate_conditions(toy_object):
            objects = self._matching(condition)
            if len(objects) < 2:
                continue
            effects = self._shared_effects(objects)
            if not effects:
                continue
            repair = len(effects & missing) + 0.5 * len(false_positive - effects)
            if repair <= 0:
                continue
            scored.append((repair / len(condition), condition, effects))
        if scored:
            _, condition, effects = max(scored, key=lambda item: (item[0], -len(item[1]), sorted(item[1])))
            self._create(condition, effects, step)


class ConstraintDrivenDynamics(LocalDynamicsAgent):
    name = "constraint_driven"
    max_condition_size = 3

    def _local_update(self, toy_object: ToyObject, step: int) -> None:
        scored = []
        for condition in self._candidate_conditions():
            objects = self._matching(condition)
            if len(objects) < 2:
                continue
            effects = self._shared_effects(objects)
            if not effects:
                continue
            if self._validity(condition, effects) < 1.0:
                continue
            support = len(objects)
            specificity_penalty = len(condition) * 0.2
            scored.append((support - specificity_penalty, condition, effects))
        if scored:
            _, condition, effects = max(scored, key=lambda item: (item[0], -len(item[1]), sorted(item[1])))
            self._create(condition, effects, step)


class MDLLikeDynamics(LocalDynamicsAgent):
    name = "mdl_like"
    max_condition_size = 4

    def _local_update(self, toy_object: ToyObject, step: int) -> None:
        current_cost = self._description_cost()
        scored = []
        for condition in self._candidate_conditions():
            objects = self._matching(condition)
            if len(objects) < 2:
                continue
            effects = self._shared_effects(objects)
            if not effects:
                continue
            candidate_cost = self._description_cost(extra=(condition, effects))
            gain = current_cost - candidate_cost
            if gain > 0:
                scored.append((gain, condition, effects))
        if scored:
            _, condition, effects = max(scored, key=lambda item: (item[0], -len(item[1]), sorted(item[1])))
            self._create(condition, effects, step)
            self._prune_mdl(step)

    def _description_cost(self, extra: tuple[frozenset[Atom], frozenset[str]] | None = None) -> float:
        alive = [item for item in self.abstractions.values() if item.alive]
        model_cost = sum(len(item.conditions) + len(item.expected_effects) for item in alive)
        errors = 0
        for obj in self.observed:
            predicted: set[str] = set()
            for abstraction in alive:
                if abstraction.conditions <= obj.atoms:
                    predicted |= set(abstraction.expected_effects)
            if extra is not None and extra[0] <= obj.atoms:
                predicted |= set(extra[1])
            errors += len(predicted ^ obj.effects)
        if extra is not None:
            model_cost += len(extra[0]) + len(extra[1])
        return model_cost * 1.5 + errors * 3.0

    def _prune_mdl(self, step: int) -> None:
        alive = [item for item in self.abstractions.values() if item.alive]
        if len(alive) <= 2:
            return
        baseline = self._description_cost()
        for abstraction in sorted(alive, key=lambda item: (item.life_reuse, item.use_count)):
            abstraction.destroyed_at = step
            if self._description_cost() <= baseline:
                return
            abstraction.destroyed_at = None


class InformationBottleneckLikeDynamics(LocalDynamicsAgent):
    name = "information_bottleneck_like"
    max_condition_size = 3

    def _local_update(self, toy_object: ToyObject, step: int) -> None:
        scored = []
        for condition in self._candidate_conditions():
            objects = self._matching(condition)
            if len(objects) < 2:
                continue
            effects = self._shared_effects(objects)
            if not effects:
                continue
            information = self._mutual_information_proxy(condition)
            compression_penalty = 0.8 * math.log2(1 + len(condition))
            bottleneck_score = information - compression_penalty
            if bottleneck_score > 0:
                scored.append((bottleneck_score, condition, effects))
        if scored:
            _, condition, effects = max(scored, key=lambda item: (item[0], -len(item[1]), sorted(item[1])))
            self._create(condition, effects, step)

    def _mutual_information_proxy(self, condition: frozenset[Atom]) -> float:
        match = self._matching(condition)
        nonmatch = [obj for obj in self.observed if not condition <= obj.atoms]
        if len(match) < 2 or len(nonmatch) < 2:
            return 0.0
        universe = self._effect_universe()
        if not universe:
            return 0.0
        return sum(abs(_effect_rate(match, effect) - _effect_rate(nonmatch, effect)) for effect in universe)


def dynamics_agent_classes() -> list[type[LocalDynamicsAgent]]:
    return [
        MergeDrivenDynamics,
        PredictionDrivenDynamics,
        ConstraintDrivenDynamics,
        MDLLikeDynamics,
        InformationBottleneckLikeDynamics,
    ]


def _effect_rate(objects: list[ToyObject], effect: str) -> float:
    return sum(1 for obj in objects if effect in obj.effects) / len(objects)


def _mean(values: list[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
