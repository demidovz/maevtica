from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import combinations

from epistemic_engine.abstractions.models import Abstraction, Atom, ToyObject
from epistemic_engine.abstractions.world import COLORS, MATERIALS, SHAPES, SIZES, TEXTURES


FEATURE_VALUES: dict[str, tuple[str, ...]] = {
    "shape": SHAPES,
    "color": COLORS,
    "size": SIZES,
    "texture": TEXTURES,
    "material": MATERIALS,
}


@dataclass(frozen=True)
class PhaseWorldSpec:
    name: str
    noise: float = 0.0
    partial_observability: float = 1.0
    transfer_mode: str = "iid"
    compositionality: float = 1.0
    regularity: float = 1.0
    volatility: float = 0.0


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    validity_weight: float = 0.0
    reuse_weight: float = 0.0
    lifetime_weight: float = 0.0
    compression_weight: float = 0.0
    novelty_weight: float = 0.0
    depth_weight: float = 0.0
    mdl_penalty: float = 0.0
    max_condition_size: int = 3
    exact_memory: bool = False


@dataclass(frozen=True)
class AblationSpec:
    name: str
    allow_composition: bool = True
    max_depth: int = 4
    abstraction_budget: int | None = None
    operation_budget: int | None = None
    shuffle_stream: bool = False


@dataclass(frozen=True)
class PhaseRunMetrics:
    objective: str
    world: str
    ablation: str
    seed: int
    abstraction_count: int
    alive_count: int
    graph_depth: int
    dag_width: int
    mean_lifetime: float
    mean_exposure_lifetime: float
    mean_reuse: float
    valid_reuse: int
    raw_transfer_reuse: int
    valid_transfer_reuse: int
    transfer_correctness: float
    graph_compression_ratio: float
    semantic_subsumption_ratio: float
    branching_factor: float
    hierarchy_emergence_time: int
    births: int
    deaths: int
    regime: str


class PhaseWorld:
    def __init__(self, spec: PhaseWorldSpec, seed: int) -> None:
        self.spec = spec
        self.rng = random.Random(seed)

    def train(self, count: int) -> list[ToyObject]:
        return [self._sample(f"train_{index}", index, transfer=False) for index in range(count)]

    def transfer(self, count: int) -> list[ToyObject]:
        return [self._sample(f"transfer_{index}", index, transfer=True) for index in range(count)]

    def _sample(self, object_id: str, index: int, *, transfer: bool) -> ToyObject:
        features = {key: self.rng.choice(values) for key, values in FEATURE_VALUES.items()}

        if transfer and self.spec.transfer_mode == "compositional" and index % 2 == 0:
            features["shape"] = "round"
            features["color"] = "red"
        if transfer and self.spec.transfer_mode == "anti_compositional":
            if features["shape"] == "round" and features["color"] == "red":
                features["color"] = "green"

        visible = self._visible_features(features)
        effects = self._effects(features, index=index, transfer=transfer)
        return ToyObject(object_id, visible, effects)

    def _visible_features(self, features: dict[str, str]) -> dict[str, str]:
        if self.spec.partial_observability >= 1.0:
            return dict(features)
        visible: dict[str, str] = {}
        for key, value in features.items():
            if self.rng.random() <= self.spec.partial_observability:
                visible[key] = value
        if not visible:
            key = self.rng.choice(tuple(features))
            visible[key] = features[key]
        return visible

    def _effects(self, features: dict[str, str], *, index: int, transfer: bool) -> frozenset[str]:
        effects: set[str] = set()

        if self.spec.name in {"random", "no_regularities"}:
            if self.spec.name == "random":
                pool = ("e0", "e1", "e2", "e3", "e4")
                effects = {effect for effect in pool if self.rng.random() < 0.35}
            return self._with_noise(effects)

        if self.rng.random() < self.spec.regularity and features["shape"] == "round":
            effects.add("rolls")
        if self.rng.random() < self.spec.regularity and features["size"] == "large":
            effects.add("leaves_trace")
        if self.rng.random() < self.spec.regularity and features["color"] == "red":
            effects.add("heats")
        if self.rng.random() < self.spec.regularity and features["material"] == "metal":
            effects.add("conducts")
        if self.spec.name in {"hierarchical", "compositional", "deterministic", "cyclic", "changing", "noisy", "partial"}:
            if self.rng.random() < self.spec.regularity and features["texture"] == "striped":
                effects.add("camouflages")

        if self.spec.name in {"hierarchical", "compositional", "cyclic", "changing", "noisy", "partial"}:
            if self.rng.random() < self.spec.compositionality and features["shape"] == "round" and features["color"] == "red":
                effects.add("glows")
            if self.rng.random() < self.spec.compositionality and features["size"] == "small" and features["texture"] == "striped":
                effects.add("hides")
            if self.rng.random() < self.spec.compositionality and features["material"] == "metal" and features["color"] == "blue":
                effects.add("magnetizes")
            if self.rng.random() < self.spec.compositionality and features["shape"] == "flat" and features["material"] == "wood":
                effects.add("floats")

        if self.spec.name == "adversarial":
            if features["shape"] == "round":
                effects.discard("rolls")
            if features["shape"] == "angular":
                effects.add("rolls")

        if self.spec.name == "cyclic":
            phase = index % 3
            if phase == 1:
                effects = {effect for effect in effects if effect not in {"glows", "hides"}}
            elif phase == 2:
                effects.add("phase_shift")

        if (self.spec.name == "changing" or self.spec.volatility > 0) and not transfer:
            # Mid-stream regime switch: red stops being reliable, blue starts heating.
            if index >= 60 or self.rng.random() < self.spec.volatility:
                effects.discard("heats")
                if features["color"] == "blue":
                    effects.add("heats")

        return self._with_noise(effects)

    def _with_noise(self, effects: set[str]) -> frozenset[str]:
        if self.spec.noise <= 0:
            return frozenset(effects)
        universe = {"rolls", "leaves_trace", "heats", "conducts", "camouflages", "glows", "hides", "magnetizes", "floats"}
        noisy = set(effects)
        for effect in universe:
            if self.rng.random() < self.spec.noise:
                if effect in noisy:
                    noisy.remove(effect)
                else:
                    noisy.add(effect)
        return frozenset(noisy)


class ObjectiveGrowthAgent:
    def __init__(self, objective: ObjectiveSpec, ablation: AblationSpec) -> None:
        self.objective = objective
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
        if self.objective.exact_memory:
            self._create(frozenset(toy_object.atoms), toy_object.effects, (), step, 1)
            return
        self._propose_best(step)

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

    def _propose_best(self, step: int) -> None:
        candidates = self._candidate_conditions()
        scored = [(self._score_candidate(condition), condition) for condition in candidates]
        scored = [(score, condition) for score, condition in scored if score > 0]
        scored.sort(key=lambda item: (-item[0], sorted(item[1])))
        for _, condition in scored[:2]:
            objects = self._matching(condition)
            effects = self._shared_effects(objects)
            if not effects:
                continue
            parents = self._parents_for(condition)
            depth = 1 + max((self.abstractions[parent].depth for parent in parents), default=0)
            if depth > self.ablation.max_depth:
                continue
            created = self._create(condition, effects, parents, step, depth)
            if created is not None:
                self.operations += 1
                if depth > 1 and self.hierarchy_emergence_time == 0:
                    self.hierarchy_emergence_time = step
                return

    def _candidate_conditions(self) -> set[frozenset[Atom]]:
        counts: dict[Atom, int] = {}
        for obj in self.observed:
            for atom in obj.atoms:
                counts[atom] = counts.get(atom, 0) + 1
        atoms = [
            atom
            for atom, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ][:10]
        candidates: set[frozenset[Atom]] = set()
        max_size = self.objective.max_condition_size
        if not self.ablation.allow_composition:
            max_size = min(max_size, 1)
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

    def _score_candidate(self, condition: frozenset[Atom]) -> float:
        objects = self._matching(condition)
        if len(objects) < 2:
            return 0.0
        shared_effects = self._shared_effects(objects)
        if not shared_effects:
            return 0.0
        validity = self._validity(condition, shared_effects)
        reuse = len(objects)
        novelty = len(shared_effects - self._covered_effects(condition))
        compression = max(0.0, reuse / max(1, len(condition)))
        depth_bonus = max(0, len(condition) - 1)
        lifetime_proxy = reuse * validity
        score = (
            self.objective.validity_weight * validity
            + self.objective.reuse_weight * reuse
            + self.objective.lifetime_weight * lifetime_proxy
            + self.objective.compression_weight * compression
            + self.objective.novelty_weight * novelty
            + self.objective.depth_weight * depth_bonus
            - self.objective.mdl_penalty * len(condition)
        )
        return score

    def _create(
        self,
        condition: frozenset[Atom],
        effects: frozenset[str],
        parents: tuple[str, ...],
        step: int,
        depth: int,
    ) -> Abstraction | None:
        if self.ablation.abstraction_budget is not None:
            alive = [item for item in self.abstractions.values() if item.alive]
            if len(alive) >= self.ablation.abstraction_budget:
                weakest = min(alive, key=lambda item: (item.life_reuse, item.use_count, -len(item.conditions)))
                weakest.destroyed_at = step
        for abstraction in self.abstractions.values():
            if abstraction.alive and abstraction.conditions == condition:
                return None
        abstraction = Abstraction(
            abstraction_id=f"{self.objective.name}_{self._next_id}",
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
        return abstraction

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
        regime = classify_regime(
            abstraction_count=len(abstractions),
            alive_count=len(alive),
            depth=depth,
            width=self._dag_width(alive),
            mean_reuse=_mean([item.use_count for item in abstractions]),
            deaths=sum(1 for item in abstractions if not item.alive),
        )
        return PhaseRunMetrics(
            objective=self.objective.name,
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
            deaths=sum(1 for item in abstractions if not item.alive),
            regime=regime,
        )

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

    def _validity(self, condition: frozenset[Atom], effects: frozenset[str]) -> float:
        objects = self._matching(condition)
        if not objects:
            return 0.0
        valid = sum(1 for obj in objects if effects <= obj.effects)
        return valid / len(objects)

    def _covered_effects(self, condition: frozenset[Atom]) -> set[str]:
        covered: set[str] = set()
        for abstraction in self.abstractions.values():
            if abstraction.alive and abstraction.conditions < condition:
                covered |= set(abstraction.expected_effects)
        return covered

    @staticmethod
    def _shared_effects(objects: list[ToyObject]) -> frozenset[str]:
        if not objects:
            return frozenset()
        shared = set(objects[0].effects)
        for obj in objects[1:]:
            shared &= obj.effects
        return frozenset(shared)

    @staticmethod
    def _dag_width(alive: list[Abstraction]) -> int:
        by_depth: dict[int, int] = {}
        for abstraction in alive:
            by_depth[abstraction.depth] = by_depth.get(abstraction.depth, 0) + 1
        return max(by_depth.values(), default=0)


def classify_regime(
    *,
    abstraction_count: int,
    alive_count: int,
    depth: int,
    width: int,
    mean_reuse: float,
    deaths: int,
) -> str:
    if abstraction_count == 0 or alive_count == 0:
        return "collapse"
    if abstraction_count > 80 and mean_reuse < 2:
        return "fragmentation"
    if depth >= 4:
        return "deep_hierarchy"
    if depth >= 3:
        return "abstraction_dag"
    if depth == 2 and mean_reuse >= 4:
        return "reusable_feature_graph"
    if depth <= 1 and mean_reuse >= 4:
        return "flat_reusable_features"
    if depth <= 1:
        return "flat_memory"
    if deaths > abstraction_count * 0.7:
        return "unstable_churn"
    return "mixed"


def objective_specs() -> list[ObjectiveSpec]:
    return [
        ObjectiveSpec("accuracy", validity_weight=20.0, max_condition_size=2),
        ObjectiveSpec("prediction", validity_weight=12.0, novelty_weight=2.0, max_condition_size=3),
        ObjectiveSpec("compression", compression_weight=5.0, mdl_penalty=0.5, max_condition_size=2),
        ObjectiveSpec("mdl", compression_weight=4.0, validity_weight=4.0, mdl_penalty=3.0, max_condition_size=3),
        ObjectiveSpec("lifetime", lifetime_weight=2.0, validity_weight=2.0, max_condition_size=2),
        ObjectiveSpec("reuse", reuse_weight=2.0, max_condition_size=2),
        ObjectiveSpec("lifetime_reuse", lifetime_weight=2.0, reuse_weight=2.0, validity_weight=1.0, depth_weight=0.5, max_condition_size=3),
        ObjectiveSpec("novelty", novelty_weight=10.0, depth_weight=1.0, max_condition_size=3),
        ObjectiveSpec("information_gain", novelty_weight=5.0, validity_weight=5.0, max_condition_size=3),
        ObjectiveSpec("curiosity", novelty_weight=7.0, reuse_weight=1.0, depth_weight=1.5, max_condition_size=4),
        ObjectiveSpec("compression_prediction", compression_weight=3.0, validity_weight=8.0, mdl_penalty=1.0, max_condition_size=3),
        ObjectiveSpec("prediction_lifetime", validity_weight=8.0, lifetime_weight=2.0, max_condition_size=3),
        ObjectiveSpec("exact_memory", exact_memory=True, max_condition_size=5),
    ]


def world_specs() -> list[PhaseWorldSpec]:
    return [
        PhaseWorldSpec("random"),
        PhaseWorldSpec("no_regularities"),
        PhaseWorldSpec("deterministic"),
        PhaseWorldSpec("hierarchical"),
        PhaseWorldSpec("compositional", transfer_mode="compositional"),
        PhaseWorldSpec("cyclic"),
        PhaseWorldSpec("changing"),
        PhaseWorldSpec("noisy", noise=0.1),
        PhaseWorldSpec("partial", partial_observability=0.6),
        PhaseWorldSpec("adversarial"),
        PhaseWorldSpec("iid", transfer_mode="iid"),
        PhaseWorldSpec("anti_compositional", transfer_mode="anti_compositional"),
    ]


def reviewer_ablations() -> list[AblationSpec]:
    return [
        AblationSpec("standard"),
        AblationSpec("equal_operation_budget", operation_budget=35),
        AblationSpec("equal_abstraction_budget", abstraction_budget=35),
        AblationSpec("shuffle_stream", shuffle_stream=True),
        AblationSpec("remove_composition", allow_composition=False),
        AblationSpec("remove_hierarchy", max_depth=1),
    ]


def _mean(values: list[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
