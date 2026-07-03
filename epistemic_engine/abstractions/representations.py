from __future__ import annotations

from itertools import combinations

from epistemic_engine.abstractions.models import Atom, HyperedgeAbstraction, ToyObject
from epistemic_engine.abstractions.phase_diagram import (
    AblationSpec,
    ObjectiveSpec,
    PhaseRunMetrics,
    classify_regime,
)


class HypergraphObjectiveGrowthAgent:
    """Alternative internal representation for implementation-invariance tests.

    A learned abstraction is a directed hyperedge:

        source_edges + tail_atoms -> head_effects

    The agent exposes the same macro-metrics as the DAG implementation, but it
    does not store abstractions as parent/child graph nodes.
    """

    def __init__(self, objective: ObjectiveSpec, ablation: AblationSpec) -> None:
        self.objective = objective
        self.ablation = ablation
        self.observed: list[ToyObject] = []
        self.hyperedges: dict[str, HyperedgeAbstraction] = {}
        self._next_id = 1
        self.operations = 0
        self.hierarchy_emergence_time = 0

    def observe(self, toy_object: ToyObject, step: int) -> None:
        self.observed.append(toy_object)
        self._reuse(toy_object, step)
        if self.ablation.operation_budget is not None and self.operations >= self.ablation.operation_budget:
            return
        if self.objective.exact_memory:
            self._create(frozenset(toy_object.atoms), toy_object.effects, frozenset(), step, 1)
            return
        self._propose_best(step)

    def _reuse(self, toy_object: ToyObject, step: int) -> None:
        for edge in self.hyperedges.values():
            if not edge.alive or not edge.tail_atoms <= toy_object.atoms:
                continue
            edge.use_count += 1
            if edge.head_effects <= toy_object.effects:
                edge.survived_observations += 1
                edge.object_ids.add(toy_object.object_id)
            else:
                edge.failure_count += 1
                edge.destroyed_at = step

    def _propose_best(self, step: int) -> None:
        candidates = self._candidate_tails()
        scored = [(self._score_candidate(tail), tail) for tail in candidates]
        scored = [(score, tail) for score, tail in scored if score > 0]
        scored.sort(key=lambda item: (-item[0], sorted(item[1])))
        for _, tail in scored[:2]:
            objects = self._matching(tail)
            effects = self._shared_effects(objects)
            if not effects:
                continue
            sources = self._source_edges_for(tail)
            rank = 1 + max((self.hyperedges[source].rank for source in sources), default=0)
            if rank > self.ablation.max_depth:
                continue
            created = self._create(tail, effects, sources, step, rank)
            if created is not None:
                self.operations += 1
                if rank > 1 and self.hierarchy_emergence_time == 0:
                    self.hierarchy_emergence_time = step
                return

    def _candidate_tails(self) -> set[frozenset[Atom]]:
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
                tail = frozenset(combo)
                if any(edge.alive and edge.tail_atoms == tail for edge in self.hyperedges.values()):
                    continue
                candidates.add(tail)
                if len(candidates) >= 160:
                    return candidates
        return candidates

    def _score_candidate(self, tail: frozenset[Atom]) -> float:
        objects = self._matching(tail)
        if len(objects) < 2:
            return 0.0
        shared_effects = self._shared_effects(objects)
        if not shared_effects:
            return 0.0
        validity = self._validity(tail, shared_effects)
        reuse = len(objects)
        novelty = len(shared_effects - self._covered_effects(tail))
        compression = max(0.0, reuse / max(1, len(tail)))
        rank_bonus = max(0, len(tail) - 1)
        lifetime_proxy = reuse * validity
        return (
            self.objective.validity_weight * validity
            + self.objective.reuse_weight * reuse
            + self.objective.lifetime_weight * lifetime_proxy
            + self.objective.compression_weight * compression
            + self.objective.novelty_weight * novelty
            + self.objective.depth_weight * rank_bonus
            - self.objective.mdl_penalty * len(tail)
        )

    def _create(
        self,
        tail: frozenset[Atom],
        effects: frozenset[str],
        sources: frozenset[str],
        step: int,
        rank: int,
    ) -> HyperedgeAbstraction | None:
        if self.ablation.abstraction_budget is not None:
            alive = [item for item in self.hyperedges.values() if item.alive]
            if len(alive) >= self.ablation.abstraction_budget:
                weakest = min(alive, key=lambda item: (item.life_reuse, item.use_count, -len(item.tail_atoms)))
                weakest.destroyed_at = step
        for edge in self.hyperedges.values():
            if edge.alive and edge.tail_atoms == tail:
                return None
        edge = HyperedgeAbstraction(
            edge_id=f"{self.objective.name}_h{self._next_id}",
            tail_atoms=tail,
            head_effects=effects,
            source_edges=sources,
            object_ids={obj.object_id for obj in self._matching(tail)},
            created_at=step,
            rank=rank,
        )
        self._next_id += 1
        self.hyperedges[edge.edge_id] = edge
        return edge

    def metrics(self, transfer: list[ToyObject]) -> PhaseRunMetrics:
        edges = list(self.hyperedges.values())
        alive = [item for item in edges if item.alive]
        valid_transfer = 0
        raw_transfer = 0
        for obj in transfer:
            for edge in alive:
                if edge.tail_atoms <= obj.atoms:
                    raw_transfer += 1
                    if edge.head_effects <= obj.effects:
                        valid_transfer += 1

        semantic_edges = 0
        total_source_edges = 0
        for edge in edges:
            for source_id in edge.source_edges:
                if source_id not in self.hyperedges:
                    continue
                total_source_edges += 1
                source = self.hyperedges[source_id]
                if source.tail_atoms <= edge.tail_atoms and source.head_effects <= edge.head_effects:
                    semantic_edges += 1

        rank = max((item.rank for item in edges), default=0)
        width = self._rank_width(alive)
        deaths = sum(1 for item in edges if not item.alive)
        regime = classify_regime(
            abstraction_count=len(edges),
            alive_count=len(alive),
            depth=rank,
            width=width,
            mean_reuse=_mean([item.use_count for item in edges]),
            deaths=deaths,
        )
        return PhaseRunMetrics(
            objective=self.objective.name,
            world="",
            ablation=self.ablation.name,
            seed=0,
            abstraction_count=len(edges),
            alive_count=len(alive),
            graph_depth=rank,
            dag_width=width,
            mean_lifetime=_mean([item.lifetime for item in edges]),
            mean_exposure_lifetime=_mean([item.survived_observations for item in edges]),
            mean_reuse=_mean([item.use_count for item in edges]),
            valid_reuse=sum(item.survived_observations for item in edges),
            raw_transfer_reuse=raw_transfer,
            valid_transfer_reuse=valid_transfer,
            transfer_correctness=valid_transfer / raw_transfer if raw_transfer else 0.0,
            graph_compression_ratio=len(self.observed) / max(1, len(alive)),
            semantic_subsumption_ratio=semantic_edges / total_source_edges if total_source_edges else 0.0,
            branching_factor=_mean([self._outgoing_count(item.edge_id) for item in edges]),
            hierarchy_emergence_time=self.hierarchy_emergence_time,
            births=len(edges),
            deaths=deaths,
            regime=regime,
        )

    def _source_edges_for(self, tail: frozenset[Atom]) -> frozenset[str]:
        if len(tail) <= 1 or not self.ablation.allow_composition:
            return frozenset()
        sources = [
            edge
            for edge in self.hyperedges.values()
            if edge.alive and edge.tail_atoms < tail
        ]
        sources.sort(key=lambda item: (-len(item.tail_atoms), -item.life_reuse))
        selected: list[str] = []
        covered: set[Atom] = set()
        for source in sources:
            if len(selected) >= 3:
                break
            selected.append(source.edge_id)
            covered |= set(source.tail_atoms)
            if covered >= tail:
                break
        return frozenset(selected) if covered >= tail else frozenset()

    def _matching(self, tail: frozenset[Atom]) -> list[ToyObject]:
        return [obj for obj in self.observed if tail <= obj.atoms]

    def _validity(self, tail: frozenset[Atom], effects: frozenset[str]) -> float:
        objects = self._matching(tail)
        if not objects:
            return 0.0
        valid = sum(1 for obj in objects if effects <= obj.effects)
        return valid / len(objects)

    def _covered_effects(self, tail: frozenset[Atom]) -> set[str]:
        covered: set[str] = set()
        for edge in self.hyperedges.values():
            if edge.alive and edge.tail_atoms < tail:
                covered |= set(edge.head_effects)
        return covered

    def _outgoing_count(self, edge_id: str) -> int:
        return sum(1 for edge in self.hyperedges.values() if edge_id in edge.source_edges)

    @staticmethod
    def _shared_effects(objects: list[ToyObject]) -> frozenset[str]:
        if not objects:
            return frozenset()
        shared = set(objects[0].effects)
        for obj in objects[1:]:
            shared &= obj.effects
        return frozenset(shared)

    @staticmethod
    def _rank_width(alive: list[HyperedgeAbstraction]) -> int:
        by_rank: dict[int, int] = {}
        for edge in alive:
            by_rank[edge.rank] = by_rank.get(edge.rank, 0) + 1
        return max(by_rank.values(), default=0)


def _mean(values: list[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
