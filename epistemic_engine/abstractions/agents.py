from __future__ import annotations

from itertools import combinations

from epistemic_engine.abstractions.models import (
    Abstraction,
    Atom,
    ExperimentSummary,
    ObjectObservation,
    StepMetrics,
    ToyObject,
)


class AbstractionAgent:
    def __init__(self, name: str, *, max_condition_size: int = 3) -> None:
        self.name = name
        self.max_condition_size = max_condition_size
        self.abstractions: dict[str, Abstraction] = {}
        self.observations: list[ObjectObservation] = []
        self._next_id = 1
        self.merges = 0
        self.splits = 0
        self.destroyed = 0
        self.created = 0
        self.compositions = 0

    def observe(self, toy_object: ToyObject, step: int) -> None:
        observation = ObjectObservation(step, toy_object)
        self.observations.append(observation)
        self.reuse_existing_abstraction(observation)
        self.propose_merge(observation)
        self.compose_abstractions(observation)

    def reuse_existing_abstraction(self, observation: ObjectObservation) -> None:
        for abstraction in list(self.abstractions.values()):
            if not abstraction.alive:
                continue
            if not abstraction.conditions <= observation.toy_object.atoms:
                continue

            abstraction.use_count += 1
            if abstraction.expected_effects <= observation.toy_object.effects:
                abstraction.survived_observations += 1
                abstraction.object_ids.add(observation.toy_object.object_id)
            else:
                self.reject_abstraction(abstraction, observation.step)
                self.propose_split(abstraction, observation)

    def propose_merge(self, observation: ObjectObservation) -> None:
        for atom in sorted(observation.toy_object.atoms):
            objects = self._objects_matching(frozenset((atom,)))
            if len(objects) < 2:
                continue
            shared_effects = self._shared_effects(objects)
            if not shared_effects:
                continue
            self._create_abstraction(
                parents=(),
                conditions=frozenset((atom,)),
                expected_effects=shared_effects,
                object_ids={obj.object_id for obj in objects},
                created_at=observation.step,
                depth=1,
            )

    def propose_split(self, failed: Abstraction, observation: ObjectObservation) -> None:
        if len(failed.conditions) >= self.max_condition_size:
            return
        candidates = sorted(observation.toy_object.atoms - failed.conditions)
        for atom in candidates:
            conditions = failed.conditions | frozenset((atom,))
            objects = self._objects_matching(conditions)
            if len(objects) < 2:
                continue
            shared_effects = self._shared_effects(objects)
            if not shared_effects:
                continue
            child = self._create_abstraction(
                parents=(failed.abstraction_id,),
                conditions=conditions,
                expected_effects=shared_effects,
                object_ids={obj.object_id for obj in objects},
                created_at=observation.step,
                depth=failed.depth + 1,
            )
            if child is not None:
                failed.children.add(child.abstraction_id)
                self.splits += 1
            return

    def reject_abstraction(self, abstraction: Abstraction, step: int) -> None:
        abstraction.failure_count += 1
        abstraction.destroyed_at = step
        self.destroyed += 1

    def compose_abstractions(self, observation: ObjectObservation) -> None:
        matching = [
            abstraction
            for abstraction in self.abstractions.values()
            if abstraction.alive and abstraction.conditions <= observation.toy_object.atoms
        ]
        matching.sort(key=lambda item: (-item.life_reuse, item.abstraction_id))

        for left, right in combinations(matching[:8], 2):
            conditions = left.conditions | right.conditions
            if len(conditions) <= max(len(left.conditions), len(right.conditions)):
                continue
            if len(conditions) > self.max_condition_size:
                continue
            objects = self._objects_matching(conditions)
            if len(objects) < 2:
                continue
            shared_effects = self._shared_effects(objects)
            parent_effects = left.expected_effects | right.expected_effects
            novel_effects = shared_effects - parent_effects
            if not novel_effects:
                continue
            child = self._create_abstraction(
                parents=(left.abstraction_id, right.abstraction_id),
                conditions=conditions,
                expected_effects=shared_effects,
                object_ids={obj.object_id for obj in objects},
                created_at=observation.step,
                depth=max(left.depth, right.depth) + 1,
            )
            if child is not None:
                left.children.add(child.abstraction_id)
                right.children.add(child.abstraction_id)
                left.composition_count += 1
                right.composition_count += 1
                self.compositions += 1
            return

    def create_new_test(self) -> None:
        # Tests are represented by feature atoms in this first MVP.
        return None

    def metrics(self, step: int) -> StepMetrics:
        abstractions = list(self.abstractions.values())
        alive = [item for item in abstractions if item.alive]
        return StepMetrics(
            step=step,
            observations=len(self.observations),
            abstractions=len(abstractions),
            alive_abstractions=len(alive),
            max_depth=max((item.depth for item in abstractions), default=0),
            mean_lifetime=_mean([item.lifetime for item in abstractions]),
            mean_reuse=_mean([item.use_count for item in abstractions]),
            merges=self.merges,
            splits=self.splits,
            destroyed=self.destroyed,
            created=self.created,
            compositions=self.compositions,
        )

    def transfer_reuse(self, objects: list[ToyObject]) -> int:
        total = 0
        for toy_object in objects:
            for abstraction in self.abstractions.values():
                if abstraction.alive and abstraction.conditions <= toy_object.atoms:
                    total += 1
        return total

    def summary(self, transfer_objects: list[ToyObject]) -> ExperimentSummary:
        abstractions = list(self.abstractions.values())
        alive = [item for item in abstractions if item.alive]
        return ExperimentSummary(
            agent_name=self.name,
            observations=len(self.observations),
            abstractions=len(abstractions),
            alive_abstractions=len(alive),
            max_depth=max((item.depth for item in abstractions), default=0),
            mean_lifetime=_mean([item.lifetime for item in abstractions]),
            mean_reuse=_mean([item.use_count for item in abstractions]),
            total_life_reuse=sum(item.life_reuse for item in abstractions),
            transfer_reuse=self.transfer_reuse(transfer_objects),
            destroyed=self.destroyed,
            merges=self.merges,
            splits=self.splits,
            compositions=self.compositions,
        )

    def _create_abstraction(
        self,
        *,
        parents: tuple[str, ...],
        conditions: frozenset[Atom],
        expected_effects: frozenset[str],
        object_ids: set[str],
        created_at: int,
        depth: int,
    ) -> Abstraction | None:
        for abstraction in self.abstractions.values():
            if abstraction.alive and abstraction.conditions == conditions:
                return None

        abstraction = Abstraction(
            abstraction_id=f"{self.name}_a{self._next_id}",
            parents=parents,
            conditions=conditions,
            expected_effects=expected_effects,
            object_ids=set(object_ids),
            created_at=created_at,
            depth=depth,
        )
        self._next_id += 1
        self.abstractions[abstraction.abstraction_id] = abstraction
        self.created += 1
        if parents:
            self.compositions += 0
        else:
            self.merges += 1
        return abstraction

    def _objects_matching(self, conditions: frozenset[Atom]) -> list[ToyObject]:
        return [
            observation.toy_object
            for observation in self.observations
            if conditions <= observation.toy_object.atoms
        ]

    @staticmethod
    def _shared_effects(objects: list[ToyObject]) -> frozenset[str]:
        if not objects:
            return frozenset()
        shared = set(objects[0].effects)
        for toy_object in objects[1:]:
            shared &= toy_object.effects
        return frozenset(shared)


class LifetimeReuseAgent(AbstractionAgent):
    def __init__(self) -> None:
        super().__init__("lifetime_reuse", max_condition_size=3)


class ErrorMinimizingBaseline(AbstractionAgent):
    def __init__(self) -> None:
        super().__init__("error_baseline", max_condition_size=5)

    def propose_merge(self, observation: ObjectObservation) -> None:
        # Baseline memorizes exact observed configurations. This avoids many
        # counterexamples but creates shallow, low-reuse abstractions.
        conditions = frozenset(observation.toy_object.atoms)
        self._create_abstraction(
            parents=(),
            conditions=conditions,
            expected_effects=observation.toy_object.effects,
            object_ids={observation.toy_object.object_id},
            created_at=observation.step,
            depth=1,
        )

    def compose_abstractions(self, observation: ObjectObservation) -> None:
        return None

    def propose_split(self, failed: Abstraction, observation: ObjectObservation) -> None:
        return None


def _mean(values: list[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
