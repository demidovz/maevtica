from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable


State = str
Result = str
Block = frozenset[State]
Partition = tuple[Block, ...]


STATE_ORDER: tuple[State, ...] = ("A0", "A1", "B0", "B1")


@dataclass(frozen=True)
class World:
    states: tuple[State, ...]
    transition: dict[State, State]


@dataclass(frozen=True)
class Test:
    name: str
    values: dict[State, Result]

    def __call__(self, state: State) -> Result:
        return self.values[state]


@dataclass(frozen=True)
class TestFamily:
    name: str
    tests: tuple[Test, ...]


@dataclass(frozen=True)
class Quotient:
    name: str
    blocks: Partition


def sort_key(state: State) -> int:
    return STATE_ORDER.index(state)


def normalize_partition(blocks: Iterable[Iterable[State]]) -> Partition:
    normalized = [frozenset(block) for block in blocks if block]
    return tuple(
        sorted(
            normalized,
            key=lambda block: tuple(sort_key(state) for state in sorted(block, key=sort_key)),
        )
    )


def format_block(block: Block) -> str:
    return "{" + ",".join(sorted(block, key=sort_key)) + "}"


def format_partition(partition: Partition) -> str:
    return "{" + ", ".join(format_block(block) for block in partition) + "}"


def signature(test_family: TestFamily, state: State) -> tuple[Result, ...]:
    return tuple(test(state) for test in test_family.tests)


def induced_quotient(test_family: TestFamily, states: tuple[State, ...]) -> Quotient:
    groups: dict[tuple[Result, ...], list[State]] = {}
    for state in states:
        groups.setdefault(signature(test_family, state), []).append(state)
    return Quotient(test_family.name, normalize_partition(groups.values()))


def state_to_block(partition: Partition) -> dict[State, Block]:
    mapping: dict[State, Block] = {}
    for block in partition:
        for state in block:
            mapping[state] = block
    return mapping


def is_sufficient(quotient: Quotient, test_family: TestFamily) -> bool:
    for block in quotient.blocks:
        for test in test_family.tests:
            values = {test(state) for state in block}
            if len(values) != 1:
                return False
    return True


def is_valid(quotient: Quotient, world: World) -> bool:
    mapping = state_to_block(quotient.blocks)
    for block in quotient.blocks:
        target_blocks = {mapping[world.transition[state]] for state in block}
        if len(target_blocks) != 1:
            return False
    return True


def quotient_transition(quotient: Quotient, world: World) -> dict[Block, Block]:
    if not is_valid(quotient, world):
        raise ValueError(f"quotient {quotient.name} has no well-defined transition")
    mapping = state_to_block(quotient.blocks)
    return {
        block: mapping[world.transition[next(iter(block))]]
        for block in quotient.blocks
    }


def all_partitions(states: tuple[State, ...]) -> list[Partition]:
    if not states:
        return [()]

    first, *rest = states
    rest_partitions = all_partitions(tuple(rest))
    partitions: list[Partition] = []

    for partition in rest_partitions:
        partitions.append(normalize_partition((frozenset((first,)), *partition)))
        for index, block in enumerate(partition):
            replaced = list(partition)
            replaced[index] = frozenset((first, *block))
            partitions.append(normalize_partition(replaced))

    unique = {partition: partition for partition in partitions}
    return sorted(unique.values(), key=lambda p: (len(p), format_partition(p)))


def is_coarser_or_equal(candidate: Quotient, reference: Quotient) -> bool:
    candidate_map = state_to_block(candidate.blocks)
    for block in reference.blocks:
        touched_candidate_blocks = {candidate_map[state] for state in block}
        if len(touched_candidate_blocks) > 1:
            return False
    return True


def is_strictly_coarser(candidate: Quotient, reference: Quotient) -> bool:
    return (
        is_coarser_or_equal(candidate, reference)
        and candidate.blocks != reference.blocks
    )


def is_minimally_sufficient(
    quotient: Quotient,
    test_family: TestFamily,
    partitions: tuple[Partition, ...],
) -> bool:
    if not is_sufficient(quotient, test_family):
        return False
    for partition in partitions:
        candidate = Quotient("candidate", partition)
        if is_strictly_coarser(candidate, quotient) and is_sufficient(candidate, test_family):
            return False
    return True


def split_blocks(before: Quotient, after: Quotient) -> dict[Block, tuple[Block, ...]]:
    splits: dict[Block, tuple[Block, ...]] = {}
    for block in before.blocks:
        pieces = tuple(after_block for after_block in after.blocks if after_block <= block)
        if len(pieces) > 1:
            splits[block] = pieces
    return splits


def merge_blocks(before: Quotient, after: Quotient) -> dict[Block, tuple[Block, ...]]:
    merges: dict[Block, tuple[Block, ...]] = {}
    for block in after.blocks:
        pieces = tuple(before_block for before_block in before.blocks if before_block <= block)
        if len(pieces) > 1:
            merges[block] = pieces
    return merges


def transition_preserving_coarsenings(
    quotient: Quotient,
    world: World,
    partitions: tuple[Partition, ...],
) -> list[Quotient]:
    coarsenings: list[Quotient] = []
    for partition in partitions:
        candidate = Quotient("coarsening", partition)
        if is_strictly_coarser(candidate, quotient) and is_valid(candidate, world):
            coarsenings.append(candidate)
    return sorted(coarsenings, key=lambda q: (len(q.blocks), format_partition(q.blocks)))


def unchanged_under_added_tests(before: Quotient, after: Quotient) -> bool:
    return before.blocks == after.blocks


def build_world() -> World:
    return World(
        states=STATE_ORDER,
        transition={
            "A0": "A1",
            "A1": "A0",
            "B0": "B1",
            "B1": "B0",
        },
    )


def build_tests() -> dict[str, Test]:
    return {
        "obs": Test(
            "obs",
            {
                "A0": "0",
                "A1": "1",
                "B0": "0",
                "B1": "1",
            },
        ),
        "mode": Test(
            "mode",
            {
                "A0": "A",
                "A1": "A",
                "B0": "B",
                "B1": "B",
            },
        ),
        "id": Test(
            "id",
            {
                "A0": "A0",
                "A1": "A1",
                "B0": "B0",
                "B1": "B1",
            },
        ),
        "parity": Test(
            "parity",
            {
                "A0": "red",
                "A1": "blue",
                "B0": "blue",
                "B1": "red",
            },
        ),
    }


def print_quotient_report(
    world: World,
    test_family: TestFamily,
    quotient: Quotient,
    partitions: tuple[Partition, ...],
) -> None:
    print(f"{test_family.name}:")
    print(f"  tests: {[test.name for test in test_family.tests]}")
    print(f"  quotient: {format_partition(quotient.blocks)}")
    print(f"  complexity: {len(quotient.blocks)}")
    print(f"  sufficient: {is_sufficient(quotient, test_family)}")
    print(f"  minimally_sufficient: {is_minimally_sufficient(quotient, test_family, partitions)}")
    print(f"  valid_dynamics: {is_valid(quotient, world)}")
    if is_valid(quotient, world):
        transitions = quotient_transition(quotient, world)
        formatted = [
            f"{format_block(source)}->{format_block(target)}"
            for source, target in transitions.items()
        ]
        print(f"  quotient_transition: {formatted}")


def print_split_report(before: Quotient, after: Quotient) -> None:
    splits = split_blocks(before, after)
    if not splits:
        print(f"  {before.name} -> {after.name}: no split")
        return
    print(f"  {before.name} -> {after.name}: split")
    for block, pieces in splits.items():
        print(
            "    "
            + format_block(block)
            + " -> "
            + ", ".join(format_block(piece) for piece in pieces)
        )


def print_merge_report(before: Quotient, after: Quotient) -> None:
    merges = merge_blocks(before, after)
    if not merges:
        print(f"  {before.name} -> {after.name}: no merge")
        return
    print(f"  {before.name} -> {after.name}: merge")
    for block, pieces in merges.items():
        print(
            "    "
            + ", ".join(format_block(piece) for piece in pieces)
            + " -> "
            + format_block(block)
        )


def assert_protocol_invariants(
    world: World,
    families: tuple[TestFamily, ...],
    quotients: tuple[Quotient, ...],
    partitions: tuple[Partition, ...],
) -> None:
    assert len(partitions) == 15
    for family, quotient in zip(families, quotients):
        assert is_sufficient(quotient, family)
        assert is_minimally_sufficient(quotient, family, partitions)
        assert is_valid(quotient, world)

    assert len(quotients[0].blocks) == 2
    assert len(quotients[1].blocks) == 4
    assert unchanged_under_added_tests(quotients[1], quotients[2])


def run_base_sequence(
    world: World,
    tests: dict[str, Test],
    partitions: tuple[Partition, ...],
) -> tuple[Quotient, Quotient, Quotient]:
    families = (
        TestFamily("T0_obs", (tests["obs"],)),
        TestFamily("T1_obs_mode", (tests["obs"], tests["mode"])),
        TestFamily("T2_obs_mode_id", (tests["obs"], tests["mode"], tests["id"])),
    )
    quotients = tuple(induced_quotient(family, world.states) for family in families)

    print("BASE SEQUENCE")
    for family, quotient in zip(families, quotients):
        print_quotient_report(world, family, quotient, partitions)

    print("  refinement:")
    print_split_report(quotients[0], quotients[1])
    print_split_report(quotients[1], quotients[2])
    print("  weakening:")
    print_merge_report(quotients[1], quotients[0])

    assert_protocol_invariants(world, families, quotients, partitions)
    return quotients


def run_artifact_counterexample(
    world: World,
    tests: dict[str, Test],
    partitions: tuple[Partition, ...],
) -> None:
    families = (
        TestFamily("T0_obs", (tests["obs"],)),
        TestFamily("T1_obs_parity", (tests["obs"], tests["parity"])),
    )
    q_obs, q_parity = tuple(induced_quotient(family, world.states) for family in families)

    print()
    print("ARTIFACT COUNTEREXAMPLE")
    for family, quotient in zip(families, (q_obs, q_parity)):
        print_quotient_report(world, family, quotient, partitions)

    print("  refinement:")
    print_split_report(q_obs, q_parity)
    print("  weakening:")
    print_merge_report(q_parity, q_obs)

    coarsenings = transition_preserving_coarsenings(q_parity, world, partitions)
    print("  valid_coarser_quotients:")
    for coarsening in coarsenings:
        print(f"    {format_partition(coarsening.blocks)}")

    print("  verdict:")
    print(
        "    stable_minimal_quotient_exists=True, "
        "but it is test-injected because it disappears when parity is removed."
    )

    assert is_minimally_sufficient(q_parity, families[1], partitions)
    assert is_valid(q_parity, world)
    assert q_parity.blocks != q_obs.blocks
    assert merge_blocks(q_parity, q_obs)


def run_exhaustive_partition_check(world: World, partitions: tuple[Partition, ...]) -> None:
    valid_count = sum(1 for partition in partitions if is_valid(Quotient("q", partition), world))
    print()
    print("EXHAUSTIVE CHECK")
    print(f"  all_partitions: {len(partitions)}")
    print(f"  valid_partitions_under_delta: {valid_count}")
    for size in range(1, len(world.states) + 1):
        candidates = [
            partition
            for partition in partitions
            if len(partition) == size and is_valid(Quotient("q", partition), world)
        ]
        if candidates:
            print(f"  valid_size_{size}: {[format_partition(p) for p in candidates]}")


def main() -> None:
    world = build_world()
    tests = build_tests()
    partitions = tuple(all_partitions(world.states))

    run_base_sequence(world, tests, partitions)
    run_artifact_counterexample(world, tests, partitions)
    run_exhaustive_partition_check(world, partitions)

    print()
    print("SUMMARY")
    print("  supported: tests induce distinctions, equivalences, quotients, splits, and merges.")
    print("  falsified: stable minimally sufficient quotient alone is not enough for concepthood.")
    print("  next_requirement: concept candidates need a non-arbitrariness or invariant criterion.")


if __name__ == "__main__":
    main()
