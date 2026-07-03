from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .metrics import compute_metrics
from .models import Artifact, ArtifactKind, CycleRecord, ResearchState


MILESTONES = {10, 50, 100, 500, 1000}


def journal_dir(root: str | Path) -> Path:
    return Path(root) / "research_journal"


def cycle_journal_path(root: str | Path, cycle_number: int) -> Path:
    return journal_dir(root) / f"cycle_{cycle_number:04d}.md"


def write_cycle_journal(root: str | Path, state: ResearchState, cycle: CycleRecord) -> Path:
    path = cycle_journal_path(root, cycle.cycle)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_cycle_journal(state, cycle), encoding="utf-8")
    write_journal_index(root, state)
    if cycle.cycle in MILESTONES:
        write_milestone_report(root, state, cycle)
    return path


def write_missing_journals(root: str | Path, state: ResearchState) -> list[Path]:
    written = []
    for cycle in state.cycles:
        path = cycle_journal_path(root, cycle.cycle)
        if not path.exists():
            written.append(write_cycle_journal(root, state, cycle))
    write_journal_index(root, state)
    return written


def render_cycle_journal(state: ResearchState, cycle: CycleRecord) -> str:
    artifacts = [state.artifacts[item] for item in cycle.artifact_ids if item in state.artifacts]
    edges = [state.edges[item] for item in cycle.edge_ids if item in state.edges]
    allocations = [state.allocations[item] for item in cycle.allocation_ids if item in state.allocations]
    bids = [state.bids[item] for item in cycle.bid_ids if item in state.bids]
    frontier = state.artifacts.get(cycle.frontier_id or "")
    next_frontier = state.artifacts.get(cycle.next_frontier_id or "")
    metrics = cycle.metrics
    role_artifacts = group_by_role(artifacts)
    by_domain = Counter()
    for allocation in allocations:
        by_domain[allocation.domain] += allocation.allocated
    active_agents = [
        f"{agent.role}:{agent.id}"
        for agent in sorted(state.agents.values(), key=lambda item: item.role)
        if agent.status == "active"
    ]
    active_domains = [
        f"{domain.name} ({domain.status}, priority {domain.priority})"
        for domain in sorted(state.domains.values(), key=lambda item: item.name)
        if domain.status == "active"
    ]
    alternatives = rejected_frontiers(state, frontier.id if frontier else None)
    explorer = role_artifacts.get("explorer", [])
    lab = role_artifacts.get("lab", [])
    engine = role_artifacts.get("engine", [])
    historian = role_artifacts.get("historian", [])
    cartographer = role_artifacts.get("cartographer", [])
    meta = role_artifacts.get("meta_observer", [])
    best_bid = max(bids, key=lambda item: item.expected_value, default=None)
    most_promising = first_kind(explorer, ArtifactKind.HYPOTHESIS)
    biggest_mistake = lowest_efficiency_allocation(allocations, artifacts)
    domain_balance = ", ".join(f"{name}: {amount:.2f}" for name, amount in sorted(by_domain.items())) or "none"
    return "\n".join(
        [
            "# Header",
            "",
            f"Cycle Number: {cycle.cycle}",
            f"Timestamp: {cycle.completed_at}",
            f"Attention Budget: {cycle.attention_budget}",
            f"Active Agents: {', '.join(active_agents) or 'none'}",
            f"Active Domains: {', '.join(active_domains) or 'none'}",
            f"Current Research Frontier: {frontier.title if frontier else 'none'}",
            "",
            "# Planner Decision",
            "",
            f"The frontier was selected because it had the strongest active priority signal: {frontier.title if frontier else 'no explicit frontier'}.",
            f"Alternative frontiers rejected: {alternatives or 'none available'}.",
            f"Expected epistemic value: {best_bid.expected_value if best_bid else metrics.get('epistemic_value', 0)}.",
            "",
            "# Attention Allocation",
            "",
            f"Attention available: {cycle.attention_budget}.",
            f"Attention spent: {cycle.attention_spent}.",
            f"Unused attention: {round(cycle.attention_budget - cycle.attention_spent, 3)}.",
            "Allocation per agent:",
            bullet_lines(
                [
                    f"{allocation.role}:{allocation.agent_id} received {allocation.allocated} for {allocation.bid_type if hasattr(allocation, 'bid_type') else 'work'}"
                    for allocation in allocations
                ]
            ),
            f"Allocation per domain: {domain_balance}.",
            f"Allocation efficiency: {metrics.get('attention_efficiency', 0.0)}.",
            "",
            "# Explorer Report",
            "",
            f"New hypotheses generated: {titles_of_kind(explorer, ArtifactKind.HYPOTHESIS)}.",
            f"Duplicate hypotheses detected: {metrics.get('duplicate_rate', 0.0)} estimated duplicate rate.",
            f"Interesting unknown regions: {titles_of_kind(artifacts, ArtifactKind.UNKNOWN_REGION)}.",
            f"Unexpected observations: {titles_of_kind(artifacts, ArtifactKind.OBSERVATION)}.",
            f"Most promising hypothesis: {most_promising.title if most_promising else 'none'}.",
            "",
            "# Lab Report",
            "",
            f"Formalizations created: {titles_of_kind(lab, ArtifactKind.DERIVED_CONCEPT)}.",
            "Definitions refined: none recorded as separate artifacts this cycle.",
            "Theorems proposed: none recorded as separate artifacts this cycle.",
            "Concept merges: none recorded as separate artifacts this cycle.",
            f"New abstractions: {titles_of_kind(lab, ArtifactKind.DERIVED_CONCEPT)}.",
            "",
            "# Engine Report",
            "",
            f"Hypotheses falsified: {count_status(state, ArtifactKind.HYPOTHESIS, 'falsified')}.",
            f"Counterexamples found: {titles_of_kind(engine, ArtifactKind.COUNTEREXAMPLE)}.",
            f"Weak assumptions exposed: {titles_of_kind(engine, ArtifactKind.CONTRADICTION)}.",
            f"Experiments proposed: {titles_of_kind(lab, ArtifactKind.EXPERIMENT)}.",
            "Confidence reductions: hypothesis confidence was reduced when Engine produced contradiction pressure.",
            "",
            "# Historian Report",
            "",
            f"Research State changes: {len(cycle.artifact_ids)} artifact writes and {len(cycle.edge_ids)} graph edge writes.",
            "Merged concepts: none recorded this cycle.",
            "Removed redundancy: redundancy was measured; immutable artifacts were not deleted.",
            f"New links created: {edge_summary(edges)}.",
            f"Applicability updates: {titles_of_kind(cartographer, ArtifactKind.APPLICABILITY_ATLAS)}.",
            f"Memory growth: {len(state.artifacts)} total artifacts, {len(state.edges)} total edges after this cycle.",
            "",
            "# Cartographer Report",
            "",
            f"Knowledge graph changes: {len(edges)} new edges in this cycle.",
            f"Compression achieved: {metrics.get('compression', 0.0)}.",
            f"Simplification achieved: {metrics.get('compression', 0.0)} compression ratio.",
            "New clusters: approximated by new derived concepts linked to hypotheses.",
            f"Disconnected regions: {count_kind(state, ArtifactKind.UNKNOWN_REGION)} unknown regions remain visible.",
            f"Remaining unknown regions: {titles_of_kind(state.active_artifacts_by_kind(ArtifactKind.UNKNOWN_REGION), ArtifactKind.UNKNOWN_REGION)}.",
            "",
            "# Meta Observer",
            "",
            f"How did today's research differ from previous cycles? {cycle_delta_sentence(state, cycle)}",
            f"Did research quality improve? Compression is {metrics.get('compression', 0.0)} and attention efficiency is {metrics.get('attention_efficiency', 0.0)}.",
            f"Did organizational behavior improve? Agent diversity is {metrics.get('organizational_diversity', 0.0)}.",
            f"Did the Studio become more efficient? Current attention efficiency is {metrics.get('attention_efficiency', 0.0)}.",
            f"Any early signs of stagnation? Stagnation score is {metrics.get('stagnation_score', 0.0)}.",
            "",
            "# Organizational Learning",
            "",
            f"What research strategy worked best? {best_strategy(state)}.",
            f"What strategy produced mostly noise? {noisiest_strategy(state)}.",
            "Should attention policy change? Increase weight for agents with compression contribution and reduce duplicate-heavy paths.",
            f"Should agent population evolve? {'yes' if cycle.organization_action_ids else 'not this cycle'}.",
            "",
            "# Metrics",
            "",
            metric_lines(state, cycle),
            "",
            "# Biggest Insight",
            "",
            biggest_insight(state, cycle, most_promising),
            "",
            "# Biggest Mistake",
            "",
            biggest_mistake,
            "",
            "# Next Experiment",
            "",
            f"The Planner selected {next_frontier.title if next_frontier else 'no next frontier'} because it currently offers the best combination of contradiction pressure, priority, and expected compression.",
            f"Expected epistemic value: {metrics.get('epistemic_value', 0.0)}.",
            f"Expected uncertainty reduction: {uncertainty_reduction_estimate(metrics)}.",
            f"Alternative options rejected: {alternatives or 'none available'}.",
            "",
            "# Human Summary",
            "",
            "Today's Studio Summary",
            "",
            human_summary(state, cycle, frontier, next_frontier),
            "",
        ]
    )


def write_milestone_report(root: str | Path, state: ResearchState, cycle: CycleRecord) -> Path:
    out_dir = journal_dir(root) / "milestones"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"milestone_{cycle.cycle:04d}.md"
    if path.exists():
        return path
    previous = previous_milestone(state, cycle.cycle)
    current = cycle.metrics
    old = previous.metrics if previous else {}
    lines = [
        f"# Milestone {cycle.cycle}",
        "",
        f"Compared with: cycle {previous.cycle if previous else 'initial state'}",
        "",
        f"- knowledge compression: {old.get('compression', 0.0)} -> {current.get('compression', 0.0)}",
        f"- research efficiency: {old.get('attention_efficiency', 0.0)} -> {current.get('attention_efficiency', 0.0)}",
        f"- organizational evolution: {len(state.organization_actions)} organization actions recorded",
        f"- agent specialization: {specialization_summary(state)}",
        f"- domain evolution: {domain_summary(state)}",
        f"- attention economy: spent {cycle.attention_spent} of {cycle.attention_budget} in the milestone cycle",
        f"- research velocity: {old.get('research_velocity', {})} -> {current.get('research_velocity', {})}",
        "- major paradigm shifts: none recorded as explicit health-report events",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_journal_index(root: str | Path, state: ResearchState) -> Path:
    base = journal_dir(root)
    base.mkdir(parents=True, exist_ok=True)
    entries = []
    for cycle in state.cycles:
        entries.append(
            {
                "cycle": cycle.cycle,
                "path": cycle_journal_path(root, cycle.cycle).name,
                "timestamp": cycle.completed_at,
                "metrics": cycle.metrics,
                "frontier_id": cycle.frontier_id,
                "next_frontier_id": cycle.next_frontier_id,
                "artifact_ids": cycle.artifact_ids,
                "edge_ids": cycle.edge_ids,
                "allocation_ids": cycle.allocation_ids,
                "bid_ids": cycle.bid_ids,
            }
        )
    path = base / "index.json"
    path.write_text(json.dumps({"entries": entries}, indent=2, sort_keys=True), encoding="utf-8")
    return path


def group_by_role(artifacts: list[Artifact]) -> dict[str, list[Artifact]]:
    grouped: dict[str, list[Artifact]] = {}
    for artifact in artifacts:
        grouped.setdefault(artifact.provenance.agent_role, []).append(artifact)
    return grouped


def first_kind(artifacts: list[Artifact], kind: str) -> Artifact | None:
    return next((artifact for artifact in artifacts if artifact.kind == kind), None)


def titles_of_kind(artifacts: list[Artifact], kind: str) -> str:
    titles = [artifact.title for artifact in artifacts if artifact.kind == kind]
    return "; ".join(titles) if titles else "none"


def count_kind(state: ResearchState, kind: str) -> int:
    return len(state.artifacts_by_kind(kind))


def count_status(state: ResearchState, kind: str, status: str) -> int:
    return len([artifact for artifact in state.artifacts_by_kind(kind) if artifact.status == status])


def bullet_lines(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "- none"


def edge_summary(edges: list) -> str:
    if not edges:
        return "none"
    counts = Counter(edge.kind for edge in edges)
    return ", ".join(f"{kind}: {count}" for kind, count in sorted(counts.items()))


def rejected_frontiers(state: ResearchState, selected_id: str | None) -> str:
    candidates = [
        artifact.title
        for artifact in state.artifacts.values()
        if artifact.kind in {ArtifactKind.UNKNOWN_REGION, ArtifactKind.OPEN_QUESTION, ArtifactKind.CONTRADICTION}
        and artifact.status == "active"
        and artifact.id != selected_id
    ]
    return "; ".join(candidates[:5])


def cycle_delta_sentence(state: ResearchState, cycle: CycleRecord) -> str:
    previous = next((item for item in state.cycles if item.cycle == cycle.cycle - 1), None)
    if not previous:
        return "This was the first persisted cycle with no prior baseline."
    old = previous.metrics.get("epistemic_value", 0.0)
    new = cycle.metrics.get("epistemic_value", 0.0)
    return f"Epistemic value moved from {old} to {new}; attention allocation is now explicit."


def best_strategy(state: ResearchState) -> str:
    agents = sorted(state.agents.values(), key=lambda item: item.reputation, reverse=True)
    if not agents:
        return "none"
    agent = agents[0]
    return f"{agent.role} / {agent.strategy} with reputation {agent.reputation}"


def noisiest_strategy(state: ResearchState) -> str:
    agents = sorted(state.agents.values(), key=lambda item: item.reputation)
    if not agents:
        return "none"
    agent = agents[0]
    return f"{agent.role} / {agent.strategy} with reputation {agent.reputation}"


def metric_lines(state: ResearchState, cycle: CycleRecord) -> str:
    metrics = cycle.metrics
    hypotheses = state.artifacts_by_kind(ArtifactKind.HYPOTHESIS)
    alive = [artifact for artifact in hypotheses if artifact.status == "active"]
    killed = [artifact for artifact in hypotheses if artifact.status in {"dead", "falsified"}]
    graph_size = len(state.artifacts) + len(state.edges)
    density = len(state.edges) / max(1, len(state.artifacts) * max(1, len(state.artifacts) - 1))
    concept_reuse = sum(item["reuse"] for item in metrics.get("most_reusable_concepts", []))
    domain_balance = ", ".join(
        f"{item['name']}:{item['priority']}" for item in metrics.get("active_domains", [])
    )
    lines = [
        f"- Total hypotheses: {len(hypotheses)}",
        f"- Alive hypotheses: {len(alive)}",
        f"- Killed hypotheses: {len(killed)}",
        "- Merged hypotheses: 0",
        f"- Counterexamples: {count_kind(state, ArtifactKind.COUNTEREXAMPLE)}",
        f"- Open questions: {count_kind(state, ArtifactKind.OPEN_QUESTION)}",
        f"- Research graph size: {graph_size}",
        f"- Graph density: {round(density, 4)}",
        f"- Compression ratio: {metrics.get('compression', 0.0)}",
        f"- Attention efficiency: {metrics.get('attention_efficiency', 0.0)}",
        f"- Research velocity: {metrics.get('research_velocity', {})}",
        f"- Stagnation score: {metrics.get('stagnation_score', 0.0)}",
        f"- Concept reuse: {concept_reuse}",
        f"- Domain balance: {domain_balance or 'none'}",
        f"- Agent diversity: {metrics.get('organizational_diversity', 0.0)}",
    ]
    return "\n".join(lines)


def biggest_insight(
    state: ResearchState, cycle: CycleRecord, most_promising: Artifact | None
) -> str:
    if most_promising:
        return (
            f"The lasting contribution was making {most_promising.title} explicit enough for Lab formalization "
            "and Engine pressure, so it can now be improved or killed by artifacts rather than discussion."
        )
    return "The lasting contribution was preserving a complete cycle trace even when no strong new hypothesis appeared."


def lowest_efficiency_allocation(allocations: list, artifacts: list[Artifact]) -> str:
    if not allocations:
        return "No allocation consumed attention in this cycle."
    by_agent = Counter(artifact.provenance.agent_id for artifact in artifacts)
    weakest = min(allocations, key=lambda item: by_agent[item.agent_id] / max(1.0, item.allocated))
    return (
        f"The weakest attention use was {weakest.role}:{weakest.agent_id}, which spent {weakest.allocated} "
        f"attention for {by_agent[weakest.agent_id]} visible artifact writes."
    )


def uncertainty_reduction_estimate(metrics: dict[str, Any]) -> float:
    value = metrics.get("counterexample_density", 0.0) + metrics.get("compression", 0.0)
    return round(min(1.0, value / 2), 3)


def human_summary(
    state: ResearchState,
    cycle: CycleRecord,
    frontier: Artifact | None,
    next_frontier: Artifact | None,
) -> str:
    text = (
        f"Cycle {cycle.cycle} investigated {frontier.title if frontier else 'the available frontier'}. "
        f"The Studio spent {cycle.attention_spent} of {cycle.attention_budget} attention, wrote "
        f"{len(cycle.artifact_ids)} artifact updates, and added {len(cycle.edge_ids)} graph links. "
        f"Compression is now {cycle.metrics.get('compression', 0.0)}, attention efficiency is "
        f"{cycle.metrics.get('attention_efficiency', 0.0)}, and stagnation score is "
        f"{cycle.metrics.get('stagnation_score', 0.0)}. The next direction is "
        f"{next_frontier.title if next_frontier else 'not yet selected'} because it remains the strongest "
        "visible opportunity to reduce uncertainty or expose a weak explanation."
    )
    words = text.split()
    return " ".join(words[:250])


def previous_milestone(state: ResearchState, cycle_number: int) -> CycleRecord | None:
    candidates = [
        cycle for cycle in state.cycles if cycle.cycle < cycle_number and cycle.cycle in MILESTONES
    ]
    return candidates[-1] if candidates else None


def specialization_summary(state: ResearchState) -> str:
    roles = Counter(agent.role for agent in state.agents.values() if agent.status == "active")
    return ", ".join(f"{role}: {count}" for role, count in sorted(roles.items())) or "none"


def domain_summary(state: ResearchState) -> str:
    return ", ".join(
        f"{domain.name}: {domain.status}/{domain.priority}"
        for domain in sorted(state.domains.values(), key=lambda item: item.name)
    ) or "none"
