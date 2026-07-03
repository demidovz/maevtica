from __future__ import annotations

from .agents import DEFAULT_AGENTS, ResearchAgent
from .journal import write_cycle_journal
from .metrics import choose_frontier, compute_metrics
from .models import AgentRecord, Artifact, ArtifactKind, CycleRecord, Provenance, ResearchState, now_iso
from .organization import (
    ATTENTION_BUDGET,
    active_agent_records,
    agent_from_record,
    allocate_attention,
    build_meta_memory,
    ensure_domains,
    evolve_agents,
    make_market_bids,
    update_agent_reputation,
    update_domains,
)
from .storage import ResearchStore


def initialize_store(store: ResearchStore, domain_seed: tuple[list, list] | None = None) -> ResearchState:
    store.ensure()
    state = store.load()
    if state.agents:
        return state
    for agent in DEFAULT_AGENTS:
        store.save_agent(agent.record())
    state = store.load()
    for domain in ensure_domains(state):
        store.save_domain(domain)
    if domain_seed:
        artifacts, edges = domain_seed
        for artifact in artifacts:
            store.add_artifact(artifact)
        for edge in edges:
            store.add_edge(edge)
    state = store.load()
    store.write_snapshot(state)
    return state


def run_cycle(
    store: ResearchStore,
    agents: list[ResearchAgent] | None = None,
    attention_budget: float = ATTENTION_BUDGET,
) -> CycleRecord:
    state = store.load()
    cycle_number = len(state.cycles) + 1
    started_at = now_iso()
    frontier_id = choose_frontier(state)
    created_artifacts: list[str] = []
    created_edges: list[str] = []
    organization_action_ids: list[str] = []
    previous_metrics = compute_metrics(state)
    previous_compression = previous_metrics.get("compression", 0.0)

    if agents is None:
        agent_records = {record.id: record for record in active_agent_records(state)}
        bids = make_market_bids(state, cycle_number, frontier_id)
        for bid in bids:
            store.add_bid(bid)
            state.bids[bid.id] = bid
        allocations = allocate_attention(state, bids, attention_budget)
        for allocation in allocations:
            store.add_allocation(allocation)
            state.allocations[allocation.id] = allocation
        run_plan = [(agent_from_record(agent_records[allocation.agent_id]), allocation) for allocation in allocations]
    else:
        agent_records = {
            agent.id: state.agents.get(agent.id) or AgentRecord(id=agent.id, role=agent.role)
            for agent in agents
        }
        bids = []
        allocations = []
        share = attention_budget / max(1, len(agents))
        run_plan = [
            (
                agent,
                _manual_allocation(agent.id, agent.role, frontier_id, cycle_number, share),
            )
            for agent in agents
        ]

    for agent, allocation in run_plan:
        output = agent.run(state, frontier_id, cycle_number, attention=allocation.allocated)
        for artifact in output.artifacts:
            store.add_artifact(artifact)
            state.artifacts[artifact.id] = artifact
            created_artifacts.append(artifact.id)
        for edge in output.edges:
            store.add_edge(edge)
            state.edges[edge.id] = edge
            created_edges.append(edge.id)
        for artifact in output.updates:
            store.update_artifact(artifact)
            state.artifacts[artifact.id] = artifact
            created_artifacts.append(artifact.id)
        current_metrics = compute_metrics(state)
        compression_delta = current_metrics.get("compression", 0.0) - previous_compression
        record = update_agent_reputation(
            state,
            agent_records,
            allocation,
            {
                "artifacts": len(output.artifacts),
                "edges": len(output.edges),
                "updates": len(output.updates),
            },
            compression_delta,
        )
        store.save_agent(record)
        state.agents[record.id] = record
        previous_compression = current_metrics.get("compression", previous_compression)

    metrics = compute_metrics(state)
    for domain in update_domains(state, allocations, metrics, cycle_number):
        store.save_domain(domain)
        state.domains[domain.name] = domain

    meta_memory = build_meta_memory(state, cycle_number, metrics)
    if meta_memory:
        store.add_artifact(meta_memory)
        state.artifacts[meta_memory.id] = meta_memory
        created_artifacts.append(meta_memory.id)

    for record, action in zip(*evolve_agents(state, cycle_number), strict=False):
        store.save_agent(record)
        state.agents[record.id] = record
        store.add_organization_action(action)
        state.organization_actions[action.id] = action
        organization_action_ids.append(action.id)

    health_report_id = None
    if cycle_number % 100 == 0:
        health_report = build_health_report(state, cycle_number, metrics)
        store.add_artifact(health_report)
        state.artifacts[health_report.id] = health_report
        created_artifacts.append(health_report.id)
        health_report_id = health_report.id

    metrics = compute_metrics(state)
    next_frontier_id = choose_frontier(state)
    cycle = CycleRecord(
        cycle=cycle_number,
        started_at=started_at,
        completed_at=now_iso(),
        frontier_id=frontier_id,
        artifact_ids=created_artifacts,
        edge_ids=created_edges,
        metrics=metrics,
        next_frontier_id=next_frontier_id,
        attention_budget=attention_budget,
        attention_spent=round(sum(allocation.allocated for allocation in allocations), 3),
        bid_ids=[bid.id for bid in bids],
        allocation_ids=[allocation.id for allocation in allocations],
        organization_action_ids=organization_action_ids,
        health_report_id=health_report_id,
    )
    store.add_cycle(cycle)
    state.cycles.append(cycle)
    store.write_snapshot(state)
    write_cycle_journal(store.root, state, cycle)
    return cycle


def run_cycles(
    store: ResearchStore, cycles: int, attention_budget: float = ATTENTION_BUDGET
) -> list[CycleRecord]:
    return [run_cycle(store, attention_budget=attention_budget) for _ in range(cycles)]


def _manual_allocation(
    agent_id: str, role: str, frontier_id: str | None, cycle: int, share: float
):
    from .models import AttentionAllocation

    return AttentionAllocation(
        agent_id=agent_id,
        role=role,
        target_id=frontier_id,
        allocated=round(share, 3),
        requested=round(share, 3),
        expected_value=1.0,
        reputation=1.0,
        domain="manual",
        cycle=cycle,
    )


def build_health_report(state: ResearchState, cycle: int, metrics: dict) -> Artifact:
    active_agents = active_agent_records(state)
    diversity = len({agent.role for agent in active_agents}) / max(1, len(active_agents))
    body = (
        f"Knowledge compression: {metrics['compression']}. "
        f"Attention efficiency: {metrics.get('attention_efficiency', 0.0)}. "
        f"Duplicate rate: {metrics.get('duplicate_rate', 0.0)}. "
        f"Counterexample ratio: {metrics['counterexample_density']}. "
        f"Concept reuse: {len(metrics['most_reusable_concepts'])}. "
        f"Research velocity: {metrics['research_velocity']}. "
        f"Stagnation score: {metrics.get('stagnation_score', 0.0)}. "
        f"Organizational diversity: {round(diversity, 3)}."
    )
    return Artifact(
        kind=ArtifactKind.HEALTH_REPORT,
        title=f"Research Health Report cycle {cycle}",
        body=body,
        provenance=Provenance(
            agent_id="organization_health",
            agent_role="meta_observer",
            cycle=cycle,
            source="self_diagnostics",
        ),
        confidence=0.85,
        metadata={"metrics": metrics, "organizational_diversity": round(diversity, 3)},
    )
