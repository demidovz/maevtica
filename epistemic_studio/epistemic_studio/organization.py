from __future__ import annotations

from collections import Counter

from .agents import (
    CartographerAgent,
    EngineAgent,
    ExplorerAgent,
    HistorianAgent,
    LabAgent,
    MetaObserverAgent,
    PlannerAgent,
    ResearchAgent,
)
from .metrics import compute_metrics
from .models import (
    AgentRecord,
    Artifact,
    ArtifactKind,
    AttentionAllocation,
    DomainRecord,
    MarketBid,
    OrganizationAction,
    Provenance,
    ResearchState,
)


ATTENTION_BUDGET = 100.0

AGENT_CLASSES: dict[str, type[ResearchAgent]] = {
    "explorer": ExplorerAgent,
    "lab": LabAgent,
    "engine": EngineAgent,
    "historian": HistorianAgent,
    "cartographer": CartographerAgent,
    "planner": PlannerAgent,
    "meta_observer": MetaObserverAgent,
}


def agent_from_record(record: AgentRecord) -> ResearchAgent:
    cls = AGENT_CLASSES.get(record.role, ResearchAgent)
    return cls(record.id)


def active_agent_records(state: ResearchState) -> list[AgentRecord]:
    return [
        agent
        for agent in sorted(state.agents.values(), key=lambda item: item.id)
        if agent.status == "active" and agent.role in AGENT_CLASSES
    ]


def ensure_domains(state: ResearchState) -> list[DomainRecord]:
    domain_names = {"possible_worlds", "agi", "maevtica", "biology", "economics"}
    for artifact in state.artifacts.values():
        domain = artifact.metadata.get("domain")
        if domain:
            domain_names.add(domain)
    records = []
    for name in sorted(domain_names):
        records.append(state.domains.get(name) or DomainRecord(name=name))
    return records


def choose_domain(state: ResearchState) -> str:
    domains = ensure_domains(state)
    active = [domain for domain in domains if domain.status == "active"]
    if not active:
        active = domains
    return max(active, key=lambda item: (item.priority, item.cycles_since_active * 0.05)).name


def make_market_bids(state: ResearchState, cycle: int, frontier_id: str | None) -> list[MarketBid]:
    frontier = state.artifacts.get(frontier_id or "")
    target_id = frontier.id if frontier else None
    metrics = compute_metrics(state)
    bids: list[MarketBid] = []
    domain = choose_domain(state)
    base_by_role = {
        "explorer": 16.0,
        "lab": 18.0,
        "engine": 22.0,
        "historian": 8.0,
        "cartographer": 10.0,
        "planner": 12.0,
        "meta_observer": 9.0,
    }
    bid_type_by_role = {
        "explorer": "invest",
        "lab": "formalize",
        "engine": "short",
        "historian": "audit",
        "cartographer": "map",
        "planner": "allocate",
        "meta_observer": "diagnose",
    }
    for agent in active_agent_records(state):
        requested = base_by_role[agent.role]
        expected = expected_value(agent, metrics, frontier, domain)
        bids.append(
            MarketBid(
                agent_id=agent.id,
                role=agent.role,
                target_id=target_id,
                bid_type=bid_type_by_role[agent.role],
                requested_attention=requested,
                expected_value=expected,
                rationale=f"{agent.role} bid weighted by reputation, domain priority, and current bottlenecks.",
                cycle=cycle,
            )
        )
    return bids


def expected_value(
    agent: AgentRecord, metrics: dict, frontier: Artifact | None, selected_domain: str
) -> float:
    bottleneck_bonus = 0.0
    if agent.role == "engine":
        bottleneck_bonus += max(0.0, 1.0 - metrics["counterexample_density"])
    if agent.role == "lab":
        bottleneck_bonus += max(0.0, 1.0 - metrics["compression"])
    if agent.role == "explorer":
        priority = frontier.metadata.get("priority", 0.5) if frontier else 0.5
        bottleneck_bonus += priority * 0.6
    if agent.role == "meta_observer" and metrics.get("stagnation_score", 0) > 0.5:
        bottleneck_bonus += 0.5
    domain_bonus = 0.15 if agent.domain == selected_domain else 0.0
    return round(max(0.05, agent.reputation + bottleneck_bonus + domain_bonus), 3)


def allocate_attention(
    state: ResearchState, bids: list[MarketBid], budget: float = ATTENTION_BUDGET
) -> list[AttentionAllocation]:
    scored = []
    for bid in bids:
        agent = state.agents[bid.agent_id]
        domain = state.domains.get(agent.domain) or DomainRecord(name=agent.domain)
        score = bid.expected_value * max(0.1, agent.reputation) * max(0.1, domain.priority)
        scored.append((score, bid, agent, domain))
    total_score = sum(score for score, *_ in scored) or 1.0
    allocations: list[AttentionAllocation] = []
    remaining = budget
    for score, bid, agent, domain in sorted(scored, key=lambda item: item[0], reverse=True):
        share = budget * (score / total_score)
        amount = min(bid.requested_attention, max(4.0, share))
        amount = min(amount, remaining)
        if amount <= 0:
            continue
        remaining -= amount
        allocations.append(
            AttentionAllocation(
                agent_id=agent.id,
                role=agent.role,
                target_id=bid.target_id,
                allocated=round(amount, 3),
                requested=bid.requested_attention,
                expected_value=bid.expected_value,
                reputation=agent.reputation,
                domain=domain.name,
                cycle=bid.cycle,
            )
        )
    return allocations


def update_agent_reputation(
    state: ResearchState,
    records: dict[str, AgentRecord],
    allocation: AttentionAllocation,
    output_counts: dict[str, int],
    compression_delta: float,
) -> AgentRecord:
    record = records[allocation.agent_id]
    artifacts = output_counts.get("artifacts", 0)
    edges = output_counts.get("edges", 0)
    updates = output_counts.get("updates", 0)
    useful = artifacts + edges + updates
    stats = dict(record.stats)
    stats["attention_spent"] = stats.get("attention_spent", 0.0) + allocation.allocated
    stats["outputs"] = stats.get("outputs", 0.0) + useful
    stats["compression_contributed"] = stats.get("compression_contributed", 0.0) + max(0.0, compression_delta)
    if allocation.role == "engine" and artifacts:
        stats["successful_falsifications"] = stats.get("successful_falsifications", 0.0) + artifacts
    if allocation.role == "lab" and artifacts:
        stats["reusable_concepts"] = stats.get("reusable_concepts", 0.0) + artifacts
    if allocation.role == "explorer" and artifacts:
        stats["successful_hypotheses"] = stats.get("successful_hypotheses", 0.0) + artifacts
    efficiency = useful / max(1.0, allocation.allocated)
    delta = (efficiency * 0.18) + (max(0.0, compression_delta) * 0.35) - 0.03
    record.reputation = round(min(3.0, max(0.1, record.reputation + delta)), 3)
    record.attention_balance = round(record.attention_balance + allocation.allocated, 3)
    record.last_cycle = allocation.cycle
    record.activity_count += useful
    record.stats = stats
    return record


def update_domains(
    state: ResearchState, allocations: list[AttentionAllocation], metrics: dict, cycle: int
) -> list[DomainRecord]:
    attention = Counter()
    for allocation in allocations:
        attention[allocation.domain] += allocation.allocated
    records = []
    for domain in ensure_domains(state):
        received = attention.get(domain.name, 0.0)
        stats = dict(domain.stats)
        stats["compression"] = metrics.get("compression", 0.0)
        stats["epistemic_value"] = metrics.get("epistemic_value", 0.0)
        if received:
            domain.attention_received += received
            domain.cycles_since_active = 0
            domain.last_cycle = cycle
            domain.priority = round(min(2.0, max(0.2, domain.priority * 0.97 + metrics["epistemic_value"] * 0.03)), 3)
            domain.status = "active"
        else:
            domain.cycles_since_active += 1
            domain.priority = round(max(0.2, domain.priority * 0.99), 3)
            if domain.cycles_since_active > 20:
                domain.status = "inactive"
            if domain.cycles_since_active > 50 and domain.priority > 0.35:
                domain.status = "active"
        domain.stats = stats
        records.append(domain)
    return records


def evolve_agents(state: ResearchState, cycle: int) -> tuple[list[AgentRecord], list[OrganizationAction]]:
    if cycle % 10 != 0:
        return [], []
    active = active_agent_records(state)
    if len(active) < 3:
        return [], []
    updated: list[AgentRecord] = []
    actions: list[OrganizationAction] = []
    weakest = min(active, key=lambda item: item.reputation)
    strongest = max(active, key=lambda item: item.reputation)
    if weakest.reputation < 0.45 and len(active) > 4:
        weakest.status = "retired"
        actions.append(
            OrganizationAction(
                action="retire",
                agent_id=weakest.id,
                reason="Low reputation after repeated attention allocation.",
                cycle=cycle,
            )
        )
        updated.append(weakest)
    if strongest.reputation > 1.35:
        clone_id = f"{strongest.id}_g{strongest.generation + 1}_{cycle}"
        clone = AgentRecord(
            id=clone_id,
            role=strongest.role,
            reputation=round(max(0.8, strongest.reputation * 0.72), 3),
            generation=strongest.generation + 1,
            parent_id=strongest.id,
            strategy=f"{strongest.strategy}_specialized",
            domain=strongest.domain,
            stats={"inherited_reputation": strongest.reputation},
        )
        actions.append(
            OrganizationAction(
                action="clone",
                agent_id=strongest.id,
                new_agent_id=clone.id,
                reason="High-reputation strategy reproduced.",
                cycle=cycle,
            )
        )
        updated.append(clone)
    return updated, actions


def build_meta_memory(state: ResearchState, cycle: int, metrics: dict) -> Artifact | None:
    if cycle % 5 != 0:
        return None
    strong_roles = sorted(
        active_agent_records(state),
        key=lambda agent: agent.stats.get("compression_contributed", 0.0),
        reverse=True,
    )[:3]
    body = (
        "Successful patterns: allocate attention to roles with compression contribution and reusable output. "
        "Failed patterns: preserve low-yield attention paths only long enough to gather evidence. "
        f"Current bottleneck: stagnation score {metrics.get('stagnation_score', 0.0)}."
    )
    return Artifact(
        kind=ArtifactKind.META_MEMORY,
        title=f"Meta Memory cycle {cycle}",
        body=body,
        provenance=Provenance(
            agent_id="organization_memory",
            agent_role="meta_memory",
            cycle=cycle,
            source="organizational_learning",
        ),
        confidence=0.7,
        metadata={
            "cycle": cycle,
            "strong_roles": [agent.role for agent in strong_roles],
            "patterns": ["successful_investigation", "failed_investigation", "bottleneck"],
        },
    )
