from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import (
    AgentRecord,
    Artifact,
    AttentionAllocation,
    CycleRecord,
    DomainRecord,
    GraphEdge,
    MarketBid,
    OrganizationAction,
    ResearchState,
    now_iso,
)


class ResearchStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.events_path = self.root / "events.jsonl"
        self.snapshots = self.root / "snapshots"
        self.cycles_dir = self.root / "cycles"

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.snapshots.mkdir(parents=True, exist_ok=True)
        self.cycles_dir.mkdir(parents=True, exist_ok=True)
        self.events_path.touch(exist_ok=True)

    def append_event(self, event_type: str, payload: dict[str, Any]) -> None:
        self.ensure()
        event = {"time": now_iso(), "type": event_type, "payload": payload}
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")

    def load(self) -> ResearchState:
        self.ensure()
        state = ResearchState()
        with self.events_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                event = json.loads(line)
                payload = event["payload"]
                match event["type"]:
                    case "artifact_created" | "artifact_updated":
                        artifact = Artifact.from_dict(payload)
                        state.artifacts[artifact.id] = artifact
                    case "edge_created":
                        edge = GraphEdge.from_dict(payload)
                        state.edges[edge.id] = edge
                    case "agent_registered" | "agent_updated":
                        agent = AgentRecord(**payload)
                        state.agents[agent.id] = agent
                    case "bid_created":
                        bid = MarketBid(**payload)
                        state.bids[bid.id] = bid
                    case "allocation_created":
                        allocation = AttentionAllocation(**payload)
                        state.allocations[allocation.id] = allocation
                    case "organization_action":
                        action = OrganizationAction(**payload)
                        state.organization_actions[action.id] = action
                    case "domain_updated":
                        domain = DomainRecord(**payload)
                        state.domains[domain.name] = domain
                    case "cycle_completed":
                        state.cycles.append(CycleRecord(**payload))
        return state

    def add_artifact(self, artifact: Artifact) -> None:
        self.append_event("artifact_created", artifact.to_dict())

    def update_artifact(self, artifact: Artifact) -> None:
        self.append_event("artifact_updated", artifact.to_dict())

    def add_edge(self, edge: GraphEdge) -> None:
        self.append_event("edge_created", edge.to_dict())

    def save_agent(self, agent: AgentRecord) -> None:
        event_type = "agent_registered" if agent.activity_count == 0 and agent.last_cycle == 0 else "agent_updated"
        self.append_event(event_type, agent.to_dict())

    def add_bid(self, bid: MarketBid) -> None:
        self.append_event("bid_created", bid.to_dict())

    def add_allocation(self, allocation: AttentionAllocation) -> None:
        self.append_event("allocation_created", allocation.to_dict())

    def add_organization_action(self, action: OrganizationAction) -> None:
        self.append_event("organization_action", action.to_dict())

    def save_domain(self, domain: DomainRecord) -> None:
        self.append_event("domain_updated", domain.to_dict())

    def add_cycle(self, cycle: CycleRecord) -> None:
        self.append_event("cycle_completed", cycle.to_dict())
        cycle_path = self.cycles_dir / f"{cycle.cycle:06d}.json"
        cycle_path.write_text(json.dumps(cycle.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    def write_snapshot(self, state: ResearchState) -> None:
        self.snapshots.mkdir(parents=True, exist_ok=True)
        payload = {
            "artifacts": {key: value.to_dict() for key, value in state.artifacts.items()},
            "edges": {key: value.to_dict() for key, value in state.edges.items()},
            "agents": {key: value.to_dict() for key, value in state.agents.items()},
            "bids": {key: value.to_dict() for key, value in state.bids.items()},
            "allocations": {key: value.to_dict() for key, value in state.allocations.items()},
            "organization_actions": {
                key: value.to_dict() for key, value in state.organization_actions.items()
            },
            "domains": {key: value.to_dict() for key, value in state.domains.items()},
            "cycles": [cycle.to_dict() for cycle in state.cycles],
        }
        (self.snapshots / "research_state.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
