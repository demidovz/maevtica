from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from epistemic_explorer.models import CycleLog, KnowledgeEdge, KnowledgeNode, utc_now


class KnowledgeStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.knowledge_dir = root / "knowledge"
        self.cycles_dir = self.knowledge_dir / "cycles"
        self.events_path = self.knowledge_dir / "events.jsonl"
        self.nodes_path = self.knowledge_dir / "nodes.json"
        self.edges_path = self.knowledge_dir / "edges.json"
        self.summary_path = self.knowledge_dir / "summary.json"

    def initialize(self) -> None:
        self.cycles_dir.mkdir(parents=True, exist_ok=True)
        if not self.nodes_path.exists():
            self._write_json(self.nodes_path, [])
        if not self.edges_path.exists():
            self._write_json(self.edges_path, [])
        if not self.events_path.exists():
            self.events_path.write_text("", encoding="utf-8")

    def load_nodes(self) -> list[KnowledgeNode]:
        return [KnowledgeNode(**item) for item in self._read_json(self.nodes_path, [])]

    def load_edges(self) -> list[KnowledgeEdge]:
        return [KnowledgeEdge(**item) for item in self._read_json(self.edges_path, [])]

    def save_nodes(self, nodes: list[KnowledgeNode]) -> None:
        self._write_json(self.nodes_path, [node.to_dict() for node in nodes])

    def save_edges(self, edges: list[KnowledgeEdge]) -> None:
        self._write_json(self.edges_path, [edge.to_dict() for edge in edges])

    def append_event(self, event_type: str, payload: dict[str, Any]) -> None:
        event = {
            "created_at": utc_now(),
            "event_type": event_type,
            "payload": payload,
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")

    def write_cycle(self, cycle_log: CycleLog) -> None:
        path = self.cycles_dir / f"{cycle_log.cycle_index:06d}.json"
        self._write_json(path, cycle_log.to_dict())
        self.append_event("cycle_completed", cycle_log.to_dict())

    def next_cycle_index(self) -> int:
        existing = [
            int(path.stem)
            for path in self.cycles_dir.glob("*.json")
            if path.stem.isdigit()
        ]
        return max(existing, default=0) + 1

    def write_summary(self, summary: dict[str, Any]) -> None:
        self._write_json(self.summary_path, summary)

    @staticmethod
    def _read_json(path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            return default
        return json.loads(content)

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
