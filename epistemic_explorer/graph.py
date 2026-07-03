from __future__ import annotations

from collections import defaultdict, deque

from epistemic_explorer.models import EdgeType, KnowledgeEdge, KnowledgeNode, NodeType
from epistemic_explorer.storage import KnowledgeStore


class KnowledgeGraph:
    def __init__(self, store: KnowledgeStore) -> None:
        self.store = store
        self.nodes: dict[str, KnowledgeNode] = {}
        self.edges: list[KnowledgeEdge] = []

    @classmethod
    def load(cls, store: KnowledgeStore) -> "KnowledgeGraph":
        graph = cls(store)
        graph.nodes = {node.node_id: node for node in store.load_nodes()}
        graph.edges = store.load_edges()
        return graph

    def save(self) -> None:
        self.store.save_nodes(list(self.nodes.values()))
        self.store.save_edges(self.edges)

    def add_node(self, node: KnowledgeNode) -> None:
        self.nodes[node.node_id] = node
        self.store.append_event("node_added", node.to_dict())

    def replace_node(self, node: KnowledgeNode) -> None:
        self.nodes[node.node_id] = node
        self.store.append_event("node_updated", node.to_dict())

    def add_edge(self, source_id: str, target_id: str, edge_type: EdgeType, **metadata: object) -> None:
        edge = KnowledgeEdge(source_id=source_id, target_id=target_id, edge_type=edge_type, metadata=dict(metadata))
        self.edges.append(edge)
        self.store.append_event("edge_added", edge.to_dict())

    def next_id(self, prefix: str) -> str:
        existing = [
            int(node_id.rsplit("-", 1)[1])
            for node_id in self.nodes
            if node_id.startswith(f"{prefix}-") and node_id.rsplit("-", 1)[1].isdigit()
        ]
        return f"{prefix}-{max(existing, default=0) + 1:04d}"

    def open_questions(self) -> list[KnowledgeNode]:
        return self.nodes_by_type("question", status="open")

    def nodes_by_type(self, node_type: NodeType, status: str | None = None) -> list[KnowledgeNode]:
        nodes = [node for node in self.nodes.values() if node.node_type == node_type]
        if status is not None:
            nodes = [node for node in nodes if node.status == status]
        return sorted(nodes, key=lambda node: node.created_at)

    def outgoing(self, node_id: str, edge_type: EdgeType | None = None) -> list[KnowledgeEdge]:
        return [
            edge
            for edge in self.edges
            if edge.source_id == node_id and (edge_type is None or edge.edge_type == edge_type)
        ]

    def incoming(self, node_id: str, edge_type: EdgeType | None = None) -> list[KnowledgeEdge]:
        return [
            edge
            for edge in self.edges
            if edge.target_id == node_id and (edge_type is None or edge.edge_type == edge_type)
        ]

    def max_depth(self) -> int:
        children: dict[str, list[str]] = defaultdict(list)
        indegree = {node_id: 0 for node_id in self.nodes}
        for edge in self.edges:
            if edge.edge_type in {"depends_on", "generalizes", "uses"}:
                children[edge.source_id].append(edge.target_id)
                indegree[edge.target_id] = indegree.get(edge.target_id, 0) + 1
        queue = deque((node_id, 1) for node_id, degree in indegree.items() if degree == 0)
        visited: set[str] = set()
        max_depth = 0
        while queue:
            node_id, depth = queue.popleft()
            if node_id in visited:
                continue
            visited.add(node_id)
            max_depth = max(max_depth, depth)
            for child_id in children.get(node_id, []):
                queue.append((child_id, depth + 1))
        if len(visited) < len(self.nodes):
            max_depth = max(max_depth, 1)
        return max_depth

    def write_graphviz(self, path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            handle.write("digraph epistemic_explorer {\n")
            handle.write("  rankdir=LR;\n")
            handle.write("  node [shape=box, fontsize=10];\n")
            for node in sorted(self.nodes.values(), key=lambda item: item.node_id):
                label = f"{node.node_id}\\n{node.node_type}\\n{node.title}\\n{node.status}"
                handle.write(f'  "{node.node_id}" [label="{label}"];\n')
            for edge in self.edges:
                handle.write(f'  "{edge.source_id}" -> "{edge.target_id}" [label="{edge.edge_type}"];\n')
            handle.write("}\n")
