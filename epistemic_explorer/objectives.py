from __future__ import annotations

from abc import ABC, abstractmethod

from epistemic_explorer.graph import KnowledgeGraph
from epistemic_explorer.models import CycleLog, ObjectiveScore


class Objective(ABC):
    name: str

    @abstractmethod
    def score_cycle(self, before: KnowledgeGraph, after: KnowledgeGraph, cycle_log: CycleLog) -> ObjectiveScore:
        raise NotImplementedError


class KnowledgeGrowthObjective(Objective):
    name = "knowledge_growth"

    def score_cycle(self, before: KnowledgeGraph, after: KnowledgeGraph, cycle_log: CycleLog) -> ObjectiveScore:
        before_questions = len(before.nodes_by_type("question", status="open"))
        after_questions = len(after.nodes_by_type("question", status="open"))
        new_testable_questions = max(0, after_questions - before_questions)
        survived = len(after.nodes_by_type("hypothesis", status="survived")) - len(
            before.nodes_by_type("hypothesis", status="survived")
        )
        falsified = len(after.nodes_by_type("hypothesis", status="falsified")) - len(
            before.nodes_by_type("hypothesis", status="falsified")
        )
        graph_depth_delta = max(0, after.max_depth() - before.max_depth())
        metrics = {
            "new_testable_questions": float(new_testable_questions),
            "hypotheses_survived": float(max(0, survived)),
            "hypotheses_falsified": float(max(0, falsified)),
            "graph_depth_delta": float(graph_depth_delta),
        }
        total = (
            metrics["new_testable_questions"]
            + 0.8 * metrics["hypotheses_survived"]
            + 1.2 * metrics["hypotheses_falsified"]
            + 0.5 * metrics["graph_depth_delta"]
        )
        return ObjectiveScore(self.name, metrics, total)


class ReuseObjective(Objective):
    name = "concept_reuse"

    def score_cycle(self, before: KnowledgeGraph, after: KnowledgeGraph, cycle_log: CycleLog) -> ObjectiveScore:
        before_reuse = sum(1 for edge in before.edges if edge.edge_type == "uses")
        after_reuse = sum(1 for edge in after.edges if edge.edge_type == "uses")
        reuse_delta = max(0, after_reuse - before_reuse)
        metrics = {
            "concept_reuse_delta": float(reuse_delta),
            "experiment_count": float(len(cycle_log.experiment_ids)),
        }
        total = metrics["concept_reuse_delta"] + 0.25 * metrics["experiment_count"]
        return ObjectiveScore(self.name, metrics, total)

