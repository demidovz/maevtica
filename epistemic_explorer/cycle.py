from __future__ import annotations

import copy
import random
from dataclasses import replace
from pathlib import Path

from epistemic_explorer.agents import (
    AnalystPolicy,
    CriticPolicy,
    ExplorerPolicy,
    FormalizerPolicy,
    MetaObserverPolicy,
    SimulatorPolicy,
)
from epistemic_explorer.graph import KnowledgeGraph
from epistemic_explorer.models import CycleLog, KnowledgeNode
from epistemic_explorer.objectives import Objective
from epistemic_explorer.storage import KnowledgeStore


class ResearchCycleRunner:
    def __init__(self, root: Path, objectives: list[Objective], seed: int) -> None:
        self.store = KnowledgeStore(root)
        self.store.initialize()
        self.graph = KnowledgeGraph.load(self.store)
        self.objectives = objectives
        self.rng = random.Random(seed)
        self.explorer = ExplorerPolicy(self.rng)
        self.critic = CriticPolicy()
        self.formalizer = FormalizerPolicy()
        self.simulator = SimulatorPolicy()
        self.analyst = AnalystPolicy()
        self.meta_observer = MetaObserverPolicy()

    def ensure_seed_question(self) -> None:
        if self.graph.open_questions():
            return
        question = self.explorer.select_question(self.graph)
        self.graph.add_node(question)
        self.graph.save()

    def run(self, cycles: int) -> list[CycleLog]:
        self.ensure_seed_question()
        logs = []
        start_index = self.store.next_cycle_index()
        for cycle_index in range(start_index, start_index + cycles):
            logs.append(self.run_one(cycle_index))
        self.graph.write_graphviz(self.store.knowledge_dir / "graph.dot")
        self.store.write_summary(self.summary(logs))
        return logs

    def run_one(self, cycle_index: int) -> CycleLog:
        before = copy.deepcopy(self.graph)
        question = self.explorer.select_question(self.graph)
        if question.node_id not in self.graph.nodes:
            self.graph.add_node(question)
        self.graph.replace_node(replace(question, status="active"))

        hypothesis_ids: list[str] = []
        critique_ids: list[str] = []
        experiment_ids: list[str] = []
        observation_ids: list[str] = []
        new_question_ids: list[str] = []

        for candidate in self.explorer.propose_hypotheses(question, cycle_index):
            hypothesis = KnowledgeNode(
                node_id=self.graph.next_id("h"),
                node_type="hypothesis",
                title=candidate.title,
                body=f"{candidate.body}\n\nFormal statement: {self.formalizer.formalize(candidate)}",
                status="formalized",
                metadata={"testable": candidate.testable, "assumptions": candidate.assumptions},
            )
            self.graph.add_node(hypothesis)
            self.graph.add_edge(question.node_id, hypothesis.node_id, "asks")
            hypothesis_ids.append(hypothesis.node_id)

            critique = self.critic.critique(candidate)
            critique_node = KnowledgeNode(
                node_id=self.graph.next_id("c"),
                node_type="critique",
                title=f"Critique of {candidate.title}",
                body=critique.summary,
                status="complete",
                metadata={
                    "counterexamples": critique.counterexamples,
                    "hidden_assumptions": critique.hidden_assumptions,
                    "severity": critique.severity,
                },
            )
            self.graph.add_node(critique_node)
            self.graph.add_edge(critique_node.node_id, hypothesis.node_id, "falsifies", severity=critique.severity)
            critique_ids.append(critique_node.node_id)

            result = self.simulator.simulate(candidate, critique, cycle_index)
            experiment = KnowledgeNode(
                node_id=self.graph.next_id("e"),
                node_type="experiment",
                title=result.experiment_title,
                body="Minimal deterministic toy simulation.",
                status="complete",
                metadata=result.setup,
            )
            self.graph.add_node(experiment)
            self.graph.add_edge(experiment.node_id, hypothesis.node_id, "tests")
            experiment_ids.append(experiment.node_id)

            observation = KnowledgeNode(
                node_id=self.graph.next_id("o"),
                node_type="observation",
                title=f"Observation for {candidate.title}",
                body="\n".join(result.observations),
                status="complete",
                metadata={"falsified": result.falsified},
            )
            self.graph.add_node(observation)
            self.graph.add_edge(experiment.node_id, observation.node_id, "produces")
            observation_ids.append(observation.node_id)

            updated_hypothesis = self.analyst.status_after_result(hypothesis, result)
            self.graph.replace_node(updated_hypothesis)
            if result.falsified:
                self.graph.add_edge(observation.node_id, hypothesis.node_id, "falsifies")
            else:
                self.graph.add_edge(observation.node_id, hypothesis.node_id, "supports")

            for concept_title in result.reusable_concepts:
                concept = self._get_or_create_concept(concept_title)
                self.graph.add_edge(hypothesis.node_id, concept.node_id, "uses")

            for title, body in self.analyst.next_questions(updated_hypothesis, result):
                next_question = KnowledgeNode(
                    node_id=self.graph.next_id("q"),
                    node_type="question",
                    title=title,
                    body=body,
                    status="open",
                )
                self.graph.add_node(next_question)
                self.graph.add_edge(updated_hypothesis.node_id, next_question.node_id, "produces")
                new_question_ids.append(next_question.node_id)

        self.graph.replace_node(replace(self.graph.nodes[question.node_id], status="closed"))
        after = copy.deepcopy(self.graph)
        interim_log = CycleLog(
            cycle_index=cycle_index,
            selected_question_id=question.node_id,
            selected_question=question.title,
            hypothesis_ids=hypothesis_ids,
            critique_ids=critique_ids,
            experiment_ids=experiment_ids,
            observation_ids=observation_ids,
            new_question_ids=new_question_ids,
            objective_scores=[],
            meta_notes=self.meta_observer.observe(self.graph, cycle_index),
        )
        scores = [objective.score_cycle(before, after, interim_log) for objective in self.objectives]
        cycle_log = replace(interim_log, objective_scores=scores)
        self.graph.save()
        self.store.write_cycle(cycle_log)
        return cycle_log

    def summary(self, logs: list[CycleLog]) -> dict[str, object]:
        objective_totals: dict[str, float] = {}
        for log in logs:
            for score in log.objective_scores:
                objective_totals[score.objective_name] = objective_totals.get(score.objective_name, 0.0) + score.total_score
        return {
            "cycles": len(logs),
            "nodes": len(self.graph.nodes),
            "edges": len(self.graph.edges),
            "open_questions": len(self.graph.open_questions()),
            "hypotheses_survived": len(self.graph.nodes_by_type("hypothesis", status="survived")),
            "hypotheses_falsified": len(self.graph.nodes_by_type("hypothesis", status="falsified")),
            "max_depth": self.graph.max_depth(),
            "objective_totals": objective_totals,
        }

    def _get_or_create_concept(self, title: str) -> KnowledgeNode:
        for node in self.graph.nodes_by_type("concept"):
            if node.title == title:
                return node
        concept = KnowledgeNode(
            node_id=self.graph.next_id("k"),
            node_type="concept",
            title=title,
            body=f"Reusable concept discovered by toy simulations: {title}",
            status="active",
        )
        self.graph.add_node(concept)
        return concept
