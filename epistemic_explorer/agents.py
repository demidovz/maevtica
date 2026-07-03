from __future__ import annotations

import random
from dataclasses import replace

from epistemic_explorer.graph import KnowledgeGraph
from epistemic_explorer.models import CritiqueResult, HypothesisCandidate, KnowledgeNode, SimulationResult


class ExplorerPolicy:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def select_question(self, graph: KnowledgeGraph) -> KnowledgeNode:
        open_questions = graph.open_questions()
        if open_questions:
            return open_questions[0]
        return KnowledgeNode(
            node_id=graph.next_id("q"),
            node_type="question",
            title="What makes a research cycle improve itself?",
            body="Seed question for the first epistemic cycle.",
        )

    def propose_hypotheses(self, question: KnowledgeNode, cycle_index: int) -> list[HypothesisCandidate]:
        templates = [
            (
                "Falsification pressure improves the next cycle",
                "A cycle becomes better when every proposed hypothesis receives at least one explicit failure mode.",
                "If each hypothesis has a critique edge, then the next question set contains more testable questions.",
            ),
            (
                "Reusable concepts are stronger than surviving hypotheses",
                "The best signal of research progress is whether concepts get reused across experiments.",
                "If concepts are reused in independent experiments, then future cycles need fewer new primitives.",
            ),
            (
                "Meta-observation prevents local optimization",
                "Periodic analysis of the research process prevents repeated low-value cycles.",
                "If cycle summaries are compared every N iterations, repeated question families should decline.",
            ),
        ]
        self.rng.shuffle(templates)
        candidates = []
        for title, body, statement in templates[:2]:
            candidates.append(
                HypothesisCandidate(
                    title=f"{title} #{cycle_index}",
                    body=f"{body} Source question: {question.title}",
                    formal_statement=statement,
                    testable=True,
                    assumptions=["cycle logs are complete", "questions can be compared by type"],
                )
            )
        return candidates


class CriticPolicy:
    def critique(self, hypothesis: HypothesisCandidate) -> CritiqueResult:
        counterexamples = []
        hidden_assumptions = list(hypothesis.assumptions)
        severity = 0.35
        if "surviving" in hypothesis.title.lower() or "surviving" in hypothesis.body.lower():
            counterexamples.append("A hypothesis can survive because it is vague rather than because it is useful.")
            severity += 0.25
        if "reused" in hypothesis.body.lower() or "reuse" in hypothesis.title.lower():
            counterexamples.append("A bad abstraction can be reused often if it is too broad.")
            severity += 0.2
        if "meta" in hypothesis.title.lower():
            counterexamples.append("Meta-analysis can consume budget without improving experiment quality.")
            severity += 0.15
        return CritiqueResult(
            summary="The hypothesis is useful only if it produces discriminating observations.",
            counterexamples=counterexamples,
            hidden_assumptions=hidden_assumptions,
            severity=min(severity, 1.0),
        )


class FormalizerPolicy:
    def formalize(self, hypothesis: HypothesisCandidate) -> str:
        return hypothesis.formal_statement


class SimulatorPolicy:
    def simulate(self, hypothesis: HypothesisCandidate, critique: CritiqueResult, cycle_index: int) -> SimulationResult:
        falsified = critique.severity >= 0.7 and cycle_index % 2 == 0
        observations = [
            f"critique_severity={critique.severity:.2f}",
            f"counterexamples={len(critique.counterexamples)}",
            f"testable={hypothesis.testable}",
        ]
        reusable = []
        if "concept" in hypothesis.title.lower() or "reuse" in hypothesis.title.lower():
            reusable.append("concept_reuse")
        if "meta" in hypothesis.title.lower():
            reusable.append("meta_observation")
        if "falsification" in hypothesis.title.lower():
            reusable.append("falsification_pressure")
        return SimulationResult(
            experiment_title=f"Toy simulation for {hypothesis.title}",
            setup={"cycle_index": cycle_index, "rule": "severity>=0.70 and even cycle falsifies"},
            observations=observations,
            falsified=falsified,
            reusable_concepts=reusable,
        )


class AnalystPolicy:
    def status_after_result(self, hypothesis_node: KnowledgeNode, result: SimulationResult) -> KnowledgeNode:
        status = "falsified" if result.falsified else "survived"
        body = f"{hypothesis_node.body}\n\nAnalysis: {'falsified' if result.falsified else 'survived'} in toy simulation."
        metadata = dict(hypothesis_node.metadata)
        metadata["last_simulation_falsified"] = result.falsified
        return replace(hypothesis_node, status=status, body=body, metadata=metadata)

    def next_questions(self, hypothesis: KnowledgeNode, result: SimulationResult) -> list[tuple[str, str]]:
        if result.falsified:
            return [
                (
                    f"What weaker version of {hypothesis.title} survives?",
                    "A falsified hypothesis should produce a narrower replacement question.",
                )
            ]
        return [
            (
                f"Can {hypothesis.title} transfer to an independent toy world?",
                "A surviving hypothesis should be tested outside its original setup.",
            )
        ]


class MetaObserverPolicy:
    def observe(self, graph: KnowledgeGraph, cycle_index: int) -> list[str]:
        notes: list[str] = []
        if cycle_index == 0 or cycle_index % 5 != 0:
            return notes
        open_questions = graph.open_questions()
        falsified = graph.nodes_by_type("hypothesis", status="falsified")
        survived = graph.nodes_by_type("hypothesis", status="survived")
        if len(open_questions) > 20:
            notes.append("Question frontier is growing quickly; prioritize selection pressure.")
        if not falsified:
            notes.append("No falsified hypotheses yet; increase critic severity or experiment discrimination.")
        if len(survived) > len(falsified) * 3 + 3:
            notes.append("Survival dominates falsification; check whether hypotheses are too vague.")
        return notes

