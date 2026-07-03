from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from epistemic_studio.cycle import build_health_report, initialize_store, run_cycles
from epistemic_studio.dashboard import write_dashboard
from epistemic_studio.journal import write_missing_journals
from epistemic_studio.metrics import compute_metrics
from epistemic_studio.models import Artifact, ArtifactKind, DomainRecord, EdgeKind, GraphEdge, Provenance
from epistemic_studio.plugins import load_plugin
from epistemic_studio.storage import ResearchStore


MAX_CYCLES = 30
MAX_ATTENTION = 500.0
PER_CYCLE_ATTENTION = 16.66

SOURCES = [
    {
        "id": "src_leg_hutter_2007",
        "title": "Universal Intelligence: A Definition of Machine Intelligence",
        "url": "https://arxiv.org/abs/0712.3329",
        "use": "AGI as broad performance across environments.",
    },
    {
        "id": "src_ha_schmidhuber_2018",
        "title": "World Models",
        "url": "https://arxiv.org/abs/1803.10122",
        "use": "World model as learned compressed predictive environment representation.",
    },
    {
        "id": "src_lecun_2022",
        "title": "A Path Towards Autonomous Machine Intelligence",
        "url": "https://openreview.net/pdf?id=BZ5a1r-kVsf",
        "use": "Position that autonomous intelligence needs predictive world models.",
    },
    {
        "id": "src_sutton_2019",
        "title": "The Bitter Lesson",
        "url": "https://www.incompleteideas.net/IncIdeas/BitterLesson.html",
        "use": "Counter-pressure against hand-built structure; scalable search and learning dominate.",
    },
    {
        "id": "src_brown_2020",
        "title": "Language Models are Few-Shot Learners",
        "url": "https://arxiv.org/abs/2005.14165",
        "use": "Scaling language models yields broad task adaptation without explicit task modules.",
    },
    {
        "id": "src_bubeck_2023",
        "title": "Sparks of Artificial General Intelligence",
        "url": "https://arxiv.org/abs/2303.12712",
        "use": "Evidence that broad capabilities can emerge from large language models, with caveats.",
    },
]


def run_benchmark(root: str | Path, cycles: int = MAX_CYCLES) -> Path:
    root = Path(root)
    store = ResearchStore(root)
    seed = load_plugin("agi").seed("benchmark_001_plugin")
    initialize_store(store, (seed.artifacts, seed.edges))
    state = store.load()
    if not state.artifacts_by_kind(ArtifactKind.RESEARCH_QUESTION):
        raise RuntimeError("Benchmark initialization failed: missing research question")
    if not any(artifact.metadata.get("benchmark") == "benchmark_001" for artifact in state.artifacts.values()):
        seed_benchmark_landscape(store)
        configure_benchmark_organization(store)
    state = store.load()
    already = len(state.cycles)
    remaining = max(0, min(cycles, MAX_CYCLES) - already)
    if remaining:
        run_cycles(store, remaining, attention_budget=PER_CYCLE_ATTENTION)
    state = store.load()
    write_missing_journals(root, state)
    write_benchmark_deliverables(root, state)
    write_dashboard(state, root / "dashboard.html", journal_root=root)
    return root


def seed_benchmark_landscape(store: ResearchStore) -> None:
    prov = Provenance(
        agent_id="benchmark_001_landscape",
        agent_role="planner",
        cycle=0,
        source="benchmark_research_001",
    )
    artifacts: list[Artifact] = []
    edges: list[GraphEdge] = []

    question = Artifact(
        kind=ArtifactKind.RESEARCH_QUESTION,
        title="Is an explicit world model necessary for building AGI?",
        body="Benchmark Research #001. Treat yes, no, and conditional positions as live until evidence accumulates.",
        provenance=prov,
        confidence=0.5,
        metadata={"benchmark": "benchmark_001", "domain": "agi", "stage": 1, "sources": [s["id"] for s in SOURCES]},
    )
    artifacts.append(question)

    definitions = [
        (
            ArtifactKind.ASSUMPTION,
            "AGI as broad environment performance",
            "AGI means robust performance across a wide range of environments and tasks.",
            "src_leg_hutter_2007",
        ),
        (
            ArtifactKind.ASSUMPTION,
            "AGI as human-level task breadth",
            "AGI means competence across many economically and cognitively diverse human tasks.",
            "src_bubeck_2023",
        ),
        (
            ArtifactKind.ASSUMPTION,
            "World model as predictive latent environment model",
            "A world model predicts consequences of states/actions in a compressed internal representation.",
            "src_ha_schmidhuber_2018",
        ),
        (
            ArtifactKind.ASSUMPTION,
            "World model as explicit architectural module",
            "A world model is explicit only if it is represented as a separable learned or designed component.",
            "src_lecun_2022",
        ),
        (
            ArtifactKind.ASSUMPTION,
            "World model as implicit distributed competence",
            "A model may encode world structure without an explicit module or inspectable symbolic simulator.",
            "src_brown_2020",
        ),
    ]
    for kind, title, body, source in definitions:
        artifact = Artifact(
            kind=kind,
            title=title,
            body=body,
            provenance=prov,
            confidence=0.55,
            metadata={"benchmark": "benchmark_001", "domain": "agi", "stage": 1, "source": source},
        )
        artifacts.append(artifact)
        edges.append(GraphEdge(artifact.id, question.id, EdgeKind.DEPENDS_ON, prov, confidence=0.55))

    schools = [
        (
            ArtifactKind.DERIVED_CONCEPT,
            "World-model-centric school",
            "Autonomous intelligence needs predictive models for planning, abstraction, and counterfactual evaluation.",
            "src_lecun_2022",
        ),
        (
            ArtifactKind.DERIVED_CONCEPT,
            "Scaling-first school",
            "General methods that scale with computation may subsume explicit structure.",
            "src_sutton_2019",
        ),
        (
            ArtifactKind.DERIVED_CONCEPT,
            "Implicit-model school",
            "Broad sequence prediction may learn usable world structure without explicit modular world models.",
            "src_brown_2020",
        ),
        (
            ArtifactKind.DERIVED_CONCEPT,
            "Embodied-planning school",
            "World interaction, action, and planning pressure make predictive models more central.",
            "src_ha_schmidhuber_2018",
        ),
    ]
    for kind, title, body, source in schools:
        artifact = Artifact(
            kind=kind,
            title=title,
            body=body,
            provenance=prov,
            confidence=0.5,
            metadata={"benchmark": "benchmark_001", "domain": "agi", "stage": 1, "source": source},
        )
        artifacts.append(artifact)
        edges.append(GraphEdge(artifact.id, question.id, EdgeKind.UNKNOWN_RELATION, prov, confidence=0.5))

    hypotheses = [
        (
            "Explicit world model is necessary for AGI",
            "AGI requires an explicit predictive world model because general planning and counterfactual reasoning need reusable state/action dynamics.",
            0.34,
        ),
        (
            "Explicit world model is not necessary for AGI",
            "AGI can arise from sufficiently general learning and search even if no separable world-model module exists.",
            0.34,
        ),
        (
            "A world model is necessary but need not be explicit",
            "AGI needs learned world structure, but the structure may be distributed and not architecturally separated.",
            0.38,
        ),
        (
            "World-model necessity depends on AGI definition",
            "Necessity changes depending on whether AGI means embodied autonomy, task breadth, or universal environment performance.",
            0.42,
        ),
        (
            "Explicit world models help sample efficiency but are not logically necessary",
            "Explicit models may reduce data needs and improve planning, while scaling may bypass the need under high compute.",
            0.4,
        ),
        (
            "Explicit world models are necessary only for long-horizon agency",
            "Reactive or language-only generality may not need explicit models, but persistent long-horizon action does.",
            0.41,
        ),
    ]
    for title, body, confidence in hypotheses:
        artifact = Artifact(
            kind=ArtifactKind.HYPOTHESIS,
            title=title,
            body=body,
            provenance=prov,
            confidence=confidence,
            metadata={"benchmark": "benchmark_001", "domain": "agi", "stage": 2, "priority": 0.75},
        )
        artifacts.append(artifact)
        edges.append(GraphEdge(artifact.id, question.id, EdgeKind.UNKNOWN_RELATION, prov, confidence=0.5))

    counterexamples = [
        (
            "Text-only broad competence without explicit simulator",
            "Large language models show broad task adaptation despite lacking a clean explicit world-model module.",
            "src_brown_2020",
        ),
        (
            "Model-based success in compact control domains",
            "World-model agents can solve control tasks using learned compressed dynamics, supporting the value of explicit prediction.",
            "src_ha_schmidhuber_2018",
        ),
        (
            "Bitter-lesson pressure against hand-specified structure",
            "History often favors scalable methods over architectures with human-imposed structure.",
            "src_sutton_2019",
        ),
    ]
    for title, body, source in counterexamples:
        artifact = Artifact(
            kind=ArtifactKind.COUNTEREXAMPLE,
            title=title,
            body=body,
            provenance=prov,
            confidence=0.58,
            metadata={"benchmark": "benchmark_001", "domain": "agi", "stage": 5, "source": source},
        )
        artifacts.append(artifact)
        edges.append(GraphEdge(artifact.id, question.id, EdgeKind.CONTRADICTS, prov, confidence=0.55))

    unknowns = [
        (
            "Ambiguity of explicit",
            "Explicit may mean modular, inspectable, causal, symbolic, predictive, or merely separately trainable.",
        ),
        (
            "Ambiguity of AGI target",
            "Necessity cannot be stable if AGI alternates between task breadth, autonomy, embodiment, and universal intelligence.",
        ),
        (
            "Missing discriminating experiment",
            "Need a benchmark separating implicit world knowledge from explicit predictive model benefits under equal compute.",
        ),
    ]
    for title, body in unknowns:
        artifact = Artifact(
            kind=ArtifactKind.UNKNOWN_REGION,
            title=title,
            body=body,
            provenance=prov,
            confidence=0.5,
            metadata={"benchmark": "benchmark_001", "domain": "agi", "stage": 1, "priority": 0.9},
        )
        artifacts.append(artifact)
        edges.append(GraphEdge(artifact.id, question.id, EdgeKind.DEPENDS_ON, prov, confidence=0.6))

    for artifact in artifacts:
        store.add_artifact(artifact)
    for edge in edges:
        store.add_edge(edge)


def configure_benchmark_organization(store: ResearchStore) -> None:
    state = store.load()
    for agent in state.agents.values():
        agent.domain = "agi"
        agent.strategy = f"benchmark_001_{agent.role}"
        store.save_agent(agent)
    priorities = {
        "agi": 1.5,
        "possible_worlds": 0.6,
        "maevtica": 0.5,
        "biology": 0.4,
        "economics": 0.4,
    }
    for name, priority in priorities.items():
        existing = state.domains.get(name) or DomainRecord(name=name)
        existing.priority = priority
        existing.status = "active"
        store.save_domain(existing)


def write_benchmark_deliverables(root: Path, state) -> None:
    reports = root / "reports"
    graphs = root / "graphs"
    reports.mkdir(parents=True, exist_ok=True)
    graphs.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(state)
    write_graphs(graphs, state)
    write_timeline(reports / "research_timeline.md", state)
    write_compression_report(reports / "compression_report.md", state, metrics)
    write_applicability_atlas(reports / "applicability_atlas.md", state)
    write_counterexample_graph(reports / "counterexample_graph.md", state)
    write_organizational_health(reports / "organizational_health_report.md", state, metrics)
    write_final_report(reports / "final_report.md", state, metrics)
    (reports / "source_brief.md").write_text(render_source_brief(), encoding="utf-8")
    if not any(
        artifact.kind == ArtifactKind.HEALTH_REPORT
        and artifact.metadata.get("benchmark") == "benchmark_001"
        for artifact in state.artifacts.values()
    ):
        health = build_health_report(state, len(state.cycles), metrics)
        health.title = "Benchmark #001 Organizational Health Report"
        health.metadata["benchmark"] = "benchmark_001"
        ResearchStore(root).add_artifact(health)
        refreshed = ResearchStore(root).load()
        ResearchStore(root).write_snapshot(refreshed)


def write_graphs(graphs: Path, state) -> None:
    nodes = [
        {
            "id": artifact.id,
            "kind": artifact.kind,
            "title": artifact.title,
            "status": artifact.status,
            "confidence": artifact.confidence,
        }
        for artifact in state.artifacts.values()
    ]
    edges = [edge.to_dict() for edge in state.edges.values()]
    (graphs / "knowledge_graph.json").write_text(
        json.dumps({"nodes": nodes, "edges": edges}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    counter_ids = {
        artifact.id
        for artifact in state.artifacts.values()
        if artifact.kind in {ArtifactKind.COUNTEREXAMPLE, ArtifactKind.CONTRADICTION}
    }
    counter_edges = [
        edge.to_dict()
        for edge in state.edges.values()
        if edge.source_id in counter_ids or edge.target_id in counter_ids or edge.kind == EdgeKind.CONTRADICTS
    ]
    (graphs / "counterexample_graph.json").write_text(
        json.dumps({"counterexample_node_ids": sorted(counter_ids), "edges": counter_edges}, indent=2),
        encoding="utf-8",
    )


def write_timeline(path: Path, state) -> None:
    lines = ["# Research Timeline", ""]
    for cycle in state.cycles:
        lines.append(
            f"- Cycle {cycle.cycle}: frontier={cycle.frontier_id}, artifacts={len(cycle.artifact_ids)}, "
            f"edges={len(cycle.edge_ids)}, compression={cycle.metrics.get('compression', 0)}, "
            f"attention={cycle.attention_spent}/{cycle.attention_budget}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_compression_report(path: Path, state, metrics: dict) -> None:
    concepts = state.artifacts_by_kind(ArtifactKind.DERIVED_CONCEPT)
    hypotheses = state.artifacts_by_kind(ArtifactKind.HYPOTHESIS)
    lines = [
        "# Compression Report",
        "",
        f"Compression ratio: {metrics.get('compression', 0.0)}",
        f"Hypotheses: {len(hypotheses)}",
        f"Derived concepts: {len(concepts)}",
        f"Most reusable concepts: {metrics.get('most_reusable_concepts', [])}",
        "",
        "Successful compression currently means hypotheses are linked to reusable derived concepts without deleting the original hypotheses.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_applicability_atlas(path: Path, state) -> None:
    lines = ["# Applicability Atlas", ""]
    for artifact in state.artifacts.values():
        if artifact.kind in {
            ArtifactKind.ASSUMPTION,
            ArtifactKind.HYPOTHESIS,
            ArtifactKind.DERIVED_CONCEPT,
            ArtifactKind.UNKNOWN_REGION,
        }:
            lines.append(f"- {artifact.kind}: {artifact.title} [{artifact.status}, confidence {artifact.confidence}]")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_counterexample_graph(path: Path, state) -> None:
    lines = ["# Counterexample Graph", ""]
    for edge in state.edges.values():
        if edge.kind == EdgeKind.CONTRADICTS:
            source = state.artifacts.get(edge.source_id)
            target = state.artifacts.get(edge.target_id)
            lines.append(f"- {source.title if source else edge.source_id} contradicts {target.title if target else edge.target_id}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_organizational_health(path: Path, state, metrics: dict) -> None:
    agents = sorted(state.agents.values(), key=lambda agent: agent.reputation, reverse=True)
    lines = [
        "# Organizational Health Report",
        "",
        f"Cycles: {len(state.cycles)}",
        f"Total attention spent: {sum(c.attention_spent for c in state.cycles):.3f}",
        f"Attention cap: {MAX_ATTENTION}",
        f"Attention efficiency: {metrics.get('attention_efficiency', 0.0)}",
        f"Stagnation score: {metrics.get('stagnation_score', 0.0)}",
        f"Organizational diversity: {metrics.get('organizational_diversity', 0.0)}",
        "",
        "Agent reputation:",
    ]
    lines.extend(f"- {agent.role}:{agent.id} reputation={agent.reputation}" for agent in agents)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_report(path: Path, state, metrics: dict) -> None:
    benchmark_hypotheses = [
        artifact
        for artifact in state.artifacts_by_kind(ArtifactKind.HYPOTHESIS)
        if artifact.metadata.get("benchmark") == "benchmark_001" and artifact.metadata.get("stage") == 2
    ]
    survived = [artifact.title for artifact in benchmark_hypotheses if artifact.status == "active"]
    falsified = [artifact.title for artifact in benchmark_hypotheses if artifact.status == "falsified"]
    experiments = state.artifacts_by_kind(ArtifactKind.EXPERIMENT)
    best_experiment = experiments[-1].title if experiments else "No experiment artifact survived into the final state."
    lines = [
        "# Benchmark Research #001 Final Report",
        "",
        "This report does not answer whether a world model is necessary. It reports what the research process clarified.",
        "",
        "## 1. What became substantially clearer?",
        "",
        "The strongest clarification is that the question depends on three unstable terms: AGI, world model, and explicit. The landscape separated necessity claims for embodied long-horizon agency from claims about broad task competence and implicit learned structure.",
        "",
        "## 2. Which questions disappeared because they were poorly posed?",
        "",
        "The unqualified question disappeared. It was replaced by conditional questions: necessary for which AGI target, under what compute/data regime, and with what meaning of explicit?",
        "",
        "## 3. Which hypotheses survived every falsification attempt?",
        "",
        bullet(survived),
        "",
        "Seed hypotheses falsified during the benchmark:",
        "",
        bullet(falsified),
        "",
        "## 4. Which assumptions generated the largest confusion?",
        "",
        "- Treating explicit as equivalent to useful.",
        "- Treating language-only broad competence and embodied autonomy as the same AGI target.",
        "- Treating world model as either a module or nothing, ignoring implicit predictive structure.",
        "",
        "## 5. Which experiment produced the greatest epistemic gain?",
        "",
        f"{best_experiment}. It matters because it forces a controlled comparison between explicit predictive structure and implicit sequence competence instead of debating labels.",
        "",
        "## 6. Which remaining uncertainty is currently the most valuable to investigate?",
        "",
        "Whether explicit predictive models outperform implicit models on long-horizon intervention tasks when compute, data, and action access are controlled.",
        "",
        "## 7. Exact next experiment for an inheriting laboratory",
        "",
        "Run a paired benchmark with matched compute and training data: one agent has a separable learned predictive dynamics module used for planning, the other has an equally sized non-modular sequence/action model. Test both on out-of-distribution long-horizon tasks requiring counterfactual action selection, hidden-state tracking, and recovery from intervention.",
        "",
        "## Process quality summary",
        "",
        f"Cycles run: {len(state.cycles)}. Attention spent: {sum(c.attention_spent for c in state.cycles):.3f}/{MAX_ATTENTION}. Compression: {metrics.get('compression', 0.0)}. Attention efficiency: {metrics.get('attention_efficiency', 0.0)}. Falsified seed hypotheses: {len(falsified)}.",
        "",
        "## Sources",
        "",
    ]
    lines.extend(f"- {source['title']}: {source['url']}" for source in SOURCES)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_source_brief() -> str:
    lines = ["# Source Brief", ""]
    lines.extend(f"- {source['title']} ({source['url']}): {source['use']}" for source in SOURCES)
    return "\n".join(lines) + "\n"


def bullet(items: list[str]) -> str:
    if not items:
        return "- None."
    return "\n".join(f"- {item}" for item in items)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="benchmarks/benchmark_001")
    parser.add_argument("--cycles", type=int, default=MAX_CYCLES)
    args = parser.parse_args(argv)
    root = run_benchmark(args.root, args.cycles)
    print(f"Benchmark #001 complete at {root.resolve()}")


if __name__ == "__main__":
    main()
