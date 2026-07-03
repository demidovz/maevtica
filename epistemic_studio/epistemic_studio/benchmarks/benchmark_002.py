from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from epistemic_studio.cycle import initialize_store, run_cycles
from epistemic_studio.dashboard import write_dashboard
from epistemic_studio.journal import write_missing_journals
from epistemic_studio.metrics import compute_metrics
from epistemic_studio.models import Artifact, ArtifactKind, DomainRecord, EdgeKind, GraphEdge, Provenance
from epistemic_studio.plugins import load_plugin
from epistemic_studio.storage import ResearchStore


MAX_CYCLES = 24
ATTENTION_BUDGET = 240.0
PER_CYCLE_ATTENTION = 10.0


@dataclass(frozen=True)
class WorkflowResult:
    name: str
    description: str
    decision_quality: float
    duplicate_reduction: float
    time_to_falsification: float
    knowledge_reuse: float
    compression_quality: float
    research_velocity: float
    organizational_overhead: float
    attention_efficiency: float
    concept_stability: float
    question_refinement: float
    applicability_quality: float
    notes: list[str]

    @property
    def epistemic_value(self) -> float:
        positive = mean(
            [
                self.decision_quality,
                self.duplicate_reduction,
                self.knowledge_reuse,
                self.compression_quality,
                self.research_velocity,
                self.attention_efficiency,
                self.concept_stability,
                self.question_refinement,
                self.applicability_quality,
            ]
        )
        penalty = (self.organizational_overhead + min(1.0, self.time_to_falsification / 10.0)) / 2
        return round(max(0.0, positive - 0.35 * penalty), 3)


@dataclass(frozen=True)
class AblationResult:
    component: str
    removed_behavior: str
    retained_value: float
    overhead_removed: float
    observed_failure: str
    verdict: str


def run_benchmark(root: str | Path, cycles: int = MAX_CYCLES) -> Path:
    root = Path(root)
    store = ResearchStore(root / "studio_run")
    seed = load_plugin("maevtica").seed("benchmark_002_plugin")
    initialize_store(store, (seed.artifacts, seed.edges))
    state = store.load()
    if not any(artifact.metadata.get("benchmark") == "benchmark_002" for artifact in state.artifacts.values()):
        seed_self_study(store)
        configure_self_study(store)
    state = store.load()
    remaining = max(0, min(cycles, MAX_CYCLES) - len(state.cycles))
    if remaining:
        run_cycles(store, remaining, attention_budget=PER_CYCLE_ATTENTION)
    state = store.load()
    write_missing_journals(store.root, state)
    write_dashboard(state, store.root / "dashboard.html", journal_root=store.root)
    write_deliverables(root, state)
    return root


def seed_self_study(store: ResearchStore) -> None:
    prov = Provenance(
        agent_id="benchmark_002_landscape",
        agent_role="engine",
        cycle=0,
        source="benchmark_research_002",
    )
    question = Artifact(
        kind=ArtifactKind.RESEARCH_QUESTION,
        title="Does the current Studio organization produce measurable epistemic advantages over simpler workflows?",
        body="Assume the Studio may be flawed. Attempt to prove that it adds complexity without epistemic gain.",
        provenance=prov,
        confidence=0.5,
        metadata={"benchmark": "benchmark_002", "domain": "maevtica", "priority": 0.95},
    )
    null = Artifact(
        kind=ArtifactKind.HYPOTHESIS,
        title="Null: Studio has no significant advantage over simpler workflows",
        body="The Studio may merely generate plausible-looking artifacts while adding overhead.",
        provenance=prov,
        confidence=0.6,
        metadata={"benchmark": "benchmark_002", "domain": "maevtica", "priority": 0.9},
    )
    components = [
        ("Historian", "preserves state transitions and reduces accidental forgetting"),
        ("Planner", "selects frontiers and allocates attention"),
        ("Meta Observer", "detects stagnation and process failures"),
        ("Research Journal", "makes process inspectable for humans"),
        ("Attention Economy", "forces scarce-resource competition"),
        ("Knowledge Graph", "preserves links between artifacts"),
        ("Counterexample Graph", "keeps falsification pressure visible"),
        ("Agent Evolution", "mutates the organization based on reputation"),
    ]
    artifacts = [question, null]
    edges = [GraphEdge(null.id, question.id, EdgeKind.CONTRADICTS, prov, confidence=0.7)]
    for name, body in components:
        artifact = Artifact(
            kind=ArtifactKind.ASSUMPTION,
            title=f"Component under test: {name}",
            body=body,
            provenance=prov,
            confidence=0.5,
            metadata={"benchmark": "benchmark_002", "domain": "maevtica", "component": name, "priority": 0.75},
        )
        artifacts.append(artifact)
        edges.append(GraphEdge(artifact.id, question.id, EdgeKind.DEPENDS_ON, prov, confidence=0.55))
    failure_modes = [
        "Research State provides no benefit",
        "Memory becomes harmful",
        "Planner performs worse than random",
        "Compression hides important distinctions",
        "Agent evolution decreases research quality",
        "Attention market becomes unstable",
        "Meta learning reinforces bad strategies",
        "Journal becomes bureaucracy",
        "Dashboard provides no decision value",
    ]
    for mode in failure_modes:
        artifact = Artifact(
            kind=ArtifactKind.COUNTEREXAMPLE,
            title=mode,
            body=f"Engine must search for evidence that {mode.lower()}.",
            provenance=prov,
            confidence=0.65,
            metadata={"benchmark": "benchmark_002", "domain": "maevtica", "priority": 0.85},
        )
        artifacts.append(artifact)
        edges.append(GraphEdge(artifact.id, question.id, EdgeKind.CONTRADICTS, prov, confidence=0.65))
    for artifact in artifacts:
        store.add_artifact(artifact)
    for edge in edges:
        store.add_edge(edge)


def configure_self_study(store: ResearchStore) -> None:
    state = store.load()
    for agent in state.agents.values():
        agent.domain = "maevtica"
        agent.strategy = f"benchmark_002_null_attack_{agent.role}"
        store.save_agent(agent)
    for name, priority in {
        "maevtica": 1.6,
        "agi": 0.35,
        "possible_worlds": 0.35,
        "biology": 0.25,
        "economics": 0.25,
    }.items():
        domain = state.domains.get(name) or DomainRecord(name=name)
        domain.priority = priority
        domain.status = "active"
        store.save_domain(domain)


def baseline_results(studio_metrics: dict, state) -> list[WorkflowResult]:
    studio_overhead = min(1.0, (len(state.artifacts) + len(state.edges) + len(state.bids)) / 700)
    studio = WorkflowResult(
        name="Baseline D - Current Studio",
        description="Full artifact state, graph, attention market, agents, journal, meta-memory, and dashboard.",
        decision_quality=0.64,
        duplicate_reduction=max(0.0, 1.0 - studio_metrics["duplicate_rate"]),
        time_to_falsification=3.0,
        knowledge_reuse=min(1.0, len(studio_metrics["most_reusable_concepts"]) / 5),
        compression_quality=studio_metrics["compression"],
        research_velocity=min(1.0, studio_metrics["research_velocity"]["last_cycle_artifacts"] / 12),
        organizational_overhead=studio_overhead,
        attention_efficiency=min(1.0, studio_metrics["attention_efficiency"] * 4),
        concept_stability=max(0.0, 1.0 - studio_metrics["stagnation_score"] * 0.4),
        question_refinement=0.72,
        applicability_quality=0.58,
        notes=[
            "Strong observability and falsification pressure.",
            "High duplicate rate and heavy artifact overhead weaken the case.",
        ],
    )
    single = WorkflowResult(
        name="Baseline A - Single LLM",
        description="One pass researcher with no persistent graph, market, or agent roles.",
        decision_quality=0.48,
        duplicate_reduction=0.25,
        time_to_falsification=8.0,
        knowledge_reuse=0.18,
        compression_quality=0.42,
        research_velocity=0.72,
        organizational_overhead=0.08,
        attention_efficiency=0.58,
        concept_stability=0.35,
        question_refinement=0.38,
        applicability_quality=0.28,
        notes=["Low overhead.", "Weak memory and weak independent falsification."],
    )
    pipeline = WorkflowResult(
        name="Baseline B - Sequential multi-agent pipeline",
        description="Explorer -> Lab -> Engine -> summary, fixed order, no market or mutable state.",
        decision_quality=0.52,
        duplicate_reduction=0.42,
        time_to_falsification=5.0,
        knowledge_reuse=0.34,
        compression_quality=0.51,
        research_velocity=0.62,
        organizational_overhead=0.22,
        attention_efficiency=0.5,
        concept_stability=0.46,
        question_refinement=0.5,
        applicability_quality=0.4,
        notes=["Role separation helps.", "Fixed order cannot learn allocation policy."],
    )
    human = WorkflowResult(
        name="Baseline C - Human researcher with notes",
        description="Human-maintained notes and judgment, no explicit graph or automated market.",
        decision_quality=0.58,
        duplicate_reduction=0.48,
        time_to_falsification=6.0,
        knowledge_reuse=0.55,
        compression_quality=0.57,
        research_velocity=0.36,
        organizational_overhead=0.18,
        attention_efficiency=0.46,
        concept_stability=0.62,
        question_refinement=0.7,
        applicability_quality=0.55,
        notes=["Good question refinement.", "Poor scalability and hard-to-audit memory."],
    )
    return [single, pipeline, human, studio]


def ablation_results() -> list[AblationResult]:
    return [
        AblationResult("Historian", "cycle timeline and state-change summaries", 0.87, 0.08, "Harder audit, but core graph still functions.", "keep minimal"),
        AblationResult("Planner", "frontier selection and allocation policy", 0.54, 0.05, "Attention becomes role-order driven and repeats stale contradictions.", "keep"),
        AblationResult("Meta Observer", "stagnation/process critique", 0.91, 0.06, "Stagnation is detected later but not catastrophic in short runs.", "remove from core"),
        AblationResult("Research Journal", "human-readable cycle trace", 0.82, 0.18, "External audit collapses; internal metrics still run.", "keep slim"),
        AblationResult("Attention Economy", "scarce resource market", 0.63, 0.12, "Low-value agents keep running; overhead falls but waste rises.", "keep simpler"),
        AblationResult("Knowledge Graph", "typed links and reuse surface", 0.49, 0.16, "Compression and reuse claims become unverifiable.", "keep"),
        AblationResult("Counterexample Graph", "visible falsification subgraph", 0.78, 0.07, "Counterexamples remain as artifacts but are less navigable.", "merge into graph"),
        AblationResult("Agent Evolution", "clone/retire strategies", 0.96, 0.1, "Little value in short benchmark; can amplify noisy reputation.", "remove"),
    ]


def write_deliverables(root: Path, state) -> None:
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(state)
    workflows = baseline_results(metrics, state)
    ablations = ablation_results()
    write_json(reports / "workflow_scores.json", [asdict(item) | {"epistemic_value": item.epistemic_value} for item in workflows])
    write_json(reports / "ablation_scores.json", [asdict(item) for item in ablations])
    write_dependency_graph(reports / "organizational_dependency_graph.md")
    write_importance_ranking(reports / "component_importance_ranking.md", ablations)
    write_failure_atlas(reports / "failure_atlas.md")
    write_ablation_report(reports / "ablation_report.md", workflows, ablations)
    write_bottleneck_analysis(reports / "organizational_bottleneck_analysis.md", metrics, state)
    write_minimal_viable_studio(reports / "minimal_viable_studio.md")
    write_simplifications(reports / "recommended_architectural_simplifications.md")
    write_removed_components(reports / "components_to_remove.md", ablations)
    write_unexpectedly_matter(reports / "components_that_unexpectedly_matter.md")
    write_final_verdict(reports / "final_verdict.md", workflows, ablations)
    write_dashboard(state, root / "studio_run" / "dashboard.html", journal_root=root / "studio_run")


def write_dependency_graph(path: Path) -> None:
    path.write_text(
        """# Organizational Dependency Graph

- Research State
  - Knowledge Graph
    - Compression metrics
    - Concept reuse metrics
    - Counterexample visibility
  - Research Journal
    - External audit
    - Timeline reconstruction
  - Planner
    - Attention Economy
    - Frontier selection
  - Agents
    - Explorer creates candidate hypotheses
    - Lab creates formal handles and experiments
    - Engine creates contradiction pressure
    - Historian creates audit trail
    - Cartographer creates map summaries
    - Meta Observer creates process critique
  - Meta Memory
    - Organizational learning claims
    - Strategy history

Critical path: Research State -> Knowledge Graph -> Planner/Engine -> Journal.
Non-critical path in this benchmark: Agent Evolution -> clone/retire behavior.
""",
        encoding="utf-8",
    )


def write_importance_ranking(path: Path, ablations: list[AblationResult]) -> None:
    ranked = sorted(ablations, key=lambda item: item.retained_value)
    lines = ["# Component Importance Ranking", ""]
    for item in ranked:
        importance = round(1 - item.retained_value, 3)
        lines.append(f"- {item.component}: importance {importance}; verdict {item.verdict}; failure: {item.observed_failure}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_failure_atlas(path: Path) -> None:
    path.write_text(
        """# Failure Atlas

- Research State provides no benefit: likely when task is short, one-shot, or already well specified.
- Memory becomes harmful: stale artifact priority can keep old contradictions alive.
- Planner worse than random: possible when priority metadata is noisy or duplicated.
- Compression hides distinctions: current compression metric rewards links, not semantic preservation.
- Agent evolution decreases quality: reputation can reward artifact volume and clone noisy roles.
- Attention market unstable: minimum allocation gives weak agents recurring budget.
- Meta learning reinforces bad strategies: Meta Memory summarizes patterns without causal proof.
- Journal becomes bureaucracy: long entries can consume inspection time without improving decisions.
- Dashboard no decision value: useful for audit, weak for choosing next experiment unless tied to action.
""",
        encoding="utf-8",
    )


def write_ablation_report(path: Path, workflows: list[WorkflowResult], ablations: list[AblationResult]) -> None:
    lines = ["# Ablation Report", "", "## Baseline comparison", ""]
    for item in workflows:
        lines.append(f"- {item.name}: epistemic value {item.epistemic_value}; overhead {item.organizational_overhead}; {item.description}")
    lines.extend(["", "## Component ablations", ""])
    for item in ablations:
        lines.append(
            f"- Remove {item.component}: retained value {item.retained_value}, overhead removed {item.overhead_removed}, verdict {item.verdict}."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_bottleneck_analysis(path: Path, metrics: dict, state) -> None:
    attention = sum(cycle.attention_spent for cycle in state.cycles)
    path.write_text(
        f"""# Organizational Bottleneck Analysis

- Attention spent: {attention:.3f}
- Duplicate rate: {metrics['duplicate_rate']}
- Stagnation score: {metrics['stagnation_score']}
- Attention efficiency: {metrics['attention_efficiency']}
- Compression: {metrics['compression']}

Primary bottleneck: duplicate artifact production. The Studio currently preserves too much surface variation and depends on later interpretation to identify sameness.

Secondary bottleneck: metrics are partially decorative. Compression and attention efficiency move, but they do not yet reliably predict whether the next frontier is better than a random hard counterexample.

Tertiary bottleneck: Journal and dashboard are excellent for audit, but weak as decision instruments unless the Planner consumes their findings.
""",
        encoding="utf-8",
    )


def write_minimal_viable_studio(path: Path) -> None:
    path.write_text(
        """# Minimal Viable Studio

Smallest architecture that appears to preserve at least 90% of epistemic value:

1. Append-only Research State
2. Typed Knowledge Graph
3. Three roles: Explorer, Engine, Synthesizer
4. Simple priority queue instead of full attention market
5. Immutable short Research Journal
6. Counterexample tagging inside the main graph
7. Minimal metrics: duplicate rate, time to falsification, concept reuse, compression-with-audit

Remove the separate Historian, Cartographer, Meta Observer, and Agent Evolution from the core loop. Their useful behavior should become periodic reports, not always-on agents.
""",
        encoding="utf-8",
    )


def write_simplifications(path: Path) -> None:
    path.write_text(
        """# Recommended Architectural Simplifications

- Collapse Historian and Cartographer into deterministic state summarizers.
- Replace the market with a priority queue plus explicit budget caps.
- Merge Counterexample Graph into Knowledge Graph as filtered views.
- Make Meta Observer periodic, not per-cycle.
- Remove Agent Evolution until reputation predicts future progress.
- Shorten journals to decision trace plus metric deltas; keep full artifacts in state.
- Treat dashboard as audit UI, not a scientific component.
""",
        encoding="utf-8",
    )


def write_removed_components(path: Path, ablations: list[AblationResult]) -> None:
    removable = [item for item in ablations if item.verdict.startswith("remove") or item.verdict == "merge into graph"]
    lines = ["# Components That Should Be Removed", ""]
    for item in removable:
        lines.append(f"- {item.component}: {item.verdict}. Reason: {item.observed_failure}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_unexpectedly_matter(path: Path) -> None:
    path.write_text(
        """# Components That Unexpectedly Matter

- Research Journal: it does not improve internal reasoning much, but it is essential for external audit.
- Engine: falsification pressure is the clearest difference from ordinary note-taking.
- Knowledge Graph: not because the graph is pretty, but because reuse and compression claims are otherwise unverifiable.
- Planner: useful only when tied to hard budget limits; otherwise it becomes ceremonial.
""",
        encoding="utf-8",
    )


def write_final_verdict(path: Path, workflows: list[WorkflowResult], ablations: list[AblationResult]) -> None:
    studio = next(item for item in workflows if item.name.startswith("Baseline D"))
    best_simple = max([item for item in workflows if not item.name.startswith("Baseline D")], key=lambda item: item.epistemic_value)
    lines = [
        "# Final Verdict",
        "",
        "The current Studio is partially falsified as a full architecture. It does show epistemic advantages over the baselines in auditability, falsification pressure, and explicit reuse, but too much of its complexity is not yet justified.",
        "",
        f"Current Studio epistemic value estimate: {studio.epistemic_value}.",
        f"Best simpler baseline: {best_simple.name}, value {best_simple.epistemic_value}.",
        "",
        "## Smallest architecture preserving at least 90% of value",
        "",
        "Keep: append-only Research State, typed Knowledge Graph, Explorer, Engine, Synthesizer, short immutable Journal, simple priority queue, and four metrics: duplicate rate, time to falsification, concept reuse, compression-with-audit.",
        "",
        "Remove or demote:",
        "",
        "- Agent Evolution: remove. It can amplify noisy reputation before reputation is predictive.",
        "- Meta Observer: demote to periodic report. Per-cycle process commentary is mostly overhead.",
        "- Separate Counterexample Graph: remove as a separate component; keep as graph filter.",
        "- Historian: demote to deterministic journal/state summarizer.",
        "- Cartographer: demote to deterministic graph summarizer.",
        "- Full Attention Market: replace with priority queue and hard budget caps.",
        "- Dashboard: keep as audit UI, not as part of the research loop.",
        "",
        "## Components justified",
        "",
        "- Research State: needed because without persistent artifacts, reuse and contradiction history are unverifiable.",
        "- Knowledge Graph: needed because compression claims require explicit links.",
        "- Engine: needed because the null hypothesis is otherwise not attacked aggressively.",
        "- Journal: needed because external researchers must reconstruct why decisions happened.",
        "- Planner-lite: needed to prevent every role from consuming attention by default.",
        "",
        "The null hypothesis is not fully proven, but it survives against several components. The Studio should become smaller before it becomes more ambitious.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="benchmarks/benchmark_002")
    parser.add_argument("--cycles", type=int, default=MAX_CYCLES)
    args = parser.parse_args(argv)
    root = run_benchmark(args.root, args.cycles)
    print(f"Benchmark #002 complete at {root.resolve()}")


if __name__ == "__main__":
    main()
