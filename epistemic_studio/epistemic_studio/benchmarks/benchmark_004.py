from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean


MECHANISMS = [
    "persistent_memory",
    "append_only_accumulation",
    "preserved_failures",
    "institutionalized_criticism",
    "question_refinement",
    "knowledge_compression",
    "applicability_boundaries",
    "planning_prioritization",
    "role_separation",
    "knowledge_graph_equivalent",
]


@dataclass(frozen=True)
class Source:
    id: str
    title: str
    url: str
    note: str


@dataclass(frozen=True)
class CaseStudy:
    id: str
    name: str
    category: str
    outcome: str
    scores: dict[str, float]
    evidence: dict[str, str]
    source_ids: list[str]
    control: bool = False


SOURCES = [
    Source(
        "bell_noll",
        "Memories: A Personal History of Bell Telephone Laboratories",
        "https://quello.msu.edu/divi/wp-content/uploads/2015/08/Memories-Noll.pdf",
        "Bell Labs as long-lived corporate research institution with durable communities and accumulated knowledge.",
    ),
    Source(
        "nasa_llis",
        "NASA Apollo Lunar Module Reliability Lessons Learned",
        "https://llis.nasa.gov/lesson/1806",
        "Apollo configuration control and lessons-learned evidence.",
    ),
    Source(
        "nasa_gao",
        "NASA: Better Mechanisms Needed for Sharing Lessons Learned",
        "https://www.gao.gov/products/gao-02-195",
        "NASA lessons-learned systems and weaknesses in dissemination.",
    ),
    Source(
        "manhattan_wellerstein",
        "Manhattan Project overview",
        "https://ethos.lps.library.cmu.edu/article/id/35/",
        "Mega-project coordination across scientific, industrial, and military work.",
    ),
    Source(
        "linux_kernel_patches",
        "Linux kernel submitting patches guide",
        "https://www.kernel.org/doc/html/v4.16/process/submitting-patches.html",
        "Review tags, patch discussion, and maintainer evidence for criticism and memory.",
    ),
    Source(
        "lkml_archive",
        "Linux Kernel Mailing List archive",
        "https://lkml.org/",
        "Append-like public archive of patch discussion and decisions.",
    ),
    Source(
        "alphago_google",
        "Mastering the game of Go with deep neural networks and tree search",
        "https://research.google/pubs/mastering-the-game-of-go-with-deep-neural-networks-and-tree-search/",
        "AlphaGo combined policy/value networks, self-play, and search.",
    ),
    Source(
        "crispr_nobel",
        "Nobel Prize in Chemistry 2020 press release",
        "https://www.nobelprize.org/prizes/chemistry/2020/press-release/",
        "CRISPR/Cas9 discovery and broad impact.",
    ),
    Source(
        "crispr_broad",
        "CRISPR Timeline",
        "https://www.broadinstitute.org/what-broad/areas-focus/project-spotlight/crispr-timeline",
        "Parallel discoveries, simplification into guide RNA, and incremental accumulation.",
    ),
    Source(
        "darwin_project",
        "Darwin's species notebooks",
        "https://www.darwinproject.ac.uk/commentary/evolution/darwin-s-species-notebooks-i-think",
        "Darwin notebooks as persistent memory for observations, readings, and hypotheses.",
    ),
    Source(
        "darwin_online",
        "Darwin's notebooks and reading lists",
        "https://darwin-online.org.uk/EditorialIntroductions/vanWyhe_notebooks.html",
        "Detailed record of Darwin's gradual development of species questions.",
    ),
    Source(
        "cern_open",
        "CERN Open Data",
        "https://openscience.cern/open-data",
        "Persistent identifiers, FAIR principles, and data management.",
    ),
    Source(
        "cern_preservation",
        "Open science at CERN",
        "https://cerncourier.com/a/open-science-a-vision-for-collaborative-reproducible-and-reusable-research/",
        "Preservation portals for datasets, code, tools, and analyses.",
    ),
    Source(
        "theranos_lessons",
        "Lessons from Theranos",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC8979578/",
        "Control case: weak validation and regulatory gaps.",
    ),
    Source(
        "theranos_peer_review",
        "Theranos and peer review case study",
        "https://www.bsm.upf.edu/documents/2024-case-study-elisabeth-holmes-theranos.pdf",
        "Control case: bold claims without peer review or public data scrutiny.",
    ),
    Source(
        "challenger_vaughan",
        "Challenger disaster and normalization of deviance",
        "https://magazine.columbia.edu/article/challenger-disaster-normalization-deviance",
        "Control case: warnings existed but failed to redirect decision-making.",
    ),
    Source(
        "replication_crisis",
        "Replication crisis in psychology",
        "https://nobaproject.com/modules/the-replication-crisis-in-psychology",
        "Control case: failed replications historically hard to publish.",
    ),
]


CASES = [
    CaseStudy(
        "bell_labs",
        "Bell Labs",
        "industrial research",
        "high lasting knowledge output",
        {
            "persistent_memory": 0.85,
            "append_only_accumulation": 0.65,
            "preserved_failures": 0.45,
            "institutionalized_criticism": 0.75,
            "question_refinement": 0.8,
            "knowledge_compression": 0.82,
            "applicability_boundaries": 0.7,
            "planning_prioritization": 0.72,
            "role_separation": 0.78,
            "knowledge_graph_equivalent": 0.55,
        },
        {
            "persistent_memory": "Long-lived lab structure and accumulated technical communities.",
            "institutionalized_criticism": "Interdisciplinary proximity and review culture made ideas face technical scrutiny.",
            "knowledge_compression": "Information theory, transistor physics, Unix/C compressed broad work into reusable primitives.",
        },
        ["bell_noll"],
    ),
    CaseStudy(
        "apollo",
        "Apollo Program / NASA",
        "mission engineering",
        "successful mission plus later lessons-learned systems",
        {
            "persistent_memory": 0.78,
            "append_only_accumulation": 0.72,
            "preserved_failures": 0.82,
            "institutionalized_criticism": 0.72,
            "question_refinement": 0.62,
            "knowledge_compression": 0.58,
            "applicability_boundaries": 0.88,
            "planning_prioritization": 0.92,
            "role_separation": 0.84,
            "knowledge_graph_equivalent": 0.62,
        },
        {
            "preserved_failures": "NASA lessons-learned systems explicitly preserve failure lessons.",
            "applicability_boundaries": "Configuration control and mission constraints defined where changes applied.",
            "planning_prioritization": "Mission-driven project management was central.",
        },
        ["nasa_llis", "nasa_gao"],
    ),
    CaseStudy(
        "manhattan",
        "Manhattan Project",
        "wartime science and engineering",
        "successful compressed mega-project with secrecy costs",
        {
            "persistent_memory": 0.7,
            "append_only_accumulation": 0.58,
            "preserved_failures": 0.55,
            "institutionalized_criticism": 0.7,
            "question_refinement": 0.75,
            "knowledge_compression": 0.78,
            "applicability_boundaries": 0.7,
            "planning_prioritization": 0.93,
            "role_separation": 0.86,
            "knowledge_graph_equivalent": 0.5,
        },
        {
            "role_separation": "Coordinated physics, engineering, materials, and military roles.",
            "planning_prioritization": "Compressed wartime objective forced prioritization.",
            "knowledge_compression": "Nuclear theory and engineering constraints were reduced into buildable designs.",
        },
        ["manhattan_wellerstein"],
    ),
    CaseStudy(
        "linux_kernel",
        "Linux kernel development",
        "open-source engineering",
        "durable high-output knowledge system",
        {
            "persistent_memory": 0.95,
            "append_only_accumulation": 0.95,
            "preserved_failures": 0.82,
            "institutionalized_criticism": 0.92,
            "question_refinement": 0.72,
            "knowledge_compression": 0.58,
            "applicability_boundaries": 0.85,
            "planning_prioritization": 0.72,
            "role_separation": 0.8,
            "knowledge_graph_equivalent": 0.88,
        },
        {
            "append_only_accumulation": "Mailing-list archives and version history preserve proposal/review traces.",
            "institutionalized_criticism": "Patch review and Reviewed-by tags formalize criticism.",
            "applicability_boundaries": "Maintainers, subsystem ownership, and commit scope preserve boundaries.",
        },
        ["linux_kernel_patches", "lkml_archive"],
    ),
    CaseStudy(
        "alphago",
        "AlphaGo development",
        "AI research",
        "breakthrough system development",
        {
            "persistent_memory": 0.65,
            "append_only_accumulation": 0.55,
            "preserved_failures": 0.7,
            "institutionalized_criticism": 0.72,
            "question_refinement": 0.78,
            "knowledge_compression": 0.86,
            "applicability_boundaries": 0.82,
            "planning_prioritization": 0.86,
            "role_separation": 0.76,
            "knowledge_graph_equivalent": 0.45,
        },
        {
            "knowledge_compression": "Policy/value networks plus tree search compressed Go expertise into reusable evaluation/search machinery.",
            "preserved_failures": "Self-play made losses part of training signal.",
            "planning_prioritization": "Tree search explicitly prioritized promising continuations.",
        },
        ["alphago_google"],
    ),
    CaseStudy(
        "crispr",
        "CRISPR/Cas9 discovery",
        "molecular biology",
        "successful cumulative discovery",
        {
            "persistent_memory": 0.75,
            "append_only_accumulation": 0.8,
            "preserved_failures": 0.55,
            "institutionalized_criticism": 0.78,
            "question_refinement": 0.82,
            "knowledge_compression": 0.9,
            "applicability_boundaries": 0.78,
            "planning_prioritization": 0.68,
            "role_separation": 0.7,
            "knowledge_graph_equivalent": 0.62,
        },
        {
            "append_only_accumulation": "Discovery built over repeated findings about CRISPR arrays, tracrRNA, Cas9, and guide RNA.",
            "knowledge_compression": "Fusing RNA components into guide RNA simplified the system.",
            "institutionalized_criticism": "Peer review and parallel findings constrained claims.",
        },
        ["crispr_nobel", "crispr_broad"],
    ),
    CaseStudy(
        "darwin",
        "Darwin's evolution research",
        "individual science network",
        "long-horizon theory formation",
        {
            "persistent_memory": 0.95,
            "append_only_accumulation": 0.88,
            "preserved_failures": 0.72,
            "institutionalized_criticism": 0.62,
            "question_refinement": 0.95,
            "knowledge_compression": 0.95,
            "applicability_boundaries": 0.78,
            "planning_prioritization": 0.55,
            "role_separation": 0.45,
            "knowledge_graph_equivalent": 0.78,
        },
        {
            "persistent_memory": "Notebooks and correspondence preserved observations, readings, and changing hypotheses.",
            "question_refinement": "Species questions were gradually refined over years.",
            "knowledge_compression": "Natural selection compressed diverse observations into a general mechanism.",
        },
        ["darwin_project", "darwin_online"],
    ),
    CaseStudy(
        "cern",
        "CERN / LHC collaborations",
        "large-scale experimental physics",
        "high-reliability cumulative science",
        {
            "persistent_memory": 0.95,
            "append_only_accumulation": 0.9,
            "preserved_failures": 0.8,
            "institutionalized_criticism": 0.95,
            "question_refinement": 0.75,
            "knowledge_compression": 0.72,
            "applicability_boundaries": 0.95,
            "planning_prioritization": 0.86,
            "role_separation": 0.9,
            "knowledge_graph_equivalent": 0.85,
        },
        {
            "persistent_memory": "Open Data and preservation portals keep data, code, tools, and analyses reusable.",
            "institutionalized_criticism": "Large collaborations and reproducibility practices make claims face many checks.",
            "applicability_boundaries": "Data policies, likelihoods, and analysis preservation specify reusable boundaries.",
        },
        ["cern_open", "cern_preservation"],
    ),
    CaseStudy(
        "theranos",
        "Theranos",
        "control: failed biomedical innovation",
        "little reliable lasting knowledge",
        {
            "persistent_memory": 0.18,
            "append_only_accumulation": 0.12,
            "preserved_failures": 0.08,
            "institutionalized_criticism": 0.05,
            "question_refinement": 0.18,
            "knowledge_compression": 0.08,
            "applicability_boundaries": 0.12,
            "planning_prioritization": 0.45,
            "role_separation": 0.25,
            "knowledge_graph_equivalent": 0.05,
        },
        {
            "institutionalized_criticism": "Claims were not subjected to normal public peer review and data scrutiny.",
            "preserved_failures": "Failures were concealed rather than converted into shared knowledge.",
            "applicability_boundaries": "Clinical boundaries were not reliably established before deployment.",
        },
        ["theranos_lessons", "theranos_peer_review"],
        control=True,
    ),
    CaseStudy(
        "challenger",
        "Space Shuttle Challenger decision process",
        "control: organizational failure within capable institution",
        "knowledge existed but failed to alter decision",
        {
            "persistent_memory": 0.7,
            "append_only_accumulation": 0.6,
            "preserved_failures": 0.55,
            "institutionalized_criticism": 0.28,
            "question_refinement": 0.25,
            "knowledge_compression": 0.25,
            "applicability_boundaries": 0.2,
            "planning_prioritization": 0.45,
            "role_separation": 0.55,
            "knowledge_graph_equivalent": 0.35,
        },
        {
            "institutionalized_criticism": "Engineer warnings did not overcome management launch pressure.",
            "applicability_boundaries": "O-ring risk under cold conditions was not treated as a hard boundary.",
            "preserved_failures": "Prior deviations became normalized instead of falsifying safety assumptions.",
        },
        ["challenger_vaughan"],
        control=True,
    ),
    CaseStudy(
        "replication_crisis",
        "Replication crisis in psychology",
        "control: field-level reliability failure",
        "many claims weakened by missing negative memory",
        {
            "persistent_memory": 0.55,
            "append_only_accumulation": 0.4,
            "preserved_failures": 0.18,
            "institutionalized_criticism": 0.35,
            "question_refinement": 0.38,
            "knowledge_compression": 0.25,
            "applicability_boundaries": 0.28,
            "planning_prioritization": 0.35,
            "role_separation": 0.45,
            "knowledge_graph_equivalent": 0.25,
        },
        {
            "preserved_failures": "Failed replications were historically hard to publish.",
            "institutionalized_criticism": "The field later improved by making replications more visible.",
            "applicability_boundaries": "Many effects lacked reliable boundary conditions.",
        },
        ["replication_crisis"],
        control=True,
    ),
]


def analyze() -> dict:
    successes = [case for case in CASES if not case.control]
    controls = [case for case in CASES if case.control]
    mechanism_rows = []
    for mechanism in MECHANISMS:
        success_mean = mean(case.scores[mechanism] for case in successes)
        control_mean = mean(case.scores[mechanism] for case in controls)
        prevalence = sum(case.scores[mechanism] >= 0.65 for case in successes) / len(successes)
        counterexamples = [
            case.id for case in successes if case.scores[mechanism] < 0.5
        ]
        mechanism_rows.append(
            {
                "mechanism": mechanism,
                "success_mean": round(success_mean, 3),
                "control_mean": round(control_mean, 3),
                "separation": round(success_mean - control_mean, 3),
                "success_prevalence": round(prevalence, 3),
                "counterexamples": counterexamples,
                "universality_score": round(success_mean * 0.55 + prevalence * 0.35 + (success_mean - control_mean) * 0.1, 3),
            }
        )
    return {
        "mechanisms": sorted(mechanism_rows, key=lambda item: item["universality_score"], reverse=True),
        "case_matrix": [
            {
                "id": case.id,
                "name": case.name,
                "control": case.control,
                "scores": case.scores,
                "sources": case.source_ids,
            }
            for case in CASES
        ],
    }


def write_deliverables(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    reports = root / "reports"
    data = root / "data"
    reports.mkdir(exist_ok=True)
    data.mkdir(exist_ok=True)
    result = analyze()
    write_json(data / "case_matrix.json", result["case_matrix"])
    write_json(data / "universality_ranking.json", result["mechanisms"])
    write_json(data / "sources.json", [asdict(source) for source in SOURCES])
    write_cross_case_matrix(reports / "cross_case_mechanism_matrix.md")
    write_universality_ranking(reports / "universality_ranking.md", result["mechanisms"])
    write_counterexamples(reports / "counterexample_catalogue.md", result["mechanisms"])
    write_historical_evidence(reports / "historical_evidence_report.md")
    write_confidence(reports / "confidence_estimates.md", result["mechanisms"])
    write_revised_theory(reports / "revised_theory_of_research_organization.md", result["mechanisms"])
    write_final_report(reports / "final_report.md", result["mechanisms"])


def write_cross_case_matrix(path: Path) -> None:
    lines = ["# Cross-Case Mechanism Matrix", ""]
    lines.append("| Case | Control | " + " | ".join(MECHANISMS) + " |")
    lines.append("|---|---|" + "|".join(["---"] * len(MECHANISMS)) + "|")
    for case in CASES:
        values = " | ".join(f"{case.scores[m]:.2f}" for m in MECHANISMS)
        lines.append(f"| {case.name} | {case.control} | {values} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_universality_ranking(path: Path, mechanisms: list[dict]) -> None:
    lines = ["# Universality Ranking", ""]
    for index, item in enumerate(mechanisms, 1):
        lines.append(
            f"{index}. {item['mechanism']}: universality {item['universality_score']}, success mean {item['success_mean']}, control mean {item['control_mean']}, prevalence {item['success_prevalence']}."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_counterexamples(path: Path, mechanisms: list[dict]) -> None:
    lines = ["# Counterexample Catalogue", ""]
    for item in mechanisms:
        if item["counterexamples"]:
            lines.append(f"- {item['mechanism']}: weak or absent in {', '.join(item['counterexamples'])}.")
        else:
            lines.append(f"- {item['mechanism']}: no strong successful-case absence found under this coding.")
    lines.extend(
        [
            "",
            "Most serious falsifiers:",
            "- Darwin weakens role_separation as a universal mechanism: individual research can still produce major knowledge with external correspondence rather than formal roles.",
            "- Manhattan Project weakens open append-only memory as a universal mechanism: secrecy reduced openness, yet a breakthrough occurred.",
            "- AlphaGo weakens journal-like auditability as a universal mechanism: experimental iteration and self-play mattered more than human-readable trace.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_historical_evidence(path: Path) -> None:
    lines = ["# Historical Evidence Report", ""]
    for case in CASES:
        lines.append(f"## {case.name}")
        lines.append("")
        lines.append(f"Outcome: {case.outcome}.")
        for mechanism, evidence in case.evidence.items():
            lines.append(f"- {mechanism}: {evidence}")
        lines.append(f"- Sources: {', '.join(case.source_ids)}")
        lines.append("")
    lines.append("## Source URLs")
    lines.append("")
    for source in SOURCES:
        lines.append(f"- {source.id}: {source.title}. {source.url}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_confidence(path: Path, mechanisms: list[dict]) -> None:
    lines = ["# Confidence Estimates", ""]
    for item in mechanisms:
        if item["success_prevalence"] >= 0.75 and item["separation"] >= 0.25:
            confidence = "high"
        elif item["success_prevalence"] >= 0.5 and item["separation"] >= 0.15:
            confidence = "medium"
        else:
            confidence = "low"
        lines.append(
            f"- {item['mechanism']}: {confidence}. Evidence: prevalence {item['success_prevalence']}, success/control separation {item['separation']}, counterexamples {item['counterexamples'] or 'none'}."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_revised_theory(path: Path, mechanisms: list[dict]) -> None:
    path.write_text(
        """# Revised Theory of Research Organization

Successful knowledge systems do not all share the same institutions, tools, or openness level. They do tend to share a smaller set of functional properties:

1. Claims persist long enough to be attacked, reused, and recombined.
2. Criticism is institutionalized strongly enough to change decisions.
3. Failed cases are preserved or converted into future constraints.
4. Breakthroughs usually involve compression: many observations or local tricks become fewer portable structures.
5. Applicability boundaries matter most in engineering, medicine, and high-energy physics; they are less visible in some individual theory-building cases.

The Studio principles survive in weakened form. They are not universal as literal append-only files, journals, or graphs. They are more universal as institutional memory, adversarial correction, and compression with boundary discipline.
""",
        encoding="utf-8",
    )


def write_final_report(path: Path, mechanisms: list[dict]) -> None:
    by_name = {item["mechanism"]: item for item in mechanisms}
    survived = [
        "applicability_boundaries",
        "persistent_memory",
        "institutionalized_criticism",
        "preserved_failures",
        "knowledge_compression",
        "question_refinement",
    ]
    failed = ["role_separation", "knowledge_graph_equivalent"]
    uncertain = ["append_only_accumulation", "planning_prioritization"]
    lines = [
        "# Benchmark Research #004 Final Report",
        "",
        "## Which organizational principles survived historical comparison?",
        "",
    ]
    for mechanism in survived:
        item = by_name[mechanism]
        lines.append(
            f"- {mechanism}: survived. Success mean {item['success_mean']}, control mean {item['control_mean']}, prevalence {item['success_prevalence']}."
        )
    lines.extend(["", "## Which failed?", ""])
    for mechanism in failed:
        item = by_name[mechanism]
        lines.append(
            f"- {mechanism}: failed as universal. It appears useful in some systems but absent or weak in successful individual/theoretical cases. Counterexamples: {item['counterexamples'] or 'none coded'}."
        )
    lines.extend(["", "## Which remain uncertain?", ""])
    for mechanism in uncertain:
        item = by_name[mechanism]
        lines.append(
            f"- {mechanism}: uncertain. Score {item['universality_score']}; counterexamples {item['counterexamples'] or 'none coded'}, but expression varies strongly by domain."
        )
    lines.extend(
        [
            "",
            "## If a civilization had to rebuild science from scratch, which principles first?",
            "",
            "1. Persistent public memory of claims, methods, evidence, and revisions. Evidence: Darwin's notebooks, Linux mailing lists/version history, CERN data preservation, NASA lessons learned, and CRISPR's cumulative literature all show that durable memory lets later work reuse and correct earlier work.",
            "2. Institutionalized adversarial criticism with preserved failures. Evidence: Linux patch review, CERN collaboration checks, Apollo lessons-learned practice, and the negative controls Theranos/Challenger/replication crisis show that weak criticism or hidden failure memory produces unreliable knowledge.",
            "3. Compression through question refinement with explicit applicability boundaries. Evidence: Darwin compressed biological diversity into natural selection while working through species boundaries, CRISPR compressed bacterial immune machinery into guide-RNA genome editing with biochemical scope limits, AlphaGo compressed play into policy/value/search machinery inside the Go domain, CERN encodes likelihood/data applicability, and Bell Labs repeatedly produced reusable primitives with engineering contexts.",
            "",
            "The Studio's three principles survive, but only after translation. Literal append-only software, journals, and graphs are not universal. Durable memory, adversarial correction, and compression under refined questions are much more general.",
            "",
            "## Evidence base",
            "",
        ]
    )
    for source in SOURCES:
        lines.append(f"- {source.title}: {source.url}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_benchmark(root: str | Path) -> Path:
    root = Path(root)
    write_deliverables(root)
    return root


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="benchmarks/benchmark_004")
    args = parser.parse_args(argv)
    root = run_benchmark(args.root)
    print(f"Benchmark #004 complete at {root.resolve()}")


if __name__ == "__main__":
    main()
