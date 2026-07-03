from __future__ import annotations

from dataclasses import dataclass

from .models import Artifact, ArtifactKind, EdgeKind, GraphEdge, Provenance


@dataclass(frozen=True)
class DomainSeed:
    artifacts: list[Artifact]
    edges: list[GraphEdge]


class DomainPlugin:
    name = "base"

    def seed(self, agent_id: str = "plugin-loader") -> DomainSeed:
        raise NotImplementedError


def _prov(agent_id: str, domain: str) -> Provenance:
    return Provenance(agent_id=agent_id, agent_role="plugin", cycle=0, source=domain)


class PossibleWorldsPlugin(DomainPlugin):
    name = "possible_worlds"

    def seed(self, agent_id: str = "plugin-loader") -> DomainSeed:
        provenance = _prov(agent_id, self.name)
        question = Artifact(
            kind=ArtifactKind.RESEARCH_QUESTION,
            title="How do possible-world models reduce explanatory duplication?",
            body="Study where multiple hypotheses become explainable by fewer world-model distinctions.",
            provenance=provenance,
            confidence=0.55,
            metadata={"domain": self.name},
        )
        unknown = Artifact(
            kind=ArtifactKind.UNKNOWN_REGION,
            title="Boundary between model simplification and lost distinctions",
            body="Identify regions where compression removes distinctions needed by counterexamples.",
            provenance=provenance,
            confidence=0.45,
            metadata={"domain": self.name, "priority": 0.8},
        )
        assumption = Artifact(
            kind=ArtifactKind.ASSUMPTION,
            title="Compression must preserve falsification pressure",
            body="A simpler map is useful only if it keeps counterexamples visible.",
            provenance=provenance,
            confidence=0.7,
            metadata={"domain": self.name},
        )
        edge = GraphEdge(
            source_id=assumption.id,
            target_id=question.id,
            kind=EdgeKind.DEPENDS_ON,
            provenance=provenance,
            confidence=0.65,
        )
        return DomainSeed(artifacts=[question, unknown, assumption], edges=[edge])


class GenericDomainPlugin(DomainPlugin):
    question: str = ""
    unknown: str = ""
    assumption: str = ""

    def seed(self, agent_id: str = "plugin-loader") -> DomainSeed:
        provenance = _prov(agent_id, self.name)
        question = Artifact(
            kind=ArtifactKind.RESEARCH_QUESTION,
            title=self.question,
            body=f"Domain seed for {self.name}; contributes only domain knowledge.",
            provenance=provenance,
            confidence=0.5,
            metadata={"domain": self.name},
        )
        unknown = Artifact(
            kind=ArtifactKind.UNKNOWN_REGION,
            title=self.unknown,
            body="Unexplored region competing for Studio attention.",
            provenance=provenance,
            confidence=0.45,
            metadata={"domain": self.name, "priority": 0.55},
        )
        assumption = Artifact(
            kind=ArtifactKind.ASSUMPTION,
            title=self.assumption,
            body="Initial domain assumption; it is disposable if later artifacts defeat it.",
            provenance=provenance,
            confidence=0.55,
            metadata={"domain": self.name},
        )
        edge = GraphEdge(assumption.id, question.id, EdgeKind.DEPENDS_ON, provenance, confidence=0.5)
        return DomainSeed(artifacts=[question, unknown, assumption], edges=[edge])


class AgiPlugin(GenericDomainPlugin):
    name = "agi"
    question = "Which AGI claims survive explicit falsification pressure?"
    unknown = "Boundary between benchmark competence and generalization"
    assumption = "Generalization claims require counterexample-rich evaluation"


class MaevticaPlugin(GenericDomainPlugin):
    name = "maevtica"
    question = "Which Maevtica research structures compress across representations?"
    unknown = "Fragile invariance under representation changes"
    assumption = "Cross-paradigm claims must survive ablation"


class BiologyPlugin(GenericDomainPlugin):
    name = "biology"
    question = "Which biological explanations reduce many observations to fewer mechanisms?"
    unknown = "Boundary between adaptive explanation and historical contingency"
    assumption = "Mechanistic reuse should be tested against exceptions"


class EconomicsPlugin(GenericDomainPlugin):
    name = "economics"
    question = "Which economic models remain useful under incentive shifts?"
    unknown = "Regime changes that invalidate stable-looking models"
    assumption = "Prediction markets need explicit failure accounting"


PLUGINS: dict[str, type[DomainPlugin]] = {
    PossibleWorldsPlugin.name: PossibleWorldsPlugin,
    AgiPlugin.name: AgiPlugin,
    MaevticaPlugin.name: MaevticaPlugin,
    BiologyPlugin.name: BiologyPlugin,
    EconomicsPlugin.name: EconomicsPlugin,
}


def load_plugin(name: str) -> DomainPlugin:
    try:
        return PLUGINS[name]()
    except KeyError as exc:
        available = ", ".join(sorted(PLUGINS))
        raise ValueError(f"Unknown domain plugin {name!r}. Available: {available}") from exc
