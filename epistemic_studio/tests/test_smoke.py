from tempfile import TemporaryDirectory
from unittest import TestCase

from epistemic_studio.cycle import initialize_store, run_cycles
from epistemic_studio.dashboard import write_dashboard
from epistemic_studio.journal import cycle_journal_path, write_missing_journals
from epistemic_studio.models import ArtifactKind
from epistemic_studio.plugins import load_plugin
from epistemic_studio.storage import ResearchStore


class StudioSmokeTest(TestCase):
    def test_research_state_cycle_and_dashboard(self) -> None:
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = temp.name

        store = ResearchStore(root)
        seed = load_plugin("possible_worlds").seed()
        initialize_store(store, (seed.artifacts, seed.edges))
        cycles = run_cycles(store, 10)
        state = store.load()
        write_missing_journals(root, state)
        out = write_dashboard(state, f"{root}/dashboard.html", journal_root=root)

        self.assertEqual(len(cycles), 10)
        self.assertTrue(store.events_path.exists())
        self.assertTrue(out.exists())
        self.assertTrue(cycle_journal_path(root, 1).exists())
        self.assertTrue(cycle_journal_path(root, 10).exists())
        self.assertTrue((store.root / "research_journal" / "milestones" / "milestone_0010.md").exists())
        self.assertGreaterEqual(len(state.agents), 7)
        self.assertTrue(state.bids)
        self.assertTrue(state.allocations)
        self.assertTrue(state.domains)
        self.assertTrue(state.artifacts_by_kind(ArtifactKind.META_MEMORY))
        self.assertTrue(state.artifacts_by_kind(ArtifactKind.RESEARCH_QUESTION))
        self.assertTrue(state.artifacts_by_kind(ArtifactKind.CONTRADICTION))
        self.assertGreater(max(agent.reputation for agent in state.agents.values()), 1.0)
        journal = cycle_journal_path(root, 10).read_text(encoding="utf-8")
        expected_sections = [
            "# Header",
            "# Planner Decision",
            "# Attention Allocation",
            "# Explorer Report",
            "# Lab Report",
            "# Engine Report",
            "# Historian Report",
            "# Cartographer Report",
            "# Meta Observer",
            "# Organizational Learning",
            "# Metrics",
            "# Biggest Insight",
            "# Biggest Mistake",
            "# Next Experiment",
            "# Human Summary",
        ]
        self.assertEqual([line for line in journal.splitlines() if line.startswith("# ")], expected_sections)
        self.assertIn("Today's Studio Summary", journal)
