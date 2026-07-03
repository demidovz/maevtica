from __future__ import annotations

import argparse
from pathlib import Path

from .cycle import initialize_store, run_cycles
from .dashboard import write_dashboard
from .journal import write_missing_journals
from .plugins import PLUGINS, load_plugin
from .storage import ResearchStore


def cmd_init(args: argparse.Namespace) -> None:
    store = ResearchStore(args.root)
    seeds = []
    domains = sorted(PLUGINS) if args.domain == "all" else [args.domain]
    for domain in domains:
        plugin = load_plugin(domain)
        seeds.append(plugin.seed())
    artifacts = [artifact for seed in seeds for artifact in seed.artifacts]
    edges = [edge for seed in seeds for edge in seed.edges]
    initialize_store(store, (artifacts, edges))
    print(f"Initialized Research State at {Path(args.root).resolve()}")


def cmd_run(args: argparse.Namespace) -> None:
    store = ResearchStore(args.root)
    initialize_store(store)
    cycles = run_cycles(store, args.cycles)
    last = cycles[-1] if cycles else None
    if last:
        print(
            f"Ran {len(cycles)} cycles. Last cycle={last.cycle} "
            f"compression={last.metrics['compression']} next_frontier={last.next_frontier_id}"
        )


def cmd_dashboard(args: argparse.Namespace) -> None:
    store = ResearchStore(args.root)
    state = store.load()
    out = write_dashboard(state, args.out, journal_root=args.root)
    print(f"Wrote dashboard to {out.resolve()}")


def cmd_journal(args: argparse.Namespace) -> None:
    store = ResearchStore(args.root)
    state = store.load()
    written = write_missing_journals(args.root, state)
    print(f"Wrote {len(written)} missing journal entries under {Path(args.root).resolve() / 'research_journal'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="epistemic-studio")
    sub = parser.add_subparsers(required=True)

    init = sub.add_parser("init", help="Initialize a Research State")
    init.add_argument("--root", default="studio_state")
    init.add_argument("--domain", default="all")
    init.set_defaults(func=cmd_init)

    run = sub.add_parser("run", help="Run research cycles")
    run.add_argument("--root", default="studio_state")
    run.add_argument("--cycles", type=int, default=1)
    run.set_defaults(func=cmd_run)

    dashboard = sub.add_parser("dashboard", help="Render static dashboard HTML")
    dashboard.add_argument("--root", default="studio_state")
    dashboard.add_argument("--out", default="studio_state/dashboard.html")
    dashboard.set_defaults(func=cmd_dashboard)

    journal = sub.add_parser("journal", help="Write missing immutable research journal entries")
    journal.add_argument("--root", default="studio_state")
    journal.set_defaults(func=cmd_journal)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
