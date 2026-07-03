from __future__ import annotations

import html
import json
from pathlib import Path

from .metrics import choose_frontier, compute_metrics
from .models import ArtifactKind, ResearchState


def render_dashboard(state: ResearchState, journal_root: str | Path | None = None) -> str:
    metrics = compute_metrics(state)
    frontier = state.artifacts.get(choose_frontier(state) or "")
    graph_nodes = [
        {"id": artifact.id, "label": artifact.title, "kind": artifact.kind, "status": artifact.status}
        for artifact in state.artifacts.values()
    ]
    graph_edges = [
        {"source": edge.source_id, "target": edge.target_id, "kind": edge.kind}
        for edge in state.edges.values()
    ]
    dead = [
        artifact
        for artifact in state.artifacts_by_kind(ArtifactKind.HYPOTHESIS)
        if artifact.status in {"dead", "falsified"}
    ]
    contradictions = state.active_artifacts_by_kind(ArtifactKind.CONTRADICTION)
    open_frontiers = [
        artifact
        for artifact in state.artifacts.values()
        if artifact.kind in {ArtifactKind.UNKNOWN_REGION, ArtifactKind.OPEN_QUESTION, ArtifactKind.FRONTIER}
        and artifact.status == "active"
    ]
    cycle_points = [
        {
            "cycle": cycle.cycle,
            "compression": cycle.metrics.get("compression", 0),
            "epistemic_value": cycle.metrics.get("epistemic_value", 0),
        }
        for cycle in state.cycles
    ]
    journal_entries = load_journal_entries(journal_root) if journal_root else []
    cycle_lookup = {
        cycle.cycle: {
            "cycle": cycle.cycle,
            "timestamp": cycle.completed_at,
            "frontier": state.artifacts.get(cycle.frontier_id or "").title
            if cycle.frontier_id in state.artifacts
            else "none",
            "next_frontier": state.artifacts.get(cycle.next_frontier_id or "").title
            if cycle.next_frontier_id in state.artifacts
            else "none",
            "metrics": cycle.metrics,
            "agents": [
                {
                    "id": agent.id,
                    "role": agent.role,
                    "reputation": agent.reputation,
                    "generation": agent.generation,
                    "status": agent.status,
                }
                for agent in state.agents.values()
            ],
            "allocations": [
                state.allocations[item].to_dict()
                for item in cycle.allocation_ids
                if item in state.allocations
            ],
            "nodes": [
                {"id": artifact.id, "label": artifact.title, "kind": artifact.kind, "status": artifact.status}
                for artifact in state.artifacts.values()
                if artifact.provenance.cycle <= cycle.cycle
            ],
            "edges": [
                {"source": edge.source_id, "target": edge.target_id, "kind": edge.kind}
                for edge in state.edges.values()
                if edge.provenance.cycle <= cycle.cycle
            ],
        }
        for cycle in state.cycles
    }
    data = {
        "nodes": graph_nodes,
        "edges": graph_edges,
        "cycle_points": cycle_points,
        "journal_entries": journal_entries,
        "cycles": cycle_lookup,
    }
    next_title = frontier.title if frontier else "Seed or select a domain frontier"
    next_body = frontier.body if frontier else "No active frontier exists yet."
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Epistemic Studio Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172026;
      --muted: #66737d;
      --line: #d8dee3;
      --panel: #f7f9fa;
      --accent: #0d6b57;
      --danger: #a83232;
      --warn: #9c6414;
      --blue: #285f9c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: #ffffff;
    }}
    header {{
      padding: 20px 28px 16px;
      border-bottom: 1px solid var(--line);
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 24px;
      align-items: end;
    }}
    h1, h2, h3 {{ margin: 0; letter-spacing: 0; }}
    h1 {{ font-size: 24px; font-weight: 720; }}
    h2 {{ font-size: 15px; margin-bottom: 10px; }}
    h3 {{ font-size: 13px; margin-bottom: 6px; }}
    .subtle {{ color: var(--muted); }}
    .shell {{ padding: 22px 28px 28px; display: grid; gap: 18px; }}
    .next {{
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(280px, .8fr);
      gap: 18px;
      align-items: stretch;
    }}
    section {{ min-width: 0; }}
    .band {{
      border-top: 1px solid var(--line);
      padding-top: 16px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      min-width: 0;
    }}
    .kpis {{
      display: grid;
      grid-template-columns: repeat(6, minmax(120px, 1fr));
      gap: 10px;
    }}
    .kpi {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fff;
      min-height: 78px;
    }}
    .kpi strong {{ display: block; font-size: 22px; margin-top: 6px; }}
    .grid {{
      display: grid;
      grid-template-columns: 1.1fr .9fr;
      gap: 18px;
    }}
    .columns {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }}
    ul {{ margin: 0; padding-left: 18px; }}
    li {{ margin: 5px 0; }}
    .graph {{
      height: 420px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: linear-gradient(#fff, #f8fafb);
      overflow: hidden;
    }}
    svg text {{ font-size: 10px; fill: var(--ink); }}
    .node {{ fill: #ffffff; stroke: var(--blue); stroke-width: 1.5; }}
    .node.contradiction {{ stroke: var(--danger); }}
    .node.frontier, .node.unknown_region, .node.open_question {{ stroke: var(--warn); }}
    .edge {{ stroke: #8a969f; stroke-width: 1; opacity: .75; }}
    .timeline {{
      display: flex;
      align-items: end;
      gap: 4px;
      height: 140px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fff;
    }}
    .bar {{ flex: 1; min-width: 8px; background: var(--accent); border-radius: 3px 3px 0 0; }}
    .journal-layout {{
      display: grid;
      grid-template-columns: 220px minmax(0, 1fr);
      gap: 12px;
    }}
    .cycle-list {{
      display: grid;
      gap: 6px;
      align-content: start;
      max-height: 540px;
      overflow: auto;
    }}
    .cycle-button {{
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 6px;
      padding: 8px 10px;
      text-align: left;
      color: var(--ink);
      cursor: pointer;
    }}
    .cycle-button.active {{ border-color: var(--accent); outline: 2px solid rgba(13, 107, 87, .16); }}
    .journal-text {{
      white-space: pre-wrap;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      line-height: 1.5;
      max-height: 540px;
      overflow: auto;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }}
    @media (max-width: 980px) {{
      header, .next, .grid, .columns, .journal-layout {{ grid-template-columns: 1fr; }}
      .kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Epistemic Studio</h1>
      <div class="subtle">Persistent Research State, artifact-only agent communication</div>
    </div>
    <div class="subtle">{len(state.cycles)} cycles · {len(state.artifacts)} artifacts · {len(state.edges)} edges</div>
  </header>
  <main class="shell">
    <section class="next">
      <div class="panel">
        <h2>What should the laboratory investigate next?</h2>
        <h1>{html.escape(next_title)}</h1>
        <p>{html.escape(next_body)}</p>
      </div>
      <div class="panel">
        <h2>Current Research State</h2>
        <div class="kpis">
          {kpi("Compression", metrics["compression"])}
          {kpi("Epistemic Value", metrics["epistemic_value"])}
          {kpi("Contradictions", metrics["contradictions"])}
          {kpi("Counterexample Density", metrics["counterexample_density"])}
          {kpi("Attention Efficiency", metrics["attention_efficiency"])}
          {kpi("Stagnation", metrics["stagnation_score"])}
        </div>
      </div>
    </section>
    <section class="grid band">
      <div>
        <h2>Knowledge Graph</h2>
        <div id="graph" class="graph"></div>
      </div>
      <div>
        <h2>Simplification And Value</h2>
        <div id="timeline" class="timeline"></div>
      </div>
    </section>
    <section class="columns band">
      {artifact_list("Open Frontiers", open_frontiers)}
      {artifact_list("Contradictions", contradictions)}
      {artifact_list("Dead Hypotheses", dead)}
    </section>
    <section class="columns band">
      {concept_list(metrics["most_reusable_concepts"])}
      {agent_list(state)}
      {atlas_list(state)}
    </section>
    <section class="columns band">
      {market_list(state)}
      {domain_list(metrics["active_domains"])}
      {organization_list(state)}
    </section>
    <section class="band">
      <h2>Research Journal</h2>
      <div class="journal-layout">
        <div id="cycle-list" class="cycle-list"></div>
        <div>
          <div class="panel" id="cycle-reconstruction">
            <h2>Cycle Reconstruction</h2>
            <p class="subtle">Select a cycle to inspect its Research State, Knowledge Graph, Agent Activity, Planner Decision, Attention Economy, and Compression Metrics.</p>
          </div>
          <pre id="journal-text" class="journal-text">No journal entries found. Run: python -m epistemic_studio.cli journal --root studio_state</pre>
        </div>
      </div>
    </section>
  </main>
  <script id="studio-data" type="application/json">{html.escape(json.dumps(data))}</script>
  <script>
    const data = JSON.parse(document.getElementById("studio-data").textContent);
    const graph = document.getElementById("graph");
    const width = graph.clientWidth || 800;
    const height = graph.clientHeight || 420;
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);
    graph.appendChild(svg);
    const nodes = data.nodes.slice(-80);
    const ids = new Map(nodes.map((n, i) => [n.id, i]));
    const radius = Math.min(width, height) * 0.38;
    const cx = width / 2;
    const cy = height / 2;
    nodes.forEach((node, i) => {{
      const angle = (Math.PI * 2 * i) / Math.max(1, nodes.length);
      node.x = cx + Math.cos(angle) * radius;
      node.y = cy + Math.sin(angle) * radius;
    }});
    data.edges.forEach(edge => {{
      if (!ids.has(edge.source) || !ids.has(edge.target)) return;
      const a = nodes[ids.get(edge.source)];
      const b = nodes[ids.get(edge.target)];
      const line = document.createElementNS(svg.namespaceURI, "line");
      line.setAttribute("class", "edge");
      line.setAttribute("x1", a.x);
      line.setAttribute("y1", a.y);
      line.setAttribute("x2", b.x);
      line.setAttribute("y2", b.y);
      svg.appendChild(line);
    }});
    nodes.forEach(node => {{
      const circle = document.createElementNS(svg.namespaceURI, "circle");
      circle.setAttribute("class", `node ${{node.kind}}`);
      circle.setAttribute("cx", node.x);
      circle.setAttribute("cy", node.y);
      circle.setAttribute("r", 5);
      svg.appendChild(circle);
      const label = document.createElementNS(svg.namespaceURI, "text");
      label.setAttribute("x", node.x + 7);
      label.setAttribute("y", node.y + 3);
      label.textContent = node.label.slice(0, 32);
      svg.appendChild(label);
    }});
    const timeline = document.getElementById("timeline");
    const maxValue = Math.max(1, ...data.cycle_points.map(p => p.epistemic_value));
    data.cycle_points.forEach(point => {{
      const bar = document.createElement("div");
      bar.className = "bar";
      bar.title = `Cycle ${{point.cycle}} · value ${{point.epistemic_value}} · compression ${{point.compression}}`;
      bar.style.height = `${{Math.max(6, (point.epistemic_value / maxValue) * 118)}}px`;
      timeline.appendChild(bar);
    }});
    const cycleList = document.getElementById("cycle-list");
    const journalText = document.getElementById("journal-text");
    const reconstruction = document.getElementById("cycle-reconstruction");
    function renderCycle(cycleNumber) {{
      const selected = data.journal_entries.find(entry => entry.cycle === cycleNumber);
      const cycle = data.cycles[String(cycleNumber)] || data.cycles[cycleNumber];
      document.querySelectorAll(".cycle-button").forEach(button => {{
        button.classList.toggle("active", Number(button.dataset.cycle) === cycleNumber);
      }});
      journalText.textContent = selected ? selected.markdown : "Journal entry missing for this cycle.";
      if (!cycle) return;
      const allocation = cycle.allocations.map(item => `${{item.role}}:${{item.allocated}}`).join(", ") || "none";
      reconstruction.innerHTML = `
        <h2>Cycle ${{cycle.cycle}} Reconstruction</h2>
        <ul>
          <li>Frontier: ${{cycle.frontier}}</li>
          <li>Next frontier: ${{cycle.next_frontier}}</li>
          <li>Graph at cycle: ${{cycle.nodes.length}} nodes, ${{cycle.edges.length}} edges</li>
          <li>Agent records visible: ${{cycle.agents.length}}</li>
          <li>Attention economy: ${{allocation}}</li>
          <li>Compression: ${{cycle.metrics.compression ?? 0}}</li>
          <li>Attention efficiency: ${{cycle.metrics.attention_efficiency ?? 0}}</li>
        </ul>`;
    }}
    data.journal_entries.forEach(entry => {{
      const button = document.createElement("button");
      button.className = "cycle-button";
      button.dataset.cycle = entry.cycle;
      button.textContent = `Cycle ${{entry.cycle}}`;
      button.addEventListener("click", () => renderCycle(entry.cycle));
      cycleList.appendChild(button);
    }});
    if (data.journal_entries.length) {{
      renderCycle(data.journal_entries[data.journal_entries.length - 1].cycle);
    }}
  </script>
</body>
</html>"""


def kpi(label: str, value: object) -> str:
    return f'<div class="kpi"><span class="subtle">{html.escape(label)}</span><strong>{html.escape(str(value))}</strong></div>'


def artifact_list(title: str, artifacts: list) -> str:
    items = "".join(f"<li>{html.escape(item.title)}</li>" for item in artifacts[:8]) or "<li class='subtle'>None recorded</li>"
    return f'<div class="panel"><h2>{html.escape(title)}</h2><ul>{items}</ul></div>'


def concept_list(concepts: list[dict]) -> str:
    items = "".join(
        f"<li>{html.escape(item['title'])} <span class='subtle'>reuse {item['reuse']}</span></li>"
        for item in concepts
    ) or "<li class='subtle'>None recorded</li>"
    return f'<div class="panel"><h2>Most Reusable Concepts</h2><ul>{items}</ul></div>'


def agent_list(state: ResearchState) -> str:
    agents = sorted(state.agents.values(), key=lambda item: item.reputation, reverse=True)
    items = "".join(
        f"<li>{html.escape(agent.role)} <span class='subtle'>rep {agent.reputation}, gen {agent.generation}, attention {agent.attention_balance}</span></li>"
        for agent in agents
    )
    return f'<div class="panel"><h2>Agent Activity</h2><ul>{items}</ul></div>'


def atlas_list(state: ResearchState) -> str:
    atlases = state.artifacts_by_kind(ArtifactKind.APPLICABILITY_ATLAS)
    items = "".join(f"<li>{html.escape(item.title)}</li>" for item in atlases[-6:]) or "<li class='subtle'>None recorded</li>"
    return f'<div class="panel"><h2>Applicability Atlas</h2><ul>{items}</ul></div>'


def market_list(state: ResearchState) -> str:
    bids = sorted(state.bids.values(), key=lambda item: item.created_at, reverse=True)[:8]
    allocations = {allocation.agent_id: allocation for allocation in state.allocations.values()}
    items = ""
    for bid in bids:
        allocation = allocations.get(bid.agent_id)
        allocated = allocation.allocated if allocation else 0
        items += (
            f"<li>{html.escape(bid.role)} {html.escape(bid.bid_type)} "
            f"<span class='subtle'>bid {bid.requested_attention}, got {allocated}, value {bid.expected_value}</span></li>"
        )
    items = items or "<li class='subtle'>No market activity yet</li>"
    return f'<div class="panel"><h2>Research Market</h2><ul>{items}</ul></div>'


def domain_list(domains: list[dict]) -> str:
    items = "".join(
        f"<li>{html.escape(domain['name'])} <span class='subtle'>priority {domain['priority']}, {html.escape(domain['status'])}</span></li>"
        for domain in domains
    ) or "<li class='subtle'>No active domains</li>"
    return f'<div class="panel"><h2>Research Ecology</h2><ul>{items}</ul></div>'


def organization_list(state: ResearchState) -> str:
    actions = sorted(state.organization_actions.values(), key=lambda item: item.created_at, reverse=True)[:8]
    reports = state.artifacts_by_kind(ArtifactKind.HEALTH_REPORT)
    memory = state.artifacts_by_kind(ArtifactKind.META_MEMORY)
    items = "".join(
        f"<li>{html.escape(action.action)} {html.escape(action.agent_id)} <span class='subtle'>{html.escape(action.reason)}</span></li>"
        for action in actions
    )
    if reports:
        items += f"<li>health report <span class='subtle'>{html.escape(reports[-1].title)}</span></li>"
    if memory:
        items += f"<li>meta memory <span class='subtle'>{html.escape(memory[-1].title)}</span></li>"
    items = items or "<li class='subtle'>No organization changes yet</li>"
    return f'<div class="panel"><h2>Organizational Learning</h2><ul>{items}</ul></div>'


def load_journal_entries(root: str | Path | None) -> list[dict]:
    if root is None:
        return []
    base = Path(root) / "research_journal"
    if not base.exists():
        return []
    entries = []
    for path in sorted(base.glob("cycle_*.md")):
        stem = path.stem.rsplit("_", 1)[-1]
        try:
            cycle = int(stem)
        except ValueError:
            continue
        entries.append({"cycle": cycle, "path": path.name, "markdown": path.read_text(encoding="utf-8")})
    return entries


def write_dashboard(
    state: ResearchState, out_path: str | Path, journal_root: str | Path | None = None
) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_dashboard(state, journal_root=journal_root), encoding="utf-8")
    return path
