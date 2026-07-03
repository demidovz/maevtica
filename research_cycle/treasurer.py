#!/usr/bin/env python3
"""Gap #1 — campaign TREASURER (казначей): the budget cap for a research run.

A "campaign" is one budgeted research run. The treasurer:
  • sizes the campaign in POINTS OF THE REAL /usage WEEK GAUGE (mst-usage):
    "20% of the weekly limit" = 20 points; open records the gauge, can_continue
    stops at open+points. Immune to tank-calibration error by construction;
  • keeps a per-campaign spend ledger (for the final report);
  • keeps an approximate token cap (fuel.toml calibration) as a SECONDARY latch
    and for sizing the Workflow's capTokens arg.
  2026-07-04 lesson: do NOT try to reproduce Anthropic's weekly math from
  usage.db — wrong window (rolling 7d vs reset-aligned) + unverifiable tank
  (computed 26% vs real 6%). Read the real gauge instead.

Enforcement note: when the loop runs as a Workflow, the Workflow's own
`budget.total` should be set to treasurer.cap() and the loop gated on
`budget.remaining()`. This module OWNS the cap math + campaign accounting +
report material; the Workflow owns the live token metering. Keeping the two in
sync is the whole point — the campaign never silently outspends its slice.

CLI:
  treasurer.py open  <campaign> --domain "<d>" --frac 0.20   # create + print cap
  treasurer.py spend <campaign> --tokens N --stage explorer   # record spend
  treasurer.py status <campaign> [--json]
  treasurer.py can-continue <campaign> [--reserve 0.10]       # exit 0=go 3=stop
"""
from __future__ import annotations
import argparse, json, sqlite3, subprocess, sys, tomllib
from datetime import datetime, timedelta, timezone
from pathlib import Path

STATE = Path.home() / ".local" / "state" / "mst" / "research_cycle"
FUEL_TOML = Path.home() / ".config" / "mst" / "fuel.toml"
USAGE_DB = Path.home() / ".claude" / "usage.db"
MST_USAGE = Path.home() / "workspace" / "maestratica" / "scripts" / "mst-usage"
WEEK_SECONDS = 7 * 24 * 3600


def live_week_pct() -> int | None:
    """REAL weekly Claude % straight from the /usage gauge (mst-usage --json).

    This is the PRIMARY weekly meter (2026-07-04 lesson: reproducing Anthropic's
    math from usage.db was wrong twice over — rolling-7d window vs reset-aligned
    week, and an unverifiable tank calibration: computed 26% vs real 6%).
    None when the gauge is unavailable."""
    try:
        out = subprocess.run([str(MST_USAGE), "--json"], capture_output=True,
                             text=True, timeout=90)
        if out.returncode != 0:
            return None
        week = json.loads(out.stdout).get("week") or {}
        pct = week.get("pct")
        return int(pct) if pct is not None else None
    except Exception:
        return None


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def weekly_limit_tokens() -> int | None:
    """Calibrated weekly Claude billable-token limit from fuel.toml (or None)."""
    try:
        cfg = tomllib.loads(FUEL_TOML.read_text())
        v = int(cfg.get("claude", {}).get("limit_week_tokens", 0) or 0)
        return v or None
    except Exception:
        return None


def weekly_used_tokens() -> int | None:
    """Best-effort billable Claude tokens in the last 7d (usage.db). None if N/A.

    Returns 0 when the db exists but has no recent rows — treated as 'unknown/low'
    by callers, NOT as a hard 'we've used nothing'."""
    if not USAGE_DB.exists():
        return None
    try:
        con = sqlite3.connect(f"file:{USAGE_DB}?mode=ro", uri=True)
        cut = (datetime.now(timezone.utc) - timedelta(seconds=WEEK_SECONDS)
               ).strftime("%Y-%m-%dT%H:%M:%S")
        row = con.execute(
            "SELECT COALESCE(SUM(input_tokens),0)+COALESCE(SUM(output_tokens),0)"
            "+COALESCE(SUM(cache_creation_tokens),0) FROM turns WHERE timestamp>=?",
            (cut,)).fetchone()
        con.close()
        return int(row[0] or 0)
    except sqlite3.Error:
        return None


def _dir(campaign: str) -> Path:
    return STATE / campaign


def open_campaign(campaign: str, domain: str, frac: float) -> dict:
    d = _dir(campaign)
    d.mkdir(parents=True, exist_ok=True)
    week0 = live_week_pct()
    wl = weekly_limit_tokens()
    if week0 is None and not wl:
        sys.exit("treasurer: no weekly meter at all — /usage gauge unreachable AND no "
                 "[claude].limit_week_tokens in ~/.config/mst/fuel.toml. Cannot size the campaign.")
    # PRIMARY: the campaign owns frac·100 points of the REAL /usage week gauge.
    # SECONDARY: a token cap from the (approximate) toml calibration, as a backstop.
    manifest = {"campaign": campaign, "domain": domain, "frac": frac,
                "cap_week_points": round(frac * 100, 1), "week_pct_at_open": week0,
                "weekly_limit_tokens": wl, "cap_tokens": int(frac * wl) if wl else None,
                "opened_at": _now()}
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    (d / "ledger.jsonl").touch()
    return manifest


def _load(campaign: str) -> dict:
    m = _dir(campaign) / "manifest.json"
    if not m.exists():
        sys.exit(f"treasurer: campaign {campaign!r} not open (run: treasurer.py open …)")
    return json.loads(m.read_text())


def record_spend(campaign: str, tokens: int, stage: str, note: str = "") -> None:
    _load(campaign)
    line = {"ts": _now(), "stage": stage, "tokens": int(tokens), "note": note}
    with (_dir(campaign) / "ledger.jsonl").open("a") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")


def spent(campaign: str) -> int:
    p = _dir(campaign) / "ledger.jsonl"
    if not p.exists():
        return 0
    return sum(json.loads(l)["tokens"] for l in p.read_text().splitlines() if l.strip())


def status(campaign: str) -> dict:
    man = _load(campaign)
    sp = spent(campaign)
    cap = man.get("cap_tokens")
    week_now = live_week_pct()
    points_used = (week_now - man["week_pct_at_open"]
                   if week_now is not None and man.get("week_pct_at_open") is not None else None)
    return {**man, "spent_tokens": sp,
            "remaining_tokens": (cap - sp) if cap else None,
            "spent_pct_of_cap": round(100 * sp / cap, 1) if cap else None,
            "week_pct_now": week_now, "week_points_used": points_used}


def can_continue(campaign: str, reserve: float = 0.10, week_ceiling_pct: int = 90) -> bool:
    """Go/stop. PRIMARY: stop when the campaign has consumed its share of the
    REAL /usage week gauge (points since open ≥ cap_week_points·(1−reserve)) or
    the whole week is nearly exhausted (≥ week_ceiling_pct). SECONDARY: the
    campaign's own token ledger vs the approximate token cap."""
    man = _load(campaign)
    week_now = live_week_pct()
    if week_now is not None:
        if week_now >= week_ceiling_pct:
            return False
        w0 = man.get("week_pct_at_open")
        if w0 is not None and (week_now - w0) >= man["cap_week_points"] * (1 - reserve):
            return False
    cap = man.get("cap_tokens")
    if cap and spent(campaign) >= cap * (1 - reserve):
        return False
    return True


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("open"); o.add_argument("campaign"); o.add_argument("--domain", required=True); o.add_argument("--frac", type=float, default=0.20)
    s = sub.add_parser("spend"); s.add_argument("campaign"); s.add_argument("--tokens", type=int, required=True); s.add_argument("--stage", required=True); s.add_argument("--note", default="")
    st = sub.add_parser("status"); st.add_argument("campaign"); st.add_argument("--json", action="store_true")
    cc = sub.add_parser("can-continue"); cc.add_argument("campaign"); cc.add_argument("--reserve", type=float, default=0.10)
    a = p.parse_args()
    if a.cmd == "open":
        m = open_campaign(a.campaign, a.domain, a.frac)
        w0 = m["week_pct_at_open"]
        gauge = (f"week gauge {w0}% now → stop at {w0 + m['cap_week_points']:.0f}%"
                 if w0 is not None else "week gauge UNAVAILABLE (token backstop only)")
        cap_s = f" · token backstop {m['cap_tokens']:,}" if m.get("cap_tokens") else ""
        print(f"campaign {a.campaign!r} opened · domain={a.domain!r} · "
              f"{m['cap_week_points']} week-points · {gauge}{cap_s}")
    elif a.cmd == "spend":
        record_spend(a.campaign, a.tokens, a.stage, a.note)
        print(f"recorded {a.tokens:,} tokens @ {a.stage}; total {spent(a.campaign):,}")
    elif a.cmd == "status":
        r = status(a.campaign)
        print(json.dumps(r, indent=2, ensure_ascii=False) if a.json
              else f"{a.campaign}: spent {r['spent_tokens']:,}/{r['cap_tokens']:,} "
                   f"({r['spent_pct_of_cap']}% of cap) · domain={r['domain']!r}")
    elif a.cmd == "can-continue":
        ok = can_continue(a.campaign, a.reserve)
        print("GO" if ok else "STOP")
        return 0 if ok else 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
