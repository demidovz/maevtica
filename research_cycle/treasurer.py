#!/usr/bin/env python3
"""Gap #1 — campaign TREASURER (казначей): the budget cap for a research run.

A "campaign" is one budgeted research run. The treasurer:
  • turns "X% of the weekly Claude limit" into an absolute token cap,
    reading the studio's calibrated weekly limit from ~/.config/mst/fuel.toml;
  • keeps a per-campaign spend ledger (for the final report + hard stop);
  • answers can_continue() so the loop stops cleanly at the cap;
  • cross-checks the real weekly Claude burn (usage.db, 7d) as a soft ceiling,
    degrading gracefully when that signal is unavailable (it currently reads 0
    in some environments — so it is a *safety* check, never the primary meter).

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
import argparse, json, sqlite3, sys, tomllib
from datetime import datetime, timedelta, timezone
from pathlib import Path

STATE = Path.home() / ".local" / "state" / "mst" / "research_cycle"
FUEL_TOML = Path.home() / ".config" / "mst" / "fuel.toml"
USAGE_DB = Path.home() / ".claude" / "usage.db"
WEEK_SECONDS = 7 * 24 * 3600


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
    wl = weekly_limit_tokens()
    if not wl:
        sys.exit("treasurer: weekly Claude limit unknown (set [claude].limit_week_tokens "
                 "in ~/.config/mst/fuel.toml). Cannot size the campaign.")
    cap = int(frac * wl)
    manifest = {"campaign": campaign, "domain": domain, "frac": frac,
                "weekly_limit_tokens": wl, "cap_tokens": cap,
                "opened_at": _now(), "weekly_used_at_open": weekly_used_tokens()}
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
    cap = man["cap_tokens"]
    used_now = weekly_used_tokens()
    return {**man, "spent_tokens": sp, "remaining_tokens": cap - sp,
            "spent_pct_of_cap": round(100 * sp / cap, 1) if cap else None,
            "weekly_used_now": used_now,
            "weekly_used_pct": (round(100 * used_now / man["weekly_limit_tokens"], 1)
                                if used_now else None)}


def can_continue(campaign: str, reserve: float = 0.10, weekly_ceiling: float = 0.90) -> bool:
    """Go/stop. Stop if campaign spent ≥ cap·(1−reserve), OR (soft) if the whole
    week's Claude burn is already ≥ weekly_ceiling of the limit."""
    man = _load(campaign)
    if spent(campaign) >= man["cap_tokens"] * (1 - reserve):
        return False
    used = weekly_used_tokens()
    if used and used >= man["weekly_limit_tokens"] * weekly_ceiling:
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
        print(f"campaign {a.campaign!r} opened · domain={a.domain!r} · "
              f"cap={m['cap_tokens']:,} tokens ({a.frac:.0%} of {m['weekly_limit_tokens']:,}/week)")
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
