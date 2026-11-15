"""Publish lab data to the dashboard repo (Mallhw/pokerbot-lab, GitHub Pages at matthewli.org/pokerbot-lab).

    python3 -I lab/publish.py          # run lab/arena.py first for fresh arena data

Copies only public arena data and our own experiment results: arena summary, rank/rating history, the experiment log
(name/result lines only) and trimmed replays of recent matches. No keys, no private info.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

LAB = Path(__file__).resolve().parent
DATA = LAB / "data"
SITE = DATA / "site"  # local clone of the dashboard repo (gitignored)
REPO = "https://github.com/Mallhw/pokerbot-lab.git"
ORG = "org_9197691edc5e"
REPLAYS = 12


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(SITE), *args], capture_output=True, text=True).stdout


def trim(match: dict) -> dict:
    us = next(s["seat"] for s in match["seats"] if (s["entry"] or {}).get("org_id") == ORG)
    keep_a = ("seat", "street", "action", "amount", "pot_before", "stack_after", "board", "say")
    return {
        "id": match["id"],
        "us": us,
        "seats": [{"seat": s["seat"], "name": (s["entry"] or {}).get("name") or f"seat {s['seat']}"} for s in match["seats"]],
        "hands": [{
            "hand_number": h["hand_number"], "button_seat": h["button_seat"], "blinds": h.get("blinds"),
            "players": h["players"], "board": h["board"], "shown": h.get("shown"), "deltas": h.get("deltas"),
            "actions": [{k: a.get(k) for k in keep_a} for a in h["actions"]],
        } for h in match["hands"]],
    }


def main() -> None:
    if not (SITE / ".git").exists():
        subprocess.run(["git", "clone", "-q", REPO, str(SITE)], check=True)
    git("pull", "-q", "--ff-only")
    out = SITE / "data"
    (out / "replays").mkdir(parents=True, exist_ok=True)

    summary = json.loads((DATA / "arena_summary.json").read_text())
    history = [json.loads(line) for line in (DATA / "arena_history.jsonl").read_text().splitlines() if line.strip()]
    exps = []
    exp_file = LAB / "experiments.jsonl"
    if exp_file.exists():
        for line in exp_file.read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                exps.append({k: e.get(k) for k in ("time", "name", "kind", "result")})

    matches = sorted((DATA / "arena" / "matches").glob("*.json"), key=lambda p: json.loads(p.read_text())["started_at"])
    replays = []
    for p in reversed(matches[-REPLAYS:]):
        m = json.loads(p.read_text())
        us = next(s["seat"] for s in m["seats"] if (s["entry"] or {}).get("org_id") == ORG)
        r = next(x for x in m["results"] if x["seat"] == us)
        (out / "replays" / f"{m['id']}.json").write_text(json.dumps(trim(m)))
        replays.append({"id": m["id"], "label": f"{m['started_at'][5:16].replace('T', ' ')} UTC · place {r['place']} · "
                                                  f"{r['net_bb']:+.0f} bb"})
    keep = {f"{r['id']}.json" for r in replays}
    for old in (out / "replays").glob("*.json"):
        if old.name not in keep:
            old.unlink()

    summary.update({"experiments": exps[-60:], "replays": replays})
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "history.json").write_text(json.dumps(history))
    sc = DATA / "scouting.html"
    if sc.exists():
        (SITE / "scouting.html").write_text(sc.read_text())
    idx = SITE / "index.html"
    t = idx.read_text()
    if "scouting.html" not in t:
        t = t.replace("Vals × Applied Compute poker arena: 6-max no-limit hold'em, 30-hand elimination games.",
                      "Vals × Applied Compute poker arena: 6-max no-limit hold'em, 30-hand elimination games. "
                      "<a href=\"scouting.html\">Scouting reports on every opponent →</a>")
        idx.write_text(t)
    git("add", "-A")
    if git("status", "--porcelain").strip():
        subprocess.run(["git", "-C", str(SITE), "-c", "user.name=Mallhw",
                        "-c", "user.email=67278784+Mallhw@users.noreply.github.com", "commit", "-q", "-m",
                        "Update dashboard data"], check=True)
        subprocess.run(["git", "-C", str(SITE), "push", "-q", "origin", "HEAD:main"], check=True)
        print("published")
    else:
        print("no changes")


if __name__ == "__main__":
    main()
