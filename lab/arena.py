"""Arena tracker for team pineapple: pull new matches, compute metrics per submitted entry, track rank/rating.

    python3 -I lab/arena.py            # update data, print a summary
    python3 -I lab/arena.py --json     # print the summary as JSON

Writes lab/data/arena/matches/*.json (public match logs), lab/data/arena_summary.json and appends
lab/data/arena_history.jsonl (rank/rating over time). Plain python3, no dependencies.
"""

from __future__ import annotations

import collections
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

LAB = Path(__file__).resolve().parent
DATA = LAB / "data"
MATCHES = DATA / "arena" / "matches"
ORG = "org_9197691edc5e"
SITE = "https://www.pokerbots.ac"
UA = {"User-Agent": "pokerbot-lab/0.1"}


def get(url: str) -> str:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def team_status() -> dict:
    txt = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", get(f"{SITE}/teams/{ORG}")))
    out = {"time": time.strftime("%Y-%m-%d %H:%M:%S")}
    for key, pat in [("rank", r"Rank #(\d+)"), ("rating", r"Rating ([\d,]+)"), ("bb100", r"bb/100 ([−+-]?[\d.]+)"),
                     ("matches", r"Ranked matches (\d+)"), ("deployment", r"(infer_[0-9a-f]+) Live")]:
        m = re.search(pat, txt)
        if m:
            v = m.group(1).replace(",", "").replace("−", "-")
            out[key] = v if key == "deployment" else float(v)
    return out


def fetch_new() -> int:
    MATCHES.mkdir(parents=True, exist_ok=True)
    ids: set[str] = set()
    for page in range(1, 8):
        found = set(re.findall(r"m_2026[0-9t]*_[0-9a-f]+", get(f"{SITE}/matches?entry={ORG}&page={page}")))
        if not found - ids:
            break
        ids |= found
    new = 0
    for mid in sorted(ids):
        path = MATCHES / f"{mid}.json"
        if path.exists() and path.stat().st_size:
            continue
        m = re.search(r"https://[a-z0-9]+\.public\.blob\.vercel-storage\.com/matches/[^\"\\ ]+\.json",
                      get(f"{SITE}/matches/{mid}"))
        if m:
            path.write_text(get(m.group(0)))
            new += 1
    return new


def chen(cards: list[str]) -> float:
    """Chen formula hand score (same scale as poker_harness.spots.chen_score)."""
    vals = {"A": 10, "K": 8, "Q": 7, "J": 6, "T": 5}
    order = "23456789TJQKA"
    r = sorted((c[0] for c in cards), key=order.index, reverse=True)
    hi = vals.get(r[0], int(r[0]) / 2 if r[0] not in vals else 0)
    if r[0] == r[1]:
        return max(5, hi * 2)
    s = hi + (2 if cards[0][1] == cards[1][1] else 0)
    gap = order.index(r[0]) - order.index(r[1]) - 1
    s -= {0: 0, 1: 1, 2: 2, 3: 4}.get(gap, 5)
    if gap <= 1 and order.index(r[0]) < order.index("Q"):
        s += 1
    return s


def analyze(files: list[Path]) -> dict:
    by_entry: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    places: dict[str, list[int]] = collections.defaultdict(list)
    busts: list[dict] = []
    for f in sorted(files, key=lambda p: p.name):
        d = json.loads(f.read_text())
        seat_entry = next((s for s in d["seats"] if (s["entry"] or {}).get("org_id") == ORG), None)
        if not seat_entry:
            continue
        s, e = seat_entry["seat"], seat_entry["entry"]
        key = f'{e.get("submission_name") or e.get("name")} ({e.get("deployment_id")})'
        r = next(x for x in d["results"] if x["seat"] == s)
        c = by_entry[key]
        places[key].append(r["place"])
        c["games"] += 1
        c["busts"] += r["eliminated_on_hand"] is not None
        c["net_bb"] += r["net_bb"]
        c["ev_bb"] += r["ev_net_bb"]
        c["hands"] += r["hands"]
        c["decisions"] += r["decisions"] or 0
        c["fallbacks"] += r["fallbacks"] or 0
        for h in d["hands"]:
            me = next((p for p in h["players"] if p["seat"] == s and p["starting_stack"] > 0), None)
            if not me:
                continue
            pre = [a for a in h["actions"] if a["seat"] == s and a["street"] == "preflop"]
            c["vpip"] += any(a["action"] in ("call", "raise") for a in pre)
            c["pfr"] += any(a["action"] == "raise" for a in pre)
            c["bigcall"] += sum(1 for a in h["actions"] if a["seat"] == s and a["action"] == "call"
                                and a["to_call"] >= 0.5 * me["starting_stack"])
        if r["eliminated_on_hand"]:
            h = d["hands"][r["eliminated_on_hand"] - 1]
            hole = next(p["hole_cards"] for p in h["players"] if p["seat"] == s)
            mine = [a["action"] for a in h["actions"] if a["seat"] == s]
            busts.append({"match": d["id"], "entry": key, "hand": h["hand_number"], "hole": hole,
                          "chen": chen(hole), "actions": mine, "board": h["board"]})
    entries = {}
    for key, c in by_entry.items():
        h = max(1, c["hands"])
        p = places[key]
        entries[key] = {
            "games": c["games"], "mean_place": round(sum(p) / len(p), 2), "wins": p.count(1), "lasts": p.count(6),
            "bust_rate": round(c["busts"] / c["games"], 3), "bb100": round(100 * c["net_bb"] / h, 1),
            "ev_bb100": round(100 * c["ev_bb"] / h, 1), "vpip": round(c["vpip"] / h, 3), "pfr": round(c["pfr"] / h, 3),
            "bigcalls100": round(100 * c["bigcall"] / h, 2), "fallback_rate": round(c["fallbacks"] / max(1, c["decisions"]), 4),
        }
    weak = [b for b in busts if b["chen"] < 8]
    return {"entries": entries, "busts": busts, "weak_busts": len(weak), "total_busts": len(busts)}


def main() -> None:
    new = fetch_new()
    status = team_status()
    summary = {"status": status, "new_matches": new, **analyze(list(MATCHES.glob("*.json")))}
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "arena_summary.json").write_text(json.dumps(summary, indent=1))
    with open(DATA / "arena_history.jsonl", "a") as f:
        f.write(json.dumps(status) + "\n")
    if "--json" in sys.argv:
        print(json.dumps(summary, indent=1))
        return
    print(f"status: {status}  (+{new} new matches)")
    for k, v in summary["entries"].items():
        print(f"  {k}: {v}")
    print(f"busts: {summary['total_busts']} total, {summary['weak_busts']} with a weak hand (Chen < 8)")


if __name__ == "__main__":
    main()
