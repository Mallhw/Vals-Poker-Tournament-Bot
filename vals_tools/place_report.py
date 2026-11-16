"""Finishing place per entry of ours on real-player tables: what the final rating (OpenSkill on placements) rewards.

A real-player table: every other seat is a team entry (no house bots). Per checkpoint of ours: matches, mean place
(1 = best, 6 = worst) with a 95% interval, share of opponents finished above, top-2 rate and bust rate. Other teams'
entries on the same tables are summarized the same way for reference.

    python -m vals_tools.place_report [--all-tables] [--since HHMM]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEAM = "org_9197691edc5e"


def summary(places: list[tuple[int, bool, int]]) -> str:
    n = len(places)
    mean = sum(p for p, _, _ in places) / n
    sd = math.sqrt(sum((p - mean) ** 2 for p, _, _ in places) / max(1, n - 1))
    beat = sum((seats - p) / (seats - 1) for p, _, seats in places) / n
    return (f"{n:4} matches  mean place {mean:.2f} ±{1.96 * sd / math.sqrt(n):.2f}  beats {beat:5.1%} of opponents  "
            f"top-2 {sum(p <= 2 for p, _, _ in places) / n:5.1%}  busted {sum(b for _, b, _ in places) / n:5.1%}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--all-tables", action="store_true", help="count tables with house bots too")
    p.add_argument("--since", default="", help="only matches started at or after this UTC time (HHMM)")
    args = p.parse_args()
    ours, theirs = defaultdict(list), defaultdict(list)
    for path in sorted(glob.glob(str(ROOT / "data" / "arena" / "matches" / "m_*.json"))):
        if args.since and Path(path).stem.split("t")[1][:4] < args.since:
            continue
        m = json.loads(Path(path).read_text(encoding="utf-8"))
        entries = {s["seat"]: s.get("entry") or {} for s in m["seats"]}
        mine = [s for s, e in entries.items() if e.get("org_id") == TEAM]
        if len(mine) != 1:
            continue
        others = [e for s, e in entries.items() if s != mine[0]]
        if not args.all_tables and any(e.get("kind") != "team" for e in others):
            continue
        seats = len(m["results"])
        for r in m["results"]:
            e = entries[r["seat"]]
            row = (r["place"], r.get("eliminated_on_hand") is not None, seats)
            if r["seat"] == mine[0]:
                ours[(e.get("deployment") or {}).get("checkpoint_id") or "base"].append(row)
            else:
                theirs[e.get("name", "?")].append(row)
    print("ours, by checkpoint" + ("" if args.all_tables else " (real-player tables only)"))
    for ck, rows in sorted(ours.items(), key=lambda kv: -len(kv[1])):
        print(f"  {ck:20} {summary(rows)}")
    print("other teams on the same tables")
    for name, rows in sorted(theirs.items(), key=lambda kv: sum(p for p, _, _ in kv[1]) / len(kv[1])):
        if len(rows) >= 20:
            print(f"  {name[:20]:20} {summary(rows)}")


if __name__ == "__main__":
    main()
