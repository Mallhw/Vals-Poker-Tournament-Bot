"""How pineapple does against each arena opponent: contested pots, chips, showdowns, folds to aggression.

    python3 -I lab/vs_opponents.py [--entry butter] [--since 2026-11-15T09:00]

For each opponent (by arena entry name), over hands where both pineapple and that opponent put chips in
voluntarily: pots contested, pineapple's net BB in them, showdowns won/lost, how often pineapple folded to that
opponent's bet/raise vs the reverse. Writes lab/data/vs_opponents.json. Use it to find who we leak chips to.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

LAB = Path(__file__).resolve().parent
MATCHES = LAB / "data" / "arena" / "matches"
ORG = "org_9197691edc5e"
VOL = ("call", "raise", "bet")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entry", help="submission name (e.g. butter, opsd4-s30); default: all of pineapple's entries")
    p.add_argument("--since", help="only matches started at/after this ISO time (UTC)")
    a = p.parse_args()
    S: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    games = 0
    for f in MATCHES.glob("*.json"):
        d = json.loads(f.read_text())
        if a.since and d["started_at"] < a.since:
            continue
        us = next((s for s in d["seats"] if (s["entry"] or {}).get("org_id") == ORG), None)
        if not us:
            continue
        ent = us["entry"].get("submission_name") or us["entry"].get("name")
        if a.entry and ent != a.entry:
            continue
        games += 1
        u = us["seat"]
        name = {s["seat"]: ((s["entry"] or {}).get("name") or f"seat{s['seat']}") for s in d["seats"]}
        for h in d["hands"]:
            ua = [x for x in h["actions"] if x["seat"] == u]
            if not any(x["action"] in VOL for x in ua):
                continue
            du = h["deltas"][str(u)] / 2
            for o, oname in name.items():
                if o == u:
                    continue
                oa = [x for x in h["actions"] if x["seat"] == o]
                if not any(x["action"] in VOL for x in oa):
                    continue
                c = S[oname]
                c["pots"] += 1
                c["net_bb_x10"] += round(du * 10)
                last = None
                for x in h["actions"]:
                    if x["seat"] in (u, o) and x["action"] in ("bet", "raise"):
                        last = x["seat"]
                    if x["seat"] == u and x["action"] == "fold" and last == o:
                        c["we_fold_to_them"] += 1
                    if x["seat"] == o and x["action"] == "fold" and last == u:
                        c["they_fold_to_us"] += 1
                shown = h.get("shown") or {}
                if h.get("showdown") and str(u) in shown and str(o) in shown:
                    c["showdowns"] += 1
                    c["sd_won"] += h["deltas"][str(u)] > h["deltas"][str(o)]
    out = {}
    for n, c in sorted(S.items(), key=lambda x: x[1]["net_bb_x10"]):
        out[n] = {"pots": c["pots"], "net_bb": c["net_bb_x10"] / 10, "net_bb_per_pot": round(c["net_bb_x10"] / 10 / max(1, c["pots"]), 2),
                  "we_fold_to_them": c["we_fold_to_them"], "they_fold_to_us": c["they_fold_to_us"],
                  "showdowns": c["showdowns"], "sd_won": c["sd_won"]}
    (LAB / "data" / "vs_opponents.json").write_text(json.dumps({"entry": a.entry, "since": a.since, "games": games, "opponents": out}, indent=1))
    print(f"entry={a.entry or 'all'} since={a.since} games={games}  (net BB is pineapple's result in pots both played)")
    print("opponent".ljust(22) + "pots".rjust(6) + "netBB".rjust(9) + "/pot".rjust(7) + "weFold".rjust(8) + "theyFold".rjust(9) + "SD won".rjust(9))
    for n, v in out.items():
        print(n[:22].ljust(22) + str(v["pots"]).rjust(6) + f'{v["net_bb"]:+.1f}'.rjust(9) + f'{v["net_bb_per_pot"]:+.2f}'.rjust(7)
              + str(v["we_fold_to_them"]).rjust(8) + str(v["they_fold_to_us"]).rjust(9) + f'{v["sd_won"]}/{v["showdowns"]}'.rjust(9))


if __name__ == "__main__":
    main()
