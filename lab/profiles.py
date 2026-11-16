"""Opponent profiles from arena matches: how each bot plays, for opponent modeling in the hint.

    python3 -I lab/profiles.py          # prints a table, writes lab/data/profiles.json

Per opponent (by entry name): hands seen, VPIP, PFR, limp%, 3-bet%, fold-to-raise%, preflop shove%, post-flop
aggression (bets+raises per call), showdown strength (mean Chen score of shown hands), and how often a big
river bet/raise was shown down as a weak hand (a bluff proxy).
"""

from __future__ import annotations

import collections
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arena import MATCHES, chen  # noqa: E402

ORG = "org_9197691edc5e"


def main() -> None:
    S: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    sd: dict[str, list[float]] = collections.defaultdict(list)
    for f in MATCHES.glob("*.json"):
        d = json.loads(f.read_text())
        name = {s["seat"]: ((s["entry"] or {}).get("name") or f"seat{s['seat']}") for s in d["seats"]}
        us = next((s["seat"] for s in d["seats"] if (s["entry"] or {}).get("org_id") == ORG), None)
        for h in d["hands"]:
            pre_all = [a for a in h["actions"] if a["street"] == "preflop"]
            for p in h["players"]:
                if p["starting_stack"] <= 0 or p["seat"] == us:
                    continue
                s, n = p["seat"], name[p["seat"]]
                c = S[n]
                c["hands"] += 1
                mine = [a for a in h["actions"] if a["seat"] == s]
                pre = [a for a in mine if a["street"] == "preflop"]
                c["vpip"] += any(a["action"] in ("call", "raise") for a in pre)
                c["pfr"] += any(a["action"] == "raise" for a in pre)
                c["shove"] += any(a["action"] == "raise" and a["stack_after"] == 0 for a in pre)
                if pre:
                    i = pre_all.index(pre[0])
                    raised = any(a["action"] == "raise" for a in pre_all[:i])
                    if raised:
                        c["faced_raise"] += 1
                        c["fold_vs_raise"] += pre[0]["action"] == "fold"
                        c["threebet"] += pre[0]["action"] == "raise"
                    else:
                        c["unopened"] += 1
                        c["limp"] += pre[0]["action"] == "call"
                post = [a for a in mine if a["street"] != "preflop"]
                c["post_aggr"] += sum(a["action"] in ("bet", "raise") for a in post)
                c["post_call"] += sum(a["action"] == "call" for a in post)
                shown = (h.get("shown") or {}).get(str(s))
                if shown:
                    sd[n].append(chen(shown))
                    big_river = any(a["street"] == "river" and a["action"] in ("bet", "raise")
                                    and (a.get("amount") or 0) >= 0.5 * max(1, a["pot_before"]) for a in mine)
                    if big_river:
                        c["river_big_shown"] += 1
                        c["river_big_weak"] += chen(shown) < 6
    out = {}
    for n, c in S.items():
        if c["hands"] < 150:
            continue
        h = c["hands"]
        out[n] = {
            "hands": h,
            "vpip": round(c["vpip"] / h, 3),
            "pfr": round(c["pfr"] / h, 3),
            "limp": round(c["limp"] / max(1, c["unopened"]), 3),
            "threebet": round(c["threebet"] / max(1, c["faced_raise"]), 3),
            "fold_vs_raise": round(c["fold_vs_raise"] / max(1, c["faced_raise"]), 3),
            "shove": round(c["shove"] / h, 3),
            "post_aggression": round(c["post_aggr"] / max(1, c["post_call"]), 2),
            "showdown_chen": round(statistics.fmean(sd[n]), 1) if sd[n] else None,
            "showdowns": len(sd[n]),
            "big_river_bluff_rate": round(c["river_big_weak"] / c["river_big_shown"], 2) if c["river_big_shown"] else None,
        }
    (Path(__file__).resolve().parent / "data" / "profiles.json").write_text(json.dumps(out, indent=1))
    cols = ["hands", "vpip", "pfr", "limp", "threebet", "fold_vs_raise", "shove", "post_aggression", "showdown_chen",
            "big_river_bluff_rate"]
    print("opponent".ljust(22) + "".join(k[:9].rjust(10) for k in cols))
    for n, v in sorted(out.items(), key=lambda x: -x[1]["hands"]):
        print(n[:22].ljust(22) + "".join(str(v[k]).rjust(10) for k in cols))


if __name__ == "__main__":
    main()
