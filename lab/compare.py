"""Side-by-side arena comparison of pineapple entries (by deployment id) from the public match logs.

    python3 -I lab/compare.py infer_a infer_b [infer_c ...] [--json]

Per entry: placement, survival, chips, preflop style (VPIP/PFR/3-bet/fold to raise/steals), postflop (c-bet,
showdowns), all-in behaviour (vs Bob the Buster and overall), and net chips vs each opponent. Plain python3.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
MATCHES = LAB / "data" / "arena" / "matches"
ORG = "org_9197691edc5e"


def stats(dep: str) -> dict:
    c = collections.Counter()
    places, bust_hands, end_chips = [], [], []
    vs_net: dict[str, float] = collections.defaultdict(float)
    name = dep
    for f in sorted(MATCHES.glob("*.json")):
        d = json.loads(f.read_text())
        s = next((s for s in d["seats"] if (s["entry"] or {}).get("org_id") == ORG), None)
        if not s or s["entry"].get("deployment_id") != dep:
            continue
        name = s["entry"].get("submission_name") or s["entry"].get("name") or dep
        seat = s["seat"]
        names = {x["seat"]: ((x["entry"] or {}).get("name") or f"seat{x['seat']}") for x in d["seats"]}
        r = next(x for x in d["results"] if x["seat"] == seat)
        places.append(r["place"])
        c["games"] += 1
        c["net_bb"] += r["net_bb"]
        c["ev_bb"] += r["ev_net_bb"]
        c["decisions"] += r["decisions"] or 0
        c["fallbacks"] += r["fallbacks"] or 0
        if r["eliminated_on_hand"]:
            bust_hands.append(r["eliminated_on_hand"])
        else:
            end_chips.append(r.get("final_stack") or r.get("stack") or 0)
        for h in d["hands"]:
            me = next((p for p in h["players"] if p["seat"] == seat and p["starting_stack"] > 0), None)
            if not me:
                continue
            c["hands"] += 1
            bb = max((h.get("blinds") or {}).values() or [2])
            acts = h["actions"]
            pre = [a for a in acts if a["street"] == "preflop"]
            mine_pre = [a for a in pre if a["seat"] == seat]
            c["vpip"] += any(a["action"] in ("call", "raise") for a in mine_pre)
            c["pfr"] += any(a["action"] == "raise" for a in mine_pre)
            # facing a raise preflop (first decision after someone else raised)
            raised_before = False
            first_raiser = None
            for a in pre:
                if a["seat"] == seat:
                    if raised_before:
                        c["face_raise"] += 1
                        c["fold_to_raise"] += a["action"] == "fold"
                        c["threebet"] += a["action"] == "raise"
                    elif first_raiser is None and a["action"] in ("raise", "fold", "call"):
                        # unopened pot: was it folded to us late (steal spot)?
                        if all(b["action"] == "fold" for b in pre if b is not a and pre.index(b) < pre.index(a)):
                            pos = (seat - h["button_seat"]) % 6
                            if pos in (0, 5) or a is mine_pre[0] and pos == 0:
                                c["steal_spot"] += 1
                                c["steal"] += a["action"] == "raise"
                    break
                if a["action"] == "raise":
                    raised_before = True
                    first_raiser = a["seat"]
            # c-bet: we were the last preflop raiser and bet the flop when we had the chance
            last_raiser = next((a["seat"] for a in reversed(pre) if a["action"] == "raise"), None)
            flop = [a for a in acts if a["street"] == "flop"]
            if last_raiser == seat and any(a["seat"] == seat for a in flop):
                c["cbet_spot"] += 1
                first_mine = next(a for a in flop if a["seat"] == seat)
                bets_before = any(a["action"] in ("bet", "raise") for a in flop[:flop.index(first_mine)])
                if not bets_before:
                    c["cbet_opp"] += 1
                    c["cbet"] += first_mine["action"] in ("bet", "raise")
            # all-in calls (calling >= half our stack)
            for a in acts:
                if a["seat"] == seat and a["action"] == "call" and (a.get("to_call") or 0) >= 0.5 * me["starting_stack"]:
                    c["big_call"] += 1
            # Bob the Buster shoves: our response
            shove = None
            for a in acts:
                if shove is None and names.get(a["seat"]) == "Bob the Buster" and a["action"] in ("raise", "bet") \
                        and a.get("stack_after") == 0:
                    shove = a
                elif shove is not None and a["seat"] == seat and a["action"] in ("call", "fold"):
                    if (a.get("to_call") or 0) >= 10 * bb:
                        c["bob_face"] += 1
                        if a["action"] == "call":
                            c["bob_call"] += 1
                            deltas = h.get("deltas") or {}
                            c["bob_call_net"] += deltas.get(str(seat), deltas.get(seat, 0)) or 0
                    break
            # showdown
            shown = h.get("shown") or {}
            if (isinstance(shown, dict) and (str(seat) in shown or seat in shown)):
                c["wtsd"] += 1
                deltas = h.get("deltas") or {}
                c["wsd"] += (deltas.get(str(seat), deltas.get(seat, 0)) or 0) > 0
            # net vs each opponent who also put chips in this pot
            deltas = h.get("deltas") or {}
            mine = deltas.get(str(seat), deltas.get(seat))
            if mine:
                vol = {a["seat"] for a in acts if a["action"] in ("call", "raise", "bet") and a["seat"] != seat}
                for o in vol:
                    vs_net[names.get(o, str(o))] += mine / bb / max(1, len(vol))
    if not c["games"]:
        return {"name": name, "games": 0}
    h = max(1, c["hands"])
    pc = collections.Counter(places)
    g = c["games"]
    return {
        "name": name, "deployment": dep, "games": g, "hands": c["hands"],
        "mean_place": round(sum(places) / g, 2), "place_dist": {p: pc.get(p, 0) for p in range(1, 7)},
        "win_rate": round(pc.get(1, 0) / g, 3), "top2_rate": round((pc.get(1, 0) + pc.get(2, 0)) / g, 3),
        "last_rate": round(pc.get(6, 0) / g, 3), "bust_rate": round(len(bust_hands) / g, 3),
        "mean_bust_hand": round(sum(bust_hands) / len(bust_hands), 1) if bust_hands else None,
        "bb100": round(100 * c["net_bb"] / h, 1), "ev_bb100": round(100 * c["ev_bb"] / h, 1),
        "vpip": round(c["vpip"] / h, 3), "pfr": round(c["pfr"] / h, 3),
        "fold_to_raise": round(c["fold_to_raise"] / max(1, c["face_raise"]), 3),
        "threebet": round(c["threebet"] / max(1, c["face_raise"]), 3),
        "steal": round(c["steal"] / max(1, c["steal_spot"]), 3), "steal_spots": c["steal_spot"],
        "cbet": round(c["cbet"] / max(1, c["cbet_opp"]), 3), "cbet_spots": c["cbet_opp"],
        "wtsd_per100": round(100 * c["wtsd"] / h, 1), "wsd": round(c["wsd"] / max(1, c["wtsd"]), 3),
        "big_calls_per100": round(100 * c["big_call"] / h, 2),
        "bob_called": f'{c["bob_call"]}/{c["bob_face"]}', "bob_call_net_chips": c["bob_call_net"],
        "fallback_rate": round(c["fallbacks"] / max(1, c["decisions"]), 4),
        "vs": dict(sorted(((k, round(v, 1)) for k, v in vs_net.items()), key=lambda kv: kv[1])),
    }


def main() -> None:
    deps = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = [stats(d) for d in deps]
    if "--json" in sys.argv:
        print(json.dumps(out, indent=1))
        return
    keys = [k for k in out[0] if k not in ("vs", "name", "deployment")]
    print(f"{'':22}" + "".join(f"{o['name'][:26]:>28}" for o in out))
    for k in keys:
        print(f"{k:22}" + "".join(f"{str(o.get(k)):>28}" for o in out))
    for o in out:
        print(f"\n{o['name']} net BB vs opponents (split by pot): {o.get('vs')}")


if __name__ == "__main__":
    main()
