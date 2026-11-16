"""What our arena seats' fallbacks (format errors, token-cap cutoffs, timeouts) cost, from data/arena/matches/m_*.json.

A failed turn is played as check if checking is legal, else fold. A fallback check costs nothing directly; a fallback
fold facing a bet gives up the pot. Its cost is estimated as the call's expected value against the hands actually still
in (every seat's hole cards are in the match file): max(0, equity x (pot + call) - call), with equity over random
runouts of the rest of the board (the harness's true_equity). It ignores later betting, so it understates folds of
strong hands that would have won more, and overstates weak ones. Run from harness/ (its venv has pokerkit):

    cd harness && uv run python ../vals_tools/fallback_cost.py [--since HH:MM UTC] [--deployment infer_...]
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness" / "src"))
TEAM = "org_9197691edc5e"


def failed(a: dict) -> bool:
    return bool(a.get("fallback") or a.get("format_error") or a.get("token_cap") or a.get("timed_out"))


def main() -> None:
    from poker_harness.engine.equity import true_equity

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matches", default=str(ROOT / "data" / "arena" / "matches"))
    p.add_argument("--since", default="", help="only matches started at or after this UTC time, HHMM (e.g. 1720)")
    p.add_argument("--deployment", action="append", help="only these deployments of ours")
    args = p.parse_args()
    by = defaultdict(lambda: {"matches": 0, "hands": 0, "decisions": 0, "fallbacks": 0, "cap": 0, "folds": 0,
                              "cost": 0.0, "worst": []})
    for path in sorted(glob.glob(f"{args.matches}/m_*.json")):
        stamp = Path(path).stem.split("t")[1][:4]
        if args.since and stamp < args.since:
            continue
        m = json.loads(Path(path).read_text(encoding="utf-8"))
        bb = m["format"].get("big_blind") or 2
        mine = {s["seat"]: (s.get("entry") or {}).get("deployment_id") for s in m["seats"]
                if (s.get("entry") or {}).get("org_id") == TEAM}
        for seat, dep in mine.items():
            if args.deployment and dep not in args.deployment:
                continue
            b = by[dep]
            b["matches"] += 1
            for h in m["hands"]:
                cards = {pl["seat"]: pl.get("hole_cards") for pl in h["players"]}
                if seat not in cards:
                    continue
                b["hands"] += 1
                folded: set[int] = set()
                for a in h["actions"]:
                    if a["seat"] == seat:
                        b["decisions"] += 1
                        if failed(a):
                            b["fallbacks"] += 1
                            b["cap"] += bool(a.get("token_cap"))
                            call = a.get("to_call") or 0
                            if a["action"] == "fold" and call > 0:
                                b["folds"] += 1
                                rivals = [cards[s] for s in cards if s != seat and s not in folded and cards[s]]
                                eq = true_equity(cards[seat], rivals, a.get("board") or []) if rivals else 1.0
                                ev = max(0.0, eq * (a["pot_before"] + call) - call) / bb
                                b["cost"] += ev
                                b["worst"].append((round(ev, 1), a["street"], a["pot_before"], call, round(eq, 2),
                                                   "".join(cards[seat]), Path(path).stem[2:], h["hand_number"]))
                    if a["action"] == "fold":
                        folded.add(a["seat"])
    for dep, b in sorted(by.items(), key=lambda kv: -kv[1]["decisions"]):
        if not b["decisions"]:
            continue
        print(f"{dep}: {b['matches']} matches, {b['hands']} hands, {b['decisions']} decisions; fallbacks {b['fallbacks']}"
              f" ({b['fallbacks'] / b['decisions']:.1%}; {b['cap']} at the token cap); fallback folds facing a bet "
              f"{b['folds']}, est. cost {b['cost']:.0f} bb = {100 * b['cost'] / max(1, b['hands']):.1f} bb/100 hands"
              f" ({b['cost'] / max(1, b['fallbacks']):.1f} bb per fallback)")
        for w in sorted(b["worst"], reverse=True)[:4]:
            print(f"    {w[0]:6.1f} bb  {w[1]:7} pot {w[2]:4} call {w[3]:4} equity {w[4]:.2f} {w[5]}  {w[6]} h{w[7]}")


if __name__ == "__main__":
    main()
