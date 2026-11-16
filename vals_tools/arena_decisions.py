"""Where does an arena entry give up chips? Every decision of one checkpoint, judged against the opponents' real cards.

Arena replays publish every hole card, so each call or fold facing a bet can be scored against the hands still in:
equity from the harness's own `true_equity` (300 sampled runouts), and the call's EV in big blinds,
`equity * (pot + to_call) - to_call`. That is hindsight with perfect information (it ignores later streets and
what the opponent's range could have been), so read it in aggregate and on examples, not as a verdict per hand.
Also: failed decisions (format error, timeout, token cap, fallback) with their reason, and bets/raises split by the
equity they had (bluffs under 30%, value over 60%) with how often everyone folded.

    harness/.venv/bin/python -m vals_tools.arena_decisions --checkpoint ckpt_db759b762b5d
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness" / "src"))
sys.path.insert(0, str(ROOT))
from poker_harness.engine.equity import true_equity  # noqa: E402
from vals_tools.arena_report import MATCHES, TEAM, positions  # noqa: E402

FLAGS = ("format_error", "timed_out", "token_cap", "fallback", "afk", "coerced")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--teams-only", action="store_true",
                        help="skip spots where the bet came from a house bot, and bets with a house bot still in")
    parser.add_argument("--min-preflop-call", type=float, default=5.0,
                        help="skip preflop calls and folds cheaper than this many big blinds (they are many and small)")
    args = parser.parse_args()

    agg = defaultdict(lambda: [0, 0.0])  # (street, action) facing a bet -> [count, EV bb]
    calls, folds, fails, bets = [], [], [], defaultdict(lambda: [0, 0])
    fail_kind, n_decisions = Counter(), 0
    for path in sorted(MATCHES.glob("m_*.json")):
        match = json.loads(path.read_text(encoding="utf-8"))
        mine = {s["seat"] for s in match["seats"] if (s.get("entry") or {}).get("org_id") == TEAM
                and ((s["entry"].get("deployment") or {}).get("checkpoint_id") == args.checkpoint)}
        if not mine:
            continue
        house = {s["seat"] for s in match["seats"] if (s.get("entry") or {}).get("kind") == "house"}
        for hand in match["hands"]:
            bb = hand.get("big_blind") or 2
            holes = {p["seat"]: p.get("hole_cards") for p in hand["players"]}
            pos = positions([p["seat"] for p in hand["players"]], hand["button_seat"])
            active = set(holes)
            acts = hand["actions"]
            aggressor = None
            for i, a in enumerate(acts):
                seat, act = a["seat"], a["action"]
                if i and a["street"] != acts[i - 1]["street"]:
                    aggressor = None
                skip = args.teams_only and (aggressor in house if (a.get("to_call") or 0) > 0
                                            else bool(house & (active - {seat})))
                if seat in mine and holes.get(seat) and not skip:
                    n_decisions += 1
                    where = f"{path.stem[2:17]} h{hand['hand_number']} {pos.get(seat)} {''.join(holes[seat])} {a['street']}"
                    board = a.get("board") or []
                    opp = [holes[s] for s in active if s != seat and holes.get(s)]
                    to_call, pot = a.get("to_call") or 0, a.get("pot_before") or 0
                    flagged = [f for f in FLAGS if a.get(f)]
                    if flagged:
                        kind = "+".join(flagged)
                        fail_kind[kind] += 1
                        fails.append((where, kind, act, to_call, (a.get("tokens") or {}).get("out"),
                                      a.get("afk_reason") or a.get("error"), a.get("tool_args")))
                    cheap = a["street"] == "preflop" and to_call < args.min_preflop_call * bb
                    if to_call > 0 and opp and act in ("call", "fold") and not cheap:
                        eq = true_equity(holes[seat], opp, board)
                        ev = (eq * (pot + to_call) - to_call) / bb
                        agg[(a["street"], act)][0] += 1
                        agg[(a["street"], act)][1] += ev if act == "call" else -max(ev, 0.0)
                        item = (ev, where, f"eq {eq:.0%} vs {len(opp)}, call {to_call} into {pot}{' [' + '+'.join(flagged) + ']' if flagged else ''}")
                        (calls if act == "call" else folds).append(item)
                    if act in ("bet", "raise") and opp and a["street"] != "preflop":
                        eq = true_equity(holes[seat], opp, board)
                        kind = "bluff <30%" if eq < 0.3 else "value >60%" if eq > 0.6 else "middle"
                        later = [b for b in acts[i + 1:] if b["street"] == a["street"]]
                        took_it = all(b["action"] == "fold" for b in later if b["seat"] != seat) and later
                        bets[(a["street"], kind)][0] += 1
                        bets[(a["street"], kind)][1] += bool(took_it)
                if act == "fold":
                    active.discard(seat)
                if act in ("bet", "raise"):
                    aggressor = seat
    print(f"{args.checkpoint}: {n_decisions} decisions")
    print("\nfacing a bet (EV in bb against the real hands; for folds, the EV of the call it passed up):")
    for (street, act), (n, ev) in sorted(agg.items()):
        print(f"  {street:7} {act:5} {n:5}  total {ev:+8.1f} bb  per decision {ev / n:+.2f}")
    print("\nbets and raises by equity at the time (vs the real hands), and how often everyone folded right away:")
    for (street, kind), (n, won) in sorted(bets.items()):
        print(f"  {street:7} {kind:11} {n:5}  folded out {won / n:.0%}")
    print(f"\nfailed decisions: {sum(fail_kind.values())} of {n_decisions}: {dict(fail_kind)}")
    for f in fails[: args.examples]:
        print(f"  {f[0]} -> played {f[2]} (to call {f[3]}, {f[4]} tokens) reason: {str(f[5])[:120]} args: {str(f[6])[:80]}")
    calls.sort()
    folds.sort()
    print(f"\nworst calls (most negative EV):")
    for ev, where, info in calls[: args.examples]:
        print(f"  {ev:+7.1f} bb  {where}  {info}")
    print(f"\ncostliest folds (passed up the most EV):")
    for ev, where, info in folds[::-1][: args.examples]:
        print(f"  {ev:+7.1f} bb  {where}  {info}")


if __name__ == "__main__":
    main()
