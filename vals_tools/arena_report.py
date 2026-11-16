"""Stats for one arena entry next to the field, from the match files `vals_tools.arena_matches` downloads.

    uv run python -m vals_tools.arena_report --checkpoint ckpt_6c4893ae55d6
    uv run python -m vals_tools.arena_report --checkpoint ckpt_6c4893ae55d6 --worst 15 --hands-out data/arena/hands.txt

"ours" is the team's seat whose deployment came from `--checkpoint` (all of the team's seats without it). The field
is every other seat in the same matches: house bots and other teams, pooled and per entry. Preflop: VPIP, PFR,
limps, 3-bets, folds to 3-bets, by position too. Postflop: c-bets, folds to flop bets, aggression, showdowns. Our
decisions: failures (format error, timeout, token cap, fallback) and output tokens. Then our biggest losses and
wins, each as a full action line, to read before calling anything a leak: these are small samples.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATCHES = ROOT / "data" / "arena" / "matches"
TEAM = "org_9197691edc5e"
AGGRESSIVE = {"bet", "raise"}
VOLUNTARY = {"call", "bet", "raise"}


def positions(seats: list[int], button: int) -> dict[int, str]:
    """Position names for the seats dealt in, clockwise from the small blind."""
    order = sorted(seats, key=lambda s: (s - button - 1) % 6)
    if len(order) == 2:
        return {order[1]: "BTN", order[0]: "BB"} if (order[1] - button) % 6 == 0 else {order[0]: "SB", order[1]: "BB"}
    names = {order[0]: "SB", order[1]: "BB", order[-1]: "BTN"}
    rest = order[2:-1]
    for seat, name in zip(reversed(rest), ["CO", "HJ", "UTG", "UTG", "UTG"]):
        names[seat] = name
    return names


class Stats:
    def __init__(self) -> None:
        self.c = Counter()
        self.by_pos = defaultdict(Counter)
        self.tokens: list[int] = []

    def rate(self, num: str, den: str) -> str:
        return f"{self.c[num] / self.c[den]:.0%} ({self.c[den]})" if self.c[den] else "-"


def hand_stats(hand: dict, stats_for) -> list[tuple[int, str, str, float]]:
    """Update each seat's Stats; return (seat, position, line, net bb) for every seat dealt in."""
    bb = hand.get("big_blind") or 2
    seats = [p["seat"] for p in hand["players"]]
    pos = positions(seats, hand["button_seat"])
    holes = {p["seat"]: p.get("hole_cards") for p in hand["players"]}
    deltas = {int(k): v for k, v in (hand.get("deltas") or {}).items()}
    folded, first_pre, raises_pre, faced_3bet = set(), {}, 0, set()
    opener, last_raiser_pre = None, None
    flop_bet_seen, flop_first = False, {}
    lines = defaultdict(list)
    for a in hand["actions"]:
        seat, street, act = a["seat"], a["street"], a["action"]
        st = stats_for(seat)
        if st is not None:
            st.c["decisions"] += 1
            for flag in ("format_error", "timed_out", "token_cap", "fallback", "afk"):
                st.c[flag] += bool(a.get(flag))
            out = (a.get("tokens") or {}).get("out")
            if out:
                st.tokens.append(out)
            st.c["talk"] += bool(a.get("say"))
        amount = f" {a['amount']}" if a.get("amount") else ""
        say = f' "{a["say"]}"' if a.get("say") else ""
        lines[street].append(f"{pos.get(seat, seat)}{'*' if st is not None and st.c.get('_ours') else ''} {act}{amount}{say}")
        if street == "preflop":
            if st is not None and seat not in first_pre:
                first_pre[seat] = act
                st.c["pre_first"] += 1
                if raises_pre == 0 and act == "call" and pos.get(seat) != "BB":
                    st.c["limp"] += 1
                if raises_pre == 1:
                    st.c["face_open"] += 1
                    st.c["threebet"] += act == "raise"
                    st.c["fold_to_open"] += act == "fold"
            if st is not None and seat == opener and raises_pre == 2 and seat not in faced_3bet:
                faced_3bet.add(seat)
                st.c["face_3bet"] += 1
                st.c["fold_to_3bet"] += act == "fold"
            if act in AGGRESSIVE:
                raises_pre += 1
                last_raiser_pre = seat
                if raises_pre == 1:
                    opener = seat
        elif street == "flop":
            if st is not None and seat not in flop_first:
                flop_first[seat] = act
                if seat == last_raiser_pre and not flop_bet_seen:
                    st.c["cbet_chance"] += 1
                    st.c["cbet"] += act in AGGRESSIVE
                if flop_bet_seen and (a.get("to_call") or 0) > 0:
                    st.c["face_flop_bet"] += 1
                    st.c["fold_flop_bet"] += act == "fold"
            if act in AGGRESSIVE:
                flop_bet_seen = True
        if street != "preflop" and st is not None:
            st.c["post_aggr"] += act in AGGRESSIVE
            st.c["post_call"] += act == "call"
        if street == "river" and st is not None and act == "call" and (a.get("to_call") or 0) > 0:
            st.c["river_call"] += 1
            st.c["river_call_won"] += deltas.get(seat, 0) > 0
        if act == "fold":
            folded.add(seat)
    board = hand.get("board") or []
    out = []
    for seat in seats:
        st = stats_for(seat)
        if st is None:
            continue
        p = pos.get(seat, "?")
        net = deltas.get(seat, 0) / bb
        st.c["hands"] += 1
        st.c["net_bb_x100"] += round(net * 100)
        st.by_pos[p]["hands"] += 1
        st.by_pos[p]["net_bb_x100"] += round(net * 100)
        vpip = first_pre.get(seat) in VOLUNTARY or any(
            a["seat"] == seat and a["street"] == "preflop" and a["action"] in VOLUNTARY for a in hand["actions"])
        pfr = any(a["seat"] == seat and a["street"] == "preflop" and a["action"] in AGGRESSIVE for a in hand["actions"])
        st.c["vpip"] += vpip
        st.c["pfr"] += pfr
        st.by_pos[p]["vpip"] += vpip
        st.by_pos[p]["pfr"] += pfr
        saw_flop = len(board) >= 3 and not any(
            a["seat"] == seat and a["action"] == "fold" and a["street"] == "preflop" for a in hand["actions"])
        if saw_flop:
            st.c["saw_flop"] += 1
            if hand.get("showdown") and seat not in folded:
                st.c["wtsd"] += 1
                st.c["wsd"] += deltas.get(seat, 0) > 0
        text = " | ".join(f"{s}: " + ", ".join(lines[s]) for s in ("preflop", "flop", "turn", "river") if lines[s])
        shown = hand.get("shown") or {}
        out.append((seat, p, f"{p} {''.join(holes.get(seat) or [])} board {' '.join(board) or '-'} :: {text}"
                    + (f" || shown {shown}" if shown else ""), net))
    return out


def report(st: Stats, label: str) -> str:
    c = st.c
    bb100 = c["net_bb_x100"] / c["hands"] if c["hands"] else 0
    tok = f"{statistics.median(st.tokens):.0f}/{sorted(st.tokens)[int(0.9 * (len(st.tokens) - 1))]}" if st.tokens else "-"
    fails = sum(c[f] for f in ("format_error", "timed_out", "token_cap", "fallback", "afk"))
    return (f"{label[:26]:26} {c['hands']:5} {bb100:+7.1f} {st.rate('vpip', 'hands'):>10} {st.rate('pfr', 'hands'):>10} "
            f"{st.rate('limp', 'pre_first'):>10} {st.rate('threebet', 'face_open'):>10} {st.rate('fold_to_3bet', 'face_3bet'):>10} "
            f"{st.rate('cbet', 'cbet_chance'):>10} {st.rate('fold_flop_bet', 'face_flop_bet'):>10} "
            f"{(c['post_aggr'] / c['post_call']) if c['post_call'] else 0:4.1f} {st.rate('wtsd', 'saw_flop'):>10} "
            f"{st.rate('wsd', 'wtsd'):>9} {st.rate('river_call_won', 'river_call'):>9} {fails / c['decisions'] if c['decisions'] else 0:5.1%} {tok:>9}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", help="our seat's checkpoint id (default: every seat of the team)")
    parser.add_argument("--team", default=TEAM)
    parser.add_argument("--matches", type=Path, default=MATCHES)
    parser.add_argument("--worst", type=int, default=12, help="how many of our biggest losses (and wins) to print")
    parser.add_argument("--hands-out", type=Path, help="write every hand of ours, one line each, to this file")
    args = parser.parse_args()

    ours, field, house, per_entry = Stats(), Stats(), Stats(), defaultdict(Stats)
    ours.c["_ours"] = 1
    our_hands, n_matches = [], 0
    for path in sorted(args.matches.glob("m_*.json")):
        match = json.loads(path.read_text(encoding="utf-8"))
        entries = {s["seat"]: s.get("entry") or {} for s in match["seats"]}
        mine = {seat for seat, e in entries.items() if e.get("org_id") == args.team
                and (not args.checkpoint or ((e.get("deployment") or {}).get("checkpoint_id") == args.checkpoint))}
        if not mine:
            continue
        n_matches += 1
        for hand in match["hands"]:
            def stats_for(seat: int) -> Stats | None:
                if seat in mine:
                    return ours
                e = entries.get(seat, {})
                if e.get("org_id") == args.team:
                    return None  # our other entries, if any
                return per_entry[e.get("name", "?")]
            for seat, p, line, net in hand_stats(hand, stats_for):
                if seat in mine:
                    our_hands.append((net, f"{path.stem} h{hand['hand_number']}", line))
    for st in per_entry.values():
        for k, v in st.c.items():
            field.c[k] += v
        field.tokens += st.tokens
    header = (f"{'entry':26} {'hands':>5} {'bb/100':>7} {'VPIP':>10} {'PFR':>10} {'limp':>10} {'3bet':>10} "
              f"{'fold v 3b':>10} {'cbet':>10} {'fold v fb':>10} {'AF':>4} {'WTSD':>10} {'W$SD':>9} {'rvr call':>9} "
              f"{'fails':>5} {'tok m/p90':>9}")
    print(f"{n_matches} matches with our seat" + (f" ({args.checkpoint})" if args.checkpoint else ""))
    print(header)
    print(report(ours, "OURS"))
    print(report(field, "field (all others)"))
    for name, st in sorted(per_entry.items(), key=lambda kv: -kv[1].c["hands"]):
        print(report(st, name))
    print("\nours by position: " + "  ".join(
        f"{p} {c['hands']}h {c['net_bb_x100'] / c['hands'] if c['hands'] else 0:+.0f}bb/100 "
        f"vpip {c['vpip'] / c['hands'] if c['hands'] else 0:.0%} pfr {c['pfr'] / c['hands'] if c['hands'] else 0:.0%}"
        for p, c in sorted(ours.by_pos.items())))
    print(f"our talk: {ours.c['talk']} of {ours.c['decisions']} decisions")
    our_hands.sort()
    print(f"\n=== our {args.worst} biggest losses (bb)")
    for net, where, line in our_hands[: args.worst]:
        print(f"{net:+7.1f} {where} {line[:700]}")
    print(f"\n=== our {min(5, args.worst)} biggest wins (bb)")
    for net, where, line in our_hands[::-1][: min(5, args.worst)]:
        print(f"{net:+7.1f} {where} {line[:500]}")
    if args.hands_out:
        args.hands_out.write_text("\n".join(f"{net:+.1f}\t{where}\t{line}" for net, where, line in our_hands), encoding="utf-8")


if __name__ == "__main__":
    main()
