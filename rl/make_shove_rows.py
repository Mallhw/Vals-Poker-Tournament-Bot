"""PokerBench-format preflop spots facing an all-in shove, labeled for an elimination event, not a cash game.

The arena showed our checkpoints calling preflop shoves with hands like K7o, J7o and A3o, which have well under the
~49% equity an even-stack call needs against what the shovers actually hold. PokerBench's shove spots are deep 4-bet
and 5-bet pots where calling is often right, so solver training doesn't teach this. Here a seat open-shoves (all in
first, everyone before it folded) and the hero, later in position, decides: call if the hero's hand has at least
`--margin` equity (default 0.55: busting ranks last, so a call needs more than a coin flip) against the shovers' actual
hands in data/arena/matches (real team entries only, not ours and not house bots), else fold.

    cd harness && uv run python ../rl/make_shove_rows.py --out ../data/pokerbench/round4
    uv run python pokerbench_convert.py ../data/pokerbench/round4/shove_train.csv --out ../data/pokerbench/round4/shove_train.jsonl

Writes shove_train.csv and a disjoint shove_heldout.csv (a different seed), plus shove_equity.json (equity per hand class).
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness" / "src"))
TEAM = "org_9197691edc5e"
RANKS = "23456789TJQKA"
SUITS = "cdhs"
ORDER = ["UTG", "HJ", "CO", "BTN", "SB", "BB"]


def hand_class(cards) -> str:
    (r1, s1), (r2, s2) = sorted(cards, key=lambda c: RANKS.index(c[0]), reverse=True)
    return r1 + r2 if r1 == r2 else f"{r1}{r2}{'s' if s1 == s2 else 'o'}"


def arena_shoves(matches: str) -> list[list[str]]:
    """Hole cards of every preflop shove (a raise to >= 90% of the raiser's stack) by a real team entry not ours."""
    out = []
    for path in glob.glob(f"{matches}/m_*.json"):
        m = json.loads(Path(path).read_text(encoding="utf-8"))
        entries = {s["seat"]: s.get("entry") or {} for s in m["seats"]}
        for h in m["hands"]:
            cards = {p["seat"]: p.get("hole_cards") for p in h["players"]}
            for a in h["actions"]:
                if a["street"] != "preflop" or a["action"] not in ("raise", "bet", "allin"):
                    continue
                stack = (a.get("stack_after") or 0) + abs(a.get("chip_change") or 0)
                e = entries.get(a["seat"], {})
                if stack > 20 and (a.get("stack_after") or 0) <= 0.1 * stack and e.get("kind") == "team" \
                        and e.get("org_id") != TEAM and cards.get(a["seat"]):
                    out.append(cards[a["seat"]])
    return out


def class_equities(shoves: list[list[str]], samples: int, seed: int) -> dict[str, float]:
    """Each hand class's equity against the shove range, heads-up, one representative combo per class (the harness's
    pure-Python hand evaluator: fast, unlike pokerkit's per-call overhead)."""
    from poker_harness.engine.style_bot import hand_value

    rng = random.Random(seed)
    deck = [r + s for r in RANKS for s in SUITS]
    out = {}
    for i, r1 in enumerate(RANKS):
        for r2 in RANKS[: i + 1]:
            for suited in ((False,) if r1 == r2 else (True, False)):
                hole = [r1 + "s", r2 + ("s" if suited else "h")] if r1 != r2 else [r1 + "s", r2 + "h"]
                won = total = 0.0
                for villain in rng.sample(shoves, min(len(shoves), 60)):
                    if set(villain) & set(hole):
                        continue
                    rest = [c for c in deck if c not in hole and c not in villain]
                    for _ in range(samples):
                        board = rng.sample(rest, 5)
                        mine, theirs = hand_value(hole + board), hand_value(villain + board)
                        won += 1.0 if mine > theirs else 0.5 if mine == theirs else 0.0
                        total += 1
                out[hand_class(hole)] = round(won / total, 4)
    return out


def rows(count: int, rng: random.Random, equity: dict[str, float], margin: float) -> list[dict[str, str]]:
    deck = [r + s for r in RANKS for s in SUITS]
    out = []
    while len(out) < count:
        shover = rng.choice(ORDER[:5])
        hero = rng.choice(ORDER[ORDER.index(shover) + 1:])
        cards = rng.sample(deck, 2)
        eq = equity[hand_class(cards)]
        pot = 101.0 if shover == "SB" else 101.5  # the shove to 100 BB plus the blinds
        out.append({"prev_line": f"{shover}/allin", "hero_pos": hero, "hero_holding": "".join(cards),
                    "correct_decision": "call" if eq >= margin else "fold", "pot_size": f"{pot:g}",
                    "shove_equity": f"{eq:.4f}"})
    return out


def write(path: Path, data: list[dict[str, str]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=ROOT / "data" / "pokerbench" / "round4")
    p.add_argument("--matches", default=str(ROOT / "data" / "arena" / "matches"))
    p.add_argument("--train", type=int, default=900)
    p.add_argument("--heldout", type=int, default=200)
    p.add_argument("--margin", type=float, default=0.55)
    p.add_argument("--samples", type=int, default=150, help="runouts per matchup")
    p.add_argument("--seed", type=int, default=67)
    args = p.parse_args()
    shoves = arena_shoves(args.matches)
    print(f"{len(shoves)} shoves by real team entries in the arena data")
    cache = args.out / "shove_equity.json"
    equity = json.loads(cache.read_text()) if cache.exists() else class_equities(shoves, args.samples, args.seed)
    cache.write_text(json.dumps(equity, indent=0))
    calls = sorted((c for c, e in equity.items() if e >= args.margin), key=equity.get, reverse=True)
    share = sum(6 if len(c) == 2 else 4 if c.endswith("s") else 12 for c in calls) / 1326
    print(f"call at >= {args.margin:.0%} equity: {len(calls)} classes, {share:.1%} of hands; weakest calls: {calls[-8:]}")
    train = rows(args.train, random.Random(args.seed), equity, args.margin)
    held = rows(args.heldout, random.Random(args.seed + 1), equity, args.margin)
    write(args.out / "shove_train.csv", train)
    write(args.out / "shove_heldout.csv", held)
    print(f"shove spots: {len(train)} train ({sum(r['correct_decision'] == 'call' for r in train)} calls), "
          f"{len(held)} held out")


if __name__ == "__main__":
    main()
