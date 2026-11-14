"""Play one event game on your machine against a deployed model, to check it before you enter it in the arena.

    uv run python play_local.py --model vals-hackathon/entry                    # your deployment vs the house model
    uv run python play_local.py --model vals-hackathon/entry --opponents bots   # vs scripted bots (no opponent calls)
    uv run python play_local.py --model vals-hackathon/entry --hands 6          # a shorter game
    uv run python play_local.py --model vals-hackathon/base --hint-file ../opsd/hint.py --compare --opponents bots

`--hint-file` (a text file, or a .py file defining HINT, such as ../opsd/hint.py) appends that hint to seat 0's
system prompt, as OPSD's teacher sees it; `--compare` plays the same seed without the hint and then with it. One game
is a quick check of format, token use and style, not a verdict: compare hints on many hands with hinted datasets
(rl/make_hint_dataset.py) and `vals_tools.score`. Scripted bots (`--opponents bots`) make the two games more alike.

The game runs under the event's rules (the "event" preset: 6 seats, talk, carried stacks, the scratchpad and its
cleanup turn, 5,000 tokens per decision and 10,000 per cleanup turn, thinking on), with your model in seat 0. Every
seat calls its model directly through the AC2 inference API with your `AC2_API_KEY`; nothing goes through a
connector, and no AC2 job runs. At the end it prints how many of your model's decisions failed (no usable
`submit_action` call, an illegal move, or a reply cut off at the cap) and writes the full game log.

The arena's entry probe only checks that your deployment answers a plain text request. A deployment that can't call
`submit_action` passes the probe and then fails every decision: its seats check or fold, and the arena pauses it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import runpy
from collections import Counter
from pathlib import Path

import httpx

from poker_harness.completions import (
    GATEWAY_URL,
    cleanup_params,
    opponent_params,
    relay_completion,
    table_rules,
    with_preset,
)
from poker_harness.engine import ScriptedPlayer
from poker_harness.game import play_session
from poker_harness.prompts import talk_fields
from poker_harness.seat import ModelSeat

HOUSE_MODEL = "vals-hackathon/ac-vals-model"


def read_hint(path: Path) -> str:
    """A hint file's text, or HINT from a .py file (such as opsd/hint.py)."""
    hint = runpy.run_path(str(path))["HINT"] if path.suffix == ".py" else path.read_text(encoding="utf-8")
    if not isinstance(hint, str) or not hint.strip():
        raise SystemExit(f"{path} has no hint text")
    return hint.strip()


def preflop_stats(log: dict, seat: str = "seat0") -> tuple[int, float, float]:
    """Hands dealt, VPIP (voluntarily put chips in preflop) and PFR (raised preflop), as fractions of hands dealt."""
    dealt = [h for h in log["hands"] if seat in h["starting_state"]]
    moves = [[a["action"]["type"] for a in h["actions"] if a["player"] == seat and a["street"] == "preflop"] for h in dealt]
    if not dealt:
        return 0, 0.0, 0.0
    vpip = sum(any(m in ("call", "bet", "raise") for m in hand) for hand in moves) / len(dealt)
    pfr = sum("raise" in hand for hand in moves) / len(dealt)
    return len(dealt), vpip, pfr


async def play(args: argparse.Namespace, seed: int, hint: str = "") -> dict:
    key = os.environ.get("AC2_API_KEY")
    if not key:
        raise SystemExit("export AC2_API_KEY first")
    params = with_preset({"preset": "event", "seed": seed, "submit_fields": ["action", "amount"]})
    if args.hands:
        params["total_hands"] = args.hands
    rules = table_rules(params)
    if rules.table_talk:  # every seat's tool gains `say`, as on the harness server
        params["submit_fields"] = talk_fields(params["submit_fields"])
    timeout = httpx.Timeout(600.0, connect=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        mine = params | {"model_label": args.model}
        players = [
            ModelSeat(
                "seat0",
                relay_completion(client, GATEWAY_URL, key, mine),
                rules,
                model=args.model,
                cleanup_complete=relay_completion(client, GATEWAY_URL, key, cleanup_params(mine)),
                system_hint=hint,
            )
        ]
        for i in range(1, rules.seats):
            if args.opponents == "bots":
                players.append(ScriptedPlayer(f"seat{i}", random.Random(seed + i).choice(["random", "calling_station"]), seed))
            else:
                theirs = opponent_params(params | {"opponent_model": args.opponent_model})
                players.append(
                    ModelSeat(
                        f"seat{i}",
                        relay_completion(client, GATEWAY_URL, key, theirs),
                        rules,
                        model=args.opponent_model,
                        cleanup_complete=relay_completion(client, GATEWAY_URL, key, cleanup_params(theirs)),
                    )
                )
        game_id = f"local-{seed}-hint" if hint else f"local-{seed}"
        return await play_session(game_id, seed, players, rules, on_hand=lambda log: print(
            f"hand {len(log['hands'])} done", flush=True
        ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="your deployment, <project>/<name>")
    parser.add_argument("--opponents", choices=["house", "bots"], default="house")
    parser.add_argument("--opponent-model", default=HOUSE_MODEL, help="model for the other seats with --opponents house")
    parser.add_argument("--hands", type=int, help="hands in the game (default: the event's 30)")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--hint-file", type=Path, help="a hint to append to seat 0's system prompt (.txt, or .py with HINT)")
    parser.add_argument("--compare", action="store_true", help="play the same seed without the hint, then with it")
    args = parser.parse_args()
    if args.compare and not args.hint_file:
        parser.error("--compare needs --hint-file")

    hint = read_hint(args.hint_file) if args.hint_file else ""
    seed = args.seed if args.seed is not None else random.getrandbits(48)
    games = [("without the hint", ""), ("with the hint", hint)] if args.compare else [("", hint)]
    nets, all_failed = [], False
    for label, text in games:
        if label:
            print(f"\n=== {label} (seed {seed})", flush=True)
        log = asyncio.run(play(args, seed, text))
        net, failed = report(args, log)
        nets.append(net)
        all_failed = all_failed or failed
    if args.compare:
        print(f"\nhint - no hint: {nets[1] - nets[0]:+.1f} BB over one game (noisy: a check, not a verdict)")
    if all_failed:
        raise SystemExit("every decision failed: don't submit this deployment until it calls submit_action")


def report(args: argparse.Namespace, log: dict) -> tuple[float, bool]:
    """Print one game's summary and write its log; returns seat 0's net BB and whether every decision failed."""
    mine = [a for hand in log["hands"] for a in hand["actions"] if a["player"] == "seat0"]
    errors = Counter(a["model_response"].get("error") or "?" for a in mine if a["model_response"].get("format_error"))
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{log['game_id']}.json"
    path.write_text(json.dumps(log, indent=1))
    big_blind = log["session_info"]["big_blind"]
    net = (log["final_stacks"]["seat0"] - 200) / big_blind
    print(f"\n{len(log['hands'])} hands, {len(mine)} decisions by {args.model}, {sum(errors.values())} failed")
    for error, count in errors.most_common():
        print(f"  {count} x {error}")
    dealt, vpip, pfr = preflop_stats(log)
    tokens = [a["model_response"]["out_tokens"] for a in mine if a["model_response"].get("out_tokens")]
    mean_tokens = f"{sum(tokens) / len(tokens):.0f}" if tokens else "?"
    print(f"seat0 preflop over {dealt} hands: VPIP {vpip:.0%}, PFR {pfr:.0%}; {mean_tokens} output tokens per decision")
    print(f"seat0 net: {net:+.1f} BB; log: {path}")
    return net, bool(mine) and sum(errors.values()) == len(mine)


if __name__ == "__main__":
    main()
