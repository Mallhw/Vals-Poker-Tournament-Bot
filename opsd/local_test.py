"""Paired local test of the OPSD hint: the same deals with and without it, against scripted bots.

Seat 0 is any OpenAI-compatible model (default: a local llama-server); the other five seats are the harness's
scripted players, two of them maniacs that raise big, so stack-off decisions come up often. Each seed is played once
per variant with identical cards and opponents, so the difference between variants is mostly the hint, not luck.
"hint" appends `opsd/hint.py`'s HINT to the system prompt exactly as OPSD's teacher sees it; "none" is the
event prompt. The official harness code is not changed.

    cd <repo>
    uv run --project harness python opsd/local_test.py --seeds 1 2 3 --hands 10 --no-thinking
    uv run --project harness python opsd/local_test.py --base-url http://127.0.0.1:8080/v1 --seeds 1 2

A local model is not the event model (Qwen3.6-35B-A3B), so treat results as a check that the hint is followed and
causes no format errors, not as a measure of arena strength.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from poker_harness.completions import cleanup_params, relay_completion, table_rules, with_preset
from poker_harness.engine import ScriptedPlayer
from poker_harness.game import play_session
from poker_harness.prompts import talk_fields
from poker_harness.seat import ModelSeat

sys.path.insert(0, str(Path(__file__).parent))
from hint import HINT, HINTS, SEPARATOR  # noqa: E402

OPPONENTS = ["maniac", "calling_station", "nit", "maniac", "random"]


def with_hint(complete, text: str = HINT):
    """Wrap a `complete(messages)` so the system message gets the hint appended (OPSD teacher's view)."""

    async def hinted(messages: list[dict[str, Any]]):
        if messages and messages[0].get("role") == "system":
            messages = [{**messages[0], "content": messages[0]["content"] + SEPARATOR + text}, *messages[1:]]
        return await complete(messages)

    return hinted


def seat_stats(log: dict[str, Any], seat: str = "seat0") -> dict[str, Any]:
    big_blind = log["session_info"]["big_blind"]
    start = log["session_info"]["starting_stacks"][log["session_info"]["player_names"].index(seat)]
    hands = [h for h in log["hands"] if seat in h["starting_state"]]
    mine = [a for h in hands for a in h["actions"] if a["player"] == seat]
    vpip = sum(
        any(a["player"] == seat and a["street"] == "preflop" and a["action"]["type"] in ("call", "raise", "bet")
            for a in h["actions"])
        for h in hands
    )
    big_calls = []
    for h in hands:
        for a in h["actions"]:
            if a["player"] != seat or a["action"]["type"] != "call":
                continue
            before = a["stack_after"] - a["chip_change"]
            if before > 0 and a["call_amount"] >= 0.4 * before:
                big_calls.append({"hand": h["hand_number"], "hole": h["starting_state"][seat]["hole_cards"],
                                  "street": a["street"], "call": a["call_amount"], "stack": before})
    final = log["final_stacks"][seat]
    return {
        "hands": len(hands),
        "net_bb": (final - start) / big_blind,
        "busted": final <= 0,
        "decisions": len(mine),
        "format_errors": sum(bool(a["model_response"].get("format_error")) for a in mine),
        "vpip": vpip / max(1, len(hands)),
        "big_calls": big_calls,
    }


async def play_one(args: argparse.Namespace, seed: int, variant: str, client: httpx.AsyncClient) -> dict[str, Any]:
    params = with_preset({"preset": "event", "seed": seed, "submit_fields": ["action", "amount"]})
    params["total_hands"] = args.hands
    if args.no_thinking:
        params["chat_template_kwargs"] = {"enable_thinking": False}
    if args.max_tokens:
        params["max_completion_tokens"] = args.max_tokens
    rules = table_rules(params)
    if rules.table_talk:
        params["submit_fields"] = talk_fields(params["submit_fields"])
    mine = params | {"model_label": args.model}
    decide = relay_completion(client, args.base_url, args.api_key, mine)
    cleanup = relay_completion(client, args.base_url, args.api_key, cleanup_params(mine))
    if variant != "none":
        text = HINT if variant == "hint" else HINTS[variant]
        decide, cleanup = with_hint(decide, text), with_hint(cleanup, text)
    players = [ModelSeat("seat0", decide, rules, model=args.model, cleanup_complete=cleanup)]
    players += [ScriptedPlayer(f"seat{i + 1}", kind, seed) for i, kind in enumerate(args.opponents)]
    t0 = time.monotonic()
    log = await play_session(f"local-{variant}-{seed}", seed, players, rules,
                             on_hand=lambda lg: print(f"  [{variant} seed {seed}] hand {len(lg['hands'])}", flush=True))
    stats = seat_stats(log) | {"seed": seed, "variant": variant, "minutes": round((time.monotonic() - t0) / 60, 1)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{variant}-{seed}.json").write_text(json.dumps(log))
    return stats


async def main_async(args: argparse.Namespace) -> None:
    results = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(1800.0, connect=30.0)) as client:
        for seed in args.seeds:
            for variant in args.variants:
                stats = await play_one(args, seed, variant, client)
                print(json.dumps({k: v for k, v in stats.items() if k != "big_calls"}), flush=True)
                for call in stats["big_calls"]:
                    print(f"    big call: {call}", flush=True)
                results.append(stats)
    (args.out / "summary.json").write_text(json.dumps(results, indent=2))
    print("\nvariant  games  net_bb  busts  format_errors/decisions  vpip  big_calls")
    for variant in args.variants:
        rs = [r for r in results if r["variant"] == variant]
        dec = sum(r["decisions"] for r in rs)
        print(f"{variant:8} {len(rs):5} {sum(r['net_bb'] for r in rs):+7.1f} {sum(r['busted'] for r in rs):6} "
              f"{sum(r['format_errors'] for r in rs):>10}/{dec:<6} {sum(r['vpip'] for r in rs) / len(rs):10.0%} "
              f"{sum(len(r['big_calls']) for r in rs):9}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1", help="OpenAI-compatible endpoint")
    parser.add_argument("--api-key", default="local", help="bearer token for the endpoint")
    parser.add_argument("--model", default="local", help="model name sent in requests")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--hands", type=int, default=10, help="hands per game (the event plays 30)")
    parser.add_argument("--variants", nargs="+", default=["none", "hint"], choices=["none", "hint", *HINTS],
                        help="'none', 'hint' (opsd/hint.py's HINT) or a named version from HINTS")
    parser.add_argument("--opponents", nargs=5, default=OPPONENTS,
                        help="archetypes for seats 1-5 (random, maniac, nit, calling_station, shover)")
    parser.add_argument("--no-thinking", action="store_true", help="turn off Qwen thinking (much faster locally)")
    parser.add_argument("--max-tokens", type=int, help="override the event's 5,000-token cap per decision")
    parser.add_argument("--out", type=Path, default=Path("runs/local_test"))
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
