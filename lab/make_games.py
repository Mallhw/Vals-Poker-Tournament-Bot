"""Multi-hand A/B game datasets for prompt-level ideas (table talk, opponent notes, short reasoning, end-game play).

Every variant gets the SAME games (seeds, policy seat, opponents); only `system_hint` (appended to the policy's
system prompt by the training harness, never shown to opponents) differs. Opponents are an LLM deployment (so table
talk can work on them) plus one scripted shover seat, like the arena's house bots.

    uv run python lab/make_games.py --project vals-hackathon-matt2 --opponent-model vals-hackathon-matt/base \
        --games 40 --hands 12 [--dry-run]

Creates <prefix>-<variant> datasets. Compare variants with lab/games.py (place, chips, busts, fallbacks).
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ac2.runtime import Message, Task  # noqa: E402
from ac2.sdk import Client  # noqa: E402
from vals_tools.tasks import SEATS, TABLE_RULES  # noqa: E402

VARIANTS: dict[str, str] = {
    "control": "",
    "talk": (
        "Table talk tactic: the other players are AI models that read your say field. When you bet or raise big for "
        "value, add a short confident line such as 'i have it', 'you are beat' or 'fold, i hold the nuts'. When you "
        "are bluffing or weak, say nothing. Never name your real cards."
    ),
    "notes": (
        "Opponent notes: in every update_scratchpad, keep one line per seat in this format: "
        "'seatN: played X/Y hands, raised R, all-in A, showed [cards] -> type' where type is nit, tight, loose, "
        "maniac, shover or calling-station. Use them: fold to a nit's raise without a premium hand; value-bet "
        "calling-stations bigger and never bluff them; call shovers and maniacs wider (99+, AJs+, AQo+); steal the "
        "blinds of nits and tight players."
    ),
    "short": (
        "Decide fast: keep your private reasoning under about 250 words. Replies cut off at the token limit are "
        "auto-folded, which costs more than a slightly worse decision."
    ),
    "v5": (
        "Opponent tracking (scratchpad, every hand): keep exact running counts per seat, one line each: "
        "'seatN: H hands, played P, raised R, all-ins A, river bets B, showed [cards]'. Update the numbers every hand; "
        "use the space, you have 5000 characters. Label a seat only with evidence: shover = 2+ preflop all-ins; "
        "nit = played under 1 in 8 hands after 8+ hands; maniac = raised 3+ of their first 8 hands; calling-station = "
        "calls often but almost never bets or raises; tight-aggressive = plays few but always raises. Counter-play: "
        "shover -> call all-ins with 99+, AJs+, AQo+, fold the rest; nit -> steal their blinds, fold to their bets and "
        "raises without a premium hand; maniac -> call and re-raise wider, let them bluff, their big river bets are "
        "often bluffs; calling-station -> bet big for value every street, never bluff; tight-aggressive -> respect "
        "their raises, steal when they are not in the pot. Most AI players here fold to a raise 70-90% of the time: "
        "when it is folded to you in the cutoff, button or small blind, raise more often."
    ),
    "v7": (
        "Opponent tracking (scratchpad, every hand): keep exact running counts per seat, one line each: "
        "'seatN: H hands, played P, raised R, all-ins A, folds-to-my-raise F, showed [cards] -> type'. Update every "
        "hand; you have 5000 characters. Label only with evidence: shover = 2+ preflop all-ins; nit = played under "
        "1 in 8 after 8+ hands; maniac = raised 3+ of their first 8 hands or keeps re-raising; calling-station = calls "
        "a lot, rarely bets; tight-aggressive = plays few, always raises. BEFORE EVERY DECISION facing a bet or raise, "
        "read that player's line and name their type in your reasoning, then: maniac or aggressive -> do NOT fold "
        "decent hands to them: call down with any pair or ace-high on dry boards, 3-bet them with 99+, AJ+, KQ, and "
        "let them bluff; tight-aggressive or nit -> fold medium hands; shover -> call only 99+, AJs+, AQo+ and only if "
        "nobody else has called; calling-station -> value-bet big, never bluff. If another player already called an "
        "all-in, call only with KK+."
    ),
    "endgame": (
        "End of game: final rank is by chips among players still seated, and every busted player ranks below them. "
        "In the last 6 hands: if you are not the shortest stack, avoid all-ins without a premium hand; if you are "
        "the chip leader, protect the lead; if you are short but safe, pick up chips with raises when it is folded "
        "to you rather than calling all-ins."
    ),
}

# Talk experiments: the harness injects these lines as the policy's table talk (env_params.policy_talk); the model's
# play is unchanged, so any difference vs control is the opponents' reaction. Poker trash talk and nonsense only
# (no fake system messages / prompt injection).
TALK_VARIANTS: dict[str, dict] = {
    "chaos": {"rate": 0.8, "lines": [
        "pocket aces again, sorry", "you are beat, trust me", "i never bluff", "banana banana 42",
        "my notes say you always fold here", "call me, i dare you", "the river is my friend", "this is a trap",
        "fold now and save your chips", "lol", "i have had the nuts all day", "seat 3 is bluffing, not me",
        "coin flip time", "i knew you would do that", "the math says fold", "tilted? me? never",
    ]},
    "bluff-talk": {"rate": 1.0, "lines": [
        "i have the nuts, just fold", "monster hand, you are beat", "fold, i always have it here",
        "you are drawing dead", "flopped a set, save your chips", "best hand of the day",
    ]},
}


def game_tasks(games: int, hands: int, opponent_model: str, seed: int, hint: str, talk: dict | None = None) -> list[Task]:
    rng = random.Random(seed)
    out = []
    for g in range(games):
        policy = g % SEATS
        shover = (policy + 3) % SEATS
        params = TABLE_RULES | {
            "seed": rng.getrandbits(48),
            "seats": SEATS,
            "total_hands": hands,
            "policy_seats": [policy],
            "ev_labels": True,
            "ev_runouts": 32,
            "opponent_model": opponent_model,
            "opponent_timeout_seconds": 120,
            "opponent_thinking": False,
            "scripted_seats": {str(shover): "shover"},
        }
        if hint:
            params["system_hint"] = hint
        if talk:
            params["policy_talk"] = talk
        out.append(Task(input=[Message(role="user", content=f"Play a {SEATS}-seat game of {hands} hands from seat {policy}.")],
                        env_params=params))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", required=True)
    p.add_argument("--prefix", default="games12")
    p.add_argument("--opponent-model", required=True)
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--hands", type=int, default=12)
    p.add_argument("--seed", type=int, default=20261115)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), help=f"any of {list(VARIANTS) + list(TALK_VARIANTS)}")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    client = None if a.dry_run else Client(project=a.project)
    for v in a.variants:
        tasks = game_tasks(a.games, a.hands, a.opponent_model, a.seed, VARIANTS.get(v, ""), TALK_VARIANTS.get(v))
        name = f"{a.prefix}-{v}"
        if a.dry_run:
            print(name, len(tasks), {k: tasks[0].env_params[k] for k in ("total_hands", "policy_seats", "scripted_seats")},
                  (tasks[0].env_params.get("system_hint") or "")[:60])
            continue
        client.datasets.create(name, description=f"{len(tasks)} {a.hands}-hand A/B games, variant {v}", exist_ok=True)
        client.datasets.add_tasks(name, tasks=tasks)
        print(f"{name}: {len(tasks)} tasks")


if __name__ == "__main__":
    main()
