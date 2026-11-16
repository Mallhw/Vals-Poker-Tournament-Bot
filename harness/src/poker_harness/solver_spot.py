"""Solver spots: one PokerBench decision per rollout, rewarded for matching the solver's play.

A task with `env_params.pokerbench_spot` (built by `rl/make_spot_dataset.py` from `harness/pokerbench_convert.py`
output) carries the exact conversation the event table shows the trained seat at that decision, the legal actions
and the solver's play. The harness sends the conversation to the policy once, through the rollout's relay with the
event's tools, and judges the reply:

- `clean`: exactly one `submit_action` call and nothing else, not cut at the token cap, legal under the event's strict
  rules (anything else would be played as check or fold at the event);
- `agrees`: the action matches the solver's, bet and raise amounts within 25% (or 2 chips); `type_agrees`: the
  action matches but the amount does not.

The reward uses the sample grader's 0-11 scale, so GRPO's settings carry over. A format failure scores 0. A clean
reply scores 1 + 10 x similarity, where similarity is continuous in how far the reply's action is from the solver's:
each action is the chips it adds now (fold and check add 0, a call adds the call amount, a raise to X adds X minus what
the hero has in already), the distance is the difference as a share of the pot after a call, and similarity is
1 / (1 + (distance / 0.25)^2). So the solver's play scores 11, a raise 10% off its size about 10.7, calling where the
solver min-raises about 6, folding where it calls a tiny bet nearly 11, and opening where it folds about 1.6. Replies
longer than 2,500 output tokens lose up to 3 (never below 1), so training does not buy agreement with longer thinking.
`agrees` and `type_agrees` stay as yes/no measures for reporting.

A spot with `"reward": "action_first"` scores the action before the size: a different action from the solver's has
similarity 0 (a clean reply scores 1), and the solver's action scores by the same chips distance as above, so only its
size matters. Under the default chips distance a call sits between folding and raising, so when the policy is unsure
whether the solver raises or folds, calling scores best on average (calling where it min-raises about 6, folding
about 1.6): the reward itself teaches passive play. Action-first takes that hedge away.

Action-first still rewards the solver's most likely action, so where the solver mostly checks or calls (most postflop
spots) an unsure policy checks. A spot with `"reward": "lean_aggressive"` scores like action-first, except that a
mistake toward aggression keeps partial credit: a bet where the solver checks or calls, or a preflop raise where it
calls or checks, scores by the chips distance as above but at most `LEAN_CAP` (reward 6). A passive mistake (checking
or calling where the solver bets or raises), a postflop raise facing a bet where it calls, and any play where it folds
score 1. So betting pays once the policy is about a third sure the solver bets, not half, and loose calls stay
penalized. `"reward": "lean_small"` is the same but keeps the partial credit only for aggression that commits at most a
third of the hero's stack (SMALL_COMMIT): the event ranks by finishing place, where small pots won often help and
big-pot gambles risk busting.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any

from .engine import Decision, LegalActions
from .engine.table import resolve
from .prompts import SUBMIT_ACTION, parse_tool_call
from .seat import invalid_action

SPOT_KEY = "pokerbench_spot"
SPOT_FIELDS = ("messages", "legal", "label", "solver_text", "source", "pot")
REWARDS = ("distance", "action_first", "lean_aggressive", "lean_small")  # the spot's optional "reward"; distance is the default
# Table talk that says something about the speaker's hand (the table lowercases talk first); kept in step with
# vals_tools/hint_report.py's HAND_TALK. A spot with "leak_penalty": N loses N points when the reply's `say` matches;
# one with "talk_penalty": M loses M points for any non-empty `say` (silence carries no information).
HAND_TALK = re.compile(
    r"\b(aces?|kings?|queens?|jacks?|tens?|deuces?|pairs?|pocket|suited|offsuit|flush|straight|sets?|trips|nuts|monster|"
    r"air|nothing|bluff\w*|strong|weak|draws?|junk|trash|premium|top pair|rockets|cowboys|hooks|ducks|snowmen|big slick)\b"
    r"|\b(?!at\b)[akqjt][- ]?[akqjt2-9]s?\b"  # card shorthand: ak, k-j, tts, a5s (not the word "at")
    r"|\b[2-9]s\b"  # a pair by rank: pocket 3s
)
LEAN_CAP = 0.5  # lean_aggressive: the most similarity a mistake toward aggression keeps (a reward of 6)
AMOUNT_TOLERANCE = 0.25  # for the yes/no `agrees`: a bet or raise within 25% of the solver's amount (or 2 chips)
AGREE, CLEAN, FAILURE = 11.0, 1.0, 0.0
DISTANCE_SCALE = 0.25  # chips-added distance, as a share of the pot after a call, that earns half the bonus
LENGTH_FREE, LENGTH_FULL, LENGTH_PENALTY = 2500, 5000, 3.0
HERO_BET = re.compile(r"\(you\)(?: \[button\])? stack \d+, bet (\d+)")  # the table line prompts.py renders for the hero
HERO_STACK = re.compile(r"\(you\)(?: \[button\])? stack (\d+)")
SMALL_COMMIT = 1 / 3  # lean_small: aggression keeps partial credit only when it commits at most this share of the stack


def hero_stack(spot: dict[str, Any]) -> int:
    """The hero's chips behind, from the last table the conversation shows it (0 if none)."""
    for message in reversed(spot["messages"]):
        if message.get("role") == "user" and (found := HERO_STACK.findall(str(message.get("content", "")))):
            return int(found[-1])
    return 0


def hero_bet(spot: dict[str, Any]) -> int:
    """The hero's chips already in on this street, from the last table the conversation shows it (0 if none)."""
    for message in reversed(spot["messages"]):
        if message.get("role") == "user" and (found := HERO_BET.findall(str(message.get("content", "")))):
            return int(found[-1])
    return 0


def chips_added(decision: Decision, legal: LegalActions, bet: int) -> int:
    """The chips an action puts in now: 0 to fold or check, the call amount to call, the rest of a raise's to-amount."""
    if decision.type in ("fold", "check"):
        return 0
    if decision.type == "call":
        return legal.call_amount
    return max(0, (decision.amount or 0) - bet)


def similarity(mine: Decision, solver: Decision, spot: dict[str, Any], legal: LegalActions) -> float:
    """1 for the solver's exact play, falling smoothly with the chips-added distance between the two actions."""
    if mine.type == solver.type and mine.amount == solver.amount:
        return 1.0
    bet = hero_bet(spot)
    pot_after_call = max(1, int(spot["pot"]) + legal.call_amount)
    distance = abs(chips_added(mine, legal, bet) - chips_added(solver, legal, bet)) / pot_after_call
    return 1.0 / (1.0 + (distance / DISTANCE_SCALE) ** 2)


def check_spot(spot: Any) -> str | None:
    """Why `spot` is not a usable solver spot, or None."""
    if not isinstance(spot, dict) or any(k not in spot for k in SPOT_FIELDS):
        return f"must be an object with {', '.join(SPOT_FIELDS)}"
    messages = spot["messages"]
    if not isinstance(messages, list) or not messages or messages[0].get("role") != "system":
        return "messages must start with the system message"
    try:
        LegalActions(**spot["legal"])
    except TypeError as exc:
        return f"legal is not a legal-actions object ({exc})"
    if not isinstance(spot["label"], dict) or "action" not in spot["label"]:
        return "label must be {action[, amount]}"
    if spot.get("reward", "distance") not in REWARDS:
        return f"reward must be one of {', '.join(REWARDS)}"
    return None


def judge(spot: dict[str, Any], reply: Any) -> dict[str, Any]:
    """Whether the reply is clean under the event's strict rules, what it plays, and how it compares with the solver."""
    legal = LegalActions(**spot["legal"])
    submits = [c for c in reply.tool_calls if c["function"]["name"] == SUBMIT_ACTION]
    error = None
    if reply.truncated:
        error = "reply hit the token cap"
    elif len(submits) != 1 or len(reply.tool_calls) != 1:
        error = f"{len(submits)} submit_action calls and {len(reply.tool_calls) - len(submits)} other calls"
    failed = {"clean": False, "decision": None, "agrees": False, "type_agrees": False, "similarity": 0.0}
    if error:
        return failed | {"error": error}
    decision, error = parse_tool_call(submits[0]["function"]["arguments"], talk=True)
    error = error or invalid_action(decision, SimpleNamespace(legal=legal))
    if error:
        return failed | {"error": error}
    mine, _ = resolve(decision, legal)
    label = spot["label"]
    solver, _ = resolve(Decision(label["action"], label.get("amount")), legal)
    type_agrees = mine.type == solver.type
    agrees = type_agrees and (
        solver.amount is None or abs(mine.amount - solver.amount) <= max(2, AMOUNT_TOLERANCE * solver.amount)
    )
    said = str(decision.model_response.get("say") or "")
    leaks = bool(HAND_TALK.search(said.lower()))
    played = {"action": mine.type, **({"amount": mine.amount} if mine.amount is not None else {})}
    close = similarity(mine, solver, spot, legal) if "pot" in spot else float(agrees)
    if spot.get("reward") == "action_first" and not type_agrees:
        close = 0.0  # a different action earns nothing: calling cannot hedge between folding and raising
    elif spot.get("reward") in ("lean_aggressive", "lean_small") and not type_agrees:
        toward = mine.type in ("bet", "raise") and solver.type in ("check", "call") and (
            mine.type == "bet" or spot.get("street") == "preflop")
        if spot.get("reward") == "lean_small":  # only small aggression: what it commits is at most a third of its stack
            toward = toward and chips_added(mine, legal, hero_bet(spot)) <= SMALL_COMMIT * max(1, hero_stack(spot))
        close = min(LEAN_CAP, close) if toward else 0.0  # only mistakes toward aggression keep partial credit
    return {"clean": True, "error": None, "decision": played, "agrees": agrees,
            "type_agrees": type_agrees and not agrees, "similarity": round(close, 4), "say": said, "leaks": leaks,
            "leak_penalty": float(spot.get("leak_penalty") or 0) if leaks else 0.0,
            "talk_penalty": float(spot.get("talk_penalty") or 0) if said.strip() else 0.0}


def spot_reward(judgment: dict[str, Any], out_tokens: int | None) -> float:
    if not judgment["clean"]:
        return FAILURE
    reward = CLEAN + (AGREE - CLEAN) * judgment["similarity"]
    over = min(1.0, max(0.0, ((out_tokens or 0) - LENGTH_FREE) / (LENGTH_FULL - LENGTH_FREE)))
    reward = max(CLEAN, reward - LENGTH_PENALTY * over)
    # any talk costs the spot's talk_penalty, and talk naming our hand its leak_penalty on top
    return max(FAILURE, reward - judgment.get("talk_penalty", 0.0) - judgment.get("leak_penalty", 0.0))


async def play_spot(
    rollout_id: str, seed: int, spot: dict[str, Any], complete: Callable[[list[dict]], Awaitable[Any]], cap: int | None
) -> dict[str, Any]:
    """Ask the policy the spot once and judge the reply. The log has no hands; `kind` tells the grader what it is."""
    reply = await complete(spot["messages"])
    if cap and (reply.out_tokens or 0) > cap:  # cut at the cap even if the server returned it whole
        reply.truncated = True
    judgment = judge(spot, reply)
    return {
        "game_id": rollout_id,
        "kind": "solver_spot",
        "hands": [],
        "session_info": {"seed": seed},
        "spot": {k: spot.get(k) for k in ("source", "street", "hero_position", "solver_text", "label", "pot", "reward")},
        "reply": {
            "reasoning": reply.reasoning[-4000:],
            "text": reply.text[-2000:],
            "tool_calls": reply.tool_calls,
            "out_tokens": reply.out_tokens,
            "reasoning_tokens": reply.reasoning_tokens,
            "truncated": reply.truncated,
        },
        "judgment": judgment,
    }


def grade_spot(log: dict[str, Any]) -> dict[str, Any]:
    """The `collect_score` reply for a solver-spot rollout."""
    judgment, spot = log["judgment"], log["spot"]
    tokens = log["reply"].get("out_tokens")
    score = spot_reward(judgment, tokens)
    played = judgment["decision"] or {}
    verdict = "agrees" if judgment["agrees"] else "same action" if judgment["type_agrees"] else (
        "differs" if judgment["clean"] else f"failed ({judgment['error']})"
    )
    return {
        "score": score,
        "reasoning": f"Solver: {spot['solver_text']}; played {played or '-'}: {verdict}; {tokens} output tokens.",
        "artifacts": {
            "kind": "solver_spot",
            "clean": judgment["clean"],
            "agrees": judgment["agrees"],
            "type_agrees": judgment["type_agrees"],
            "similarity": judgment["similarity"],
            "error": judgment["error"],
            "played": played,
            "solver": spot["label"],
            "street": spot.get("street"),
            "out_tokens": tokens,
            "deal": f"spot:{(spot.get('source') or {}).get('file')}:{(spot.get('source') or {}).get('row')}",
        },
    }
