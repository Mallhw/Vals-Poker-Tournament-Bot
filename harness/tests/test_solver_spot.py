"""Solver-spot rollouts: one PokerBench decision through the relay, rewarded continuously for closeness to the solver."""

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Request
from test_grading import grade
from test_server import play

from poker_harness.seat import Completion
from poker_harness.server import create_app
from poker_harness.solver_spot import SPOT_KEY, check_spot, grade_spot, hero_bet, judge, spot_reward

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pokerbench_convert as pc

# First to act under the gun with AsKs: the solver opens to 2 BB, a raise to 4 chips at the event's 1/2.
OPEN = {"prev_line": "", "hero_pos": "UTG", "hero_holding": "AsKs", "correct_decision": "2.0bb", "pot_size": "1.5"}
CONVERTED = asyncio.run(pc.convert_row(OPEN, {"file": "test.csv", "row": 7}))
SPOT = {k: CONVERTED[k] for k in ("messages", "legal", "label", "solver_text", "source", "pot", "street", "hero_position")}


def table(bet: int = 0) -> list[dict]:
    return [{"role": "system", "content": "rules"},
            {"role": "user", "content": f"Table:\n  seat 2: seat2 stack 190, bet 10\n  seat 3: seat3 (you) stack 168, bet {bet}\n"}]


def spot(label: dict, legal: dict, pot: int, bet: int = 0) -> dict:
    return {"messages": table(bet), "legal": legal, "label": label, "solver_text": "x", "source": {"row": 1}, "pot": pot}


def reply(arguments: dict, tokens: int = 500, truncated: bool = False) -> Completion:
    call = {"id": "c1", "type": "function", "function": {"name": "submit_action", "arguments": json.dumps(arguments)}}
    return Completion(text="", tool_calls=[call], out_tokens=tokens, truncated=truncated)


def reward(s: dict, arguments: dict, tokens: int = 500, truncated: bool = False) -> float:
    return spot_reward(judge(s, reply(arguments, tokens, truncated)), tokens)


FREE = {"can_fold": False, "can_check": True, "call_amount": 0, "raise_type": "bet", "min_raise_to": 2, "max_raise_to": 168}


def test_the_reward_is_continuous_in_how_far_the_action_is_from_the_solvers() -> None:
    bet_20 = spot({"action": "bet", "amount": 20}, FREE, pot=40)
    assert reward(bet_20, {"action": "bet", "amount": 20}) == 11.0
    assert 10.4 < reward(bet_20, {"action": "bet", "amount": 18}) < 10.8  # 10% under the solver's size
    assert reward(bet_20, {"action": "bet", "amount": 18}) > reward(bet_20, {"action": "bet", "amount": 30}) > reward(
        bet_20, {"action": "check"}
    )
    facing_10 = {"can_fold": True, "can_check": False, "call_amount": 10, "raise_type": "raise", "min_raise_to": 20,
                 "max_raise_to": 168}
    min_raise = spot({"action": "raise", "amount": 20}, facing_10, pot=30)
    assert reward(min_raise, {"action": "call"}) == pytest.approx(6.0)  # calling where the solver min-raises
    tiny = {"can_fold": True, "can_check": False, "call_amount": 1, "raise_type": "raise", "min_raise_to": 2,
            "max_raise_to": 168}
    assert reward(spot({"action": "call"}, tiny, pot=40), {"action": "fold"}) > 10.8  # folding to a tiny bet
    open_ = {"can_fold": True, "can_check": False, "call_amount": 2, "raise_type": "raise", "min_raise_to": 4,
             "max_raise_to": 200}
    assert reward(spot({"action": "raise", "amount": 5}, open_, pot=3), {"action": "fold"}) == pytest.approx(1.588, abs=0.01)


def test_action_first_scores_the_action_before_the_size_so_calling_cannot_hedge() -> None:
    facing_10 = {"can_fold": True, "can_check": False, "call_amount": 10, "raise_type": "raise", "min_raise_to": 20,
                 "max_raise_to": 168}
    min_raise = spot({"action": "raise", "amount": 20}, facing_10, pot=30) | {"reward": "action_first"}
    assert reward(min_raise, {"action": "call"}) == reward(min_raise, {"action": "fold"}) == 1.0  # both just clean
    assert reward(min_raise, {"action": "raise", "amount": 20}) == 11.0
    # the right action, 4 chips too big in a pot of 40 once called: sized as before, 1 + 10 / (1 + 0.4^2)
    assert reward(min_raise, {"action": "raise", "amount": 24}) == pytest.approx(9.621, abs=1e-3)
    folds = spot({"action": "fold"}, facing_10, pot=30) | {"reward": "action_first"}
    assert reward(folds, {"action": "fold"}) == 11.0 and reward(folds, {"action": "call"}) == 1.0  # loose: no credit
    bet_20 = spot({"action": "bet", "amount": 20}, FREE, pot=40) | {"reward": "action_first"}
    assert reward(bet_20, {"action": "check"}) == reward(bet_20, {"action": "fold"}) == 1.0  # fold plays as check
    assert reward(bet_20 | {"reward": "distance"}, {"action": "check"}) == pytest.approx(3.0)  # the default, named
    assert check_spot(SPOT | {"reward": "action_first"}) is None and "reward" in check_spot(SPOT | {"reward": "x"})


def test_lean_aggressive_keeps_partial_credit_only_for_mistakes_toward_aggression() -> None:
    lean = {"reward": "lean_aggressive", "street": "flop"}
    checks = spot({"action": "check"}, FREE, pot=40) | lean
    assert reward(checks, {"action": "check"}) == 11.0
    assert reward(checks, {"action": "bet", "amount": 2}) == pytest.approx(6.0)  # a min bet: capped at 6
    assert reward(checks, {"action": "bet", "amount": 13}) == pytest.approx(4.72, abs=0.01)  # about a third of the pot
    bets = spot({"action": "bet", "amount": 20}, FREE, pot=40) | lean
    assert reward(bets, {"action": "check"}) == 1.0  # a passive mistake earns nothing
    facing_10 = {"can_fold": True, "can_check": False, "call_amount": 10, "raise_type": "raise", "min_raise_to": 20,
                 "max_raise_to": 168}
    calls = spot({"action": "call"}, facing_10, pot=30) | lean
    assert reward(calls, {"action": "raise", "amount": 20}) == 1.0  # a postflop raise facing a bet: no credit
    assert reward(calls | {"street": "preflop"}, {"action": "raise", "amount": 20}) == pytest.approx(6.0)  # iso/3-bet
    folds = spot({"action": "fold"}, facing_10, pot=30) | lean | {"street": "preflop"}
    assert reward(folds, {"action": "raise", "amount": 20}) == 1.0 and reward(folds, {"action": "call"}) == 1.0
    assert check_spot(SPOT | {"reward": "lean_aggressive"}) is None


def test_lean_small_keeps_partial_credit_only_for_aggression_that_commits_a_third_of_the_stack_or_less() -> None:
    facing_50 = {"can_fold": True, "can_check": False, "call_amount": 50, "raise_type": "raise", "min_raise_to": 100,
                 "max_raise_to": 150}
    big = spot({"action": "call"}, facing_50, pot=300) | {"street": "preflop"}
    big["messages"] = [{"role": "system", "content": "rules"},
                       {"role": "user", "content": "Table:\n  seat2: seat2 stack 400, bet 50\n  seat3 (you) stack 150, bet 0\n"}]
    shove = {"action": "raise", "amount": 150}
    assert 4.0 < reward(big | {"reward": "lean_aggressive"}, shove) < 6.0  # an all-in where it calls: partial credit
    assert reward(big | {"reward": "lean_small"}, shove) == 1.0  # commits all of its stack: none under lean_small
    checks = spot({"action": "check"}, FREE, pot=40) | {"reward": "lean_small", "street": "flop"}
    assert reward(checks, {"action": "bet", "amount": 13}) == pytest.approx(4.72, abs=0.01)  # 13 of 168: still credited
    assert reward(checks, {"action": "check"}) == 11.0 and check_spot(SPOT | {"reward": "lean_small"}) is None


def test_talk_that_names_our_hand_costs_the_spots_leak_penalty() -> None:
    checks = spot({"action": "check"}, FREE, pot=40) | {"reward": "lean_aggressive", "street": "flop", "leak_penalty": 3}
    assert reward(checks, {"action": "check"}) == 11.0  # silent
    assert reward(checks, {"action": "check", "say": "nice hand earlier"}) == 11.0  # harmless talk
    assert reward(checks, {"action": "check", "say": "I have the nuts"}) == 8.0
    assert reward(checks, {"action": "bet", "amount": 20, "say": "pocket aces baby"}) == pytest.approx(0.0, abs=1.0)
    bets = spot({"action": "bet", "amount": 20}, FREE, pot=40) | {"reward": "lean_aggressive", "street": "flop",
                                                                  "leak_penalty": 3}
    assert reward(bets, {"action": "check", "say": "top pair"}) == 0.0  # 1 - 3: never below a format failure
    assert reward(checks | {"leak_penalty": 0}, {"action": "check", "say": "I have the nuts"}) == 11.0  # off unless set
    assert judge(checks, reply({"action": "check", "say": "Weak hand, I check"}))["leaks"] is True
    leaks = lambda say: judge(checks, reply({"action": "check", "say": say}))["leaks"]  # noqa: E731
    assert all(map(leaks, ["too bad, ak beats whatever you have", "raising with pocket 3s on the button", "kj",
                           "ladies and gentlemen, pocket rockets!", "a-k suited!", "cowboys again"]))
    assert not any(map(leaks, ["look at that", "good luck all", "nice hand", "lets see a flop."]))
    quiet = checks | {"talk_penalty": 2}
    assert reward(quiet, {"action": "check"}) == 11.0 and reward(quiet, {"action": "check", "say": ""}) == 11.0
    assert reward(quiet, {"action": "check", "say": "nice hand"}) == 9.0  # any talk: -2
    assert reward(quiet, {"action": "check", "say": "I have the nuts"}) == 6.0  # a leak: -2 - 3


def test_failures_score_zero_and_long_replies_lose_up_to_three() -> None:
    bet_20 = spot({"action": "bet", "amount": 20}, FREE, pot=40)
    assert reward(bet_20, {"action": "bet", "amount": 20}, truncated=True) == 0.0
    # Folding when it could check is played as a check at the event, so it scores exactly like one.
    assert reward(bet_20, {"action": "fold"}) == reward(bet_20, {"action": "check"}) == pytest.approx(3.0)
    assert reward(bet_20, {"action": "raise", "amount": 500}) == 0.0  # outside the legal range: a format failure
    assert reward(bet_20, {"action": "bet", "amount": 20}, tokens=3750) == pytest.approx(9.5)
    assert reward(bet_20, {"action": "check"}, tokens=5000) == 1.0  # never below a clean reply's floor


def test_the_heros_bet_is_read_from_the_last_table_and_raises_count_only_what_they_add() -> None:
    assert hero_bet({"messages": table(bet=10)}) == 10 and hero_bet({"messages": [{"role": "user", "content": "x"}]}) == 0
    facing = {"can_fold": True, "can_check": False, "call_amount": 20, "raise_type": "raise", "min_raise_to": 50,
              "max_raise_to": 168}
    s = spot({"action": "raise", "amount": 60}, facing, pot=60, bet=10)  # hero has 10 in: the raise to 60 adds 50
    assert reward(s, {"action": "raise", "amount": 60}) == 11.0
    # 5 chips apart in a pot of 80 once called: 1 / (1 + (0.0625 / 0.25)^2)
    assert judge(s, reply({"action": "raise", "amount": 55}))["similarity"] == pytest.approx(0.9412, abs=1e-4)


def test_a_converted_spot_checks_and_the_grade_names_the_deal() -> None:
    assert check_spot(SPOT) is None and "pot" in check_spot({k: v for k, v in SPOT.items() if k != "pot"})
    log = {"spot": SPOT, "reply": {"out_tokens": 400}, "judgment": judge(SPOT, reply({"action": "raise", "amount": 4}))}
    graded = grade_spot(log)
    assert graded["score"] == 11.0 and graded["artifacts"]["agrees"] and graded["artifacts"]["deal"] == "spot:test.csv:7"


def relay(requests: list[dict]) -> FastAPI:
    app = FastAPI()

    @app.post("/job/trace/v1/chat/completions")
    async def chat(request: Request) -> dict:
        body = await request.json()
        requests.append(body)
        call = {"id": "c1", "type": "function", "function": {"name": "submit_action", "arguments": '{"action": "raise", "amount": 4}'}}
        message = {"role": "assistant", "content": "", "reasoning_content": "AKs opens.", "tool_calls": [call]}
        return {"choices": [{"message": message, "finish_reason": "tool_calls"}], "usage": {"completion_tokens": 300}}

    return app


async def test_a_spot_task_asks_the_policy_once_and_is_graded_against_the_solver(tmp_path: Path) -> None:
    requests: list[dict] = []
    app = create_app(relay_transport=httpx.ASGITransport(app=relay(requests)), runs_dir=tmp_path)
    params = {"preset": "event", "submit_fields": ["action", "amount"], "seats": 6, "total_hands": 1, "seed": 7, SPOT_KEY: SPOT}
    task = {"input": [{"role": "user", "content": "decide"}], "env_params": params}
    status, outputs = await play(app, task)
    assert status["status"] == "completed" and outputs["log"]["kind"] == "solver_spot" and len(requests) == 1
    body = requests[0]
    assert body["messages"] == SPOT["messages"] and body["max_completion_tokens"] == 5000
    assert [{"type": "function", **t["function"]} for t in body["tools"]] == CONVERTED["tools"]  # what the event shows
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://h") as client:
        grading = {"rollout_id": "gw_1", "rollout_status": "completed", "task": task}
        graded, score = await grade(client, grading)
    assert graded["status"] == "completed" and score.json()["score"] == 11.0
    assert list(tmp_path.glob("*.json"))


async def test_a_malformed_spot_is_rejected(tmp_path: Path) -> None:
    app = create_app(runs_dir=tmp_path)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://h") as client:
        task = {"input": [], "env_params": {"seed": 1, SPOT_KEY: {"messages": []}}}
        response = await client.post("/submit", json={"ac_api_url": "http://r/v1", "ac_api_key": "k", "task": task})
    assert response.status_code == 422 and SPOT_KEY in response.text
