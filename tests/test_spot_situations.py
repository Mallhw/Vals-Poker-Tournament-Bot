"""rl/spot_situations.py and rl/make_exploit_rows.py: situations read from the prompt, and round 4's exploit labels."""

import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rl"))
import spot_situations as ss  # noqa: E402

spec = importlib.util.spec_from_file_location("make_exploit_rows", ROOT / "rl" / "make_exploit_rows.py")
mer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mer)


def hero_call(action: str) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "submit_action", "arguments": json.dumps({"action": action, "amount": 5})}}]}


def user(*lines: str, street: str = "preflop") -> dict:
    return {"role": "user", "content": "\n".join(f"  {x}" for x in lines) + f"\n\nStreet: {street}\nLegal actions: x"}


def spot(street: str, position: str, *messages: dict) -> dict:
    return {"street": street, "hero_position": position, "messages": [{"role": "system", "content": "rules"}, *messages]}


def test_first_in_needs_everyone_before_the_hero_to_fold() -> None:
    opened = spot("preflop", "BTN", user("preflop: seat2 (seat 2) fold", "preflop: seat3 (seat 3) fold"))
    limped = spot("preflop", "BTN", user("preflop: seat2 (seat 2) call", "preflop: seat3 (seat 3) fold"))
    assert ss.first_in(opened) and ss.situation(opened) == "first in BTN"
    assert not ss.first_in(limped) and ss.situation(limped) == "preflop 0-1 raises"
    assert ss.situation(spot("preflop", "BB", user("preflop: seat2 (seat 2) raise to 5", "preflop: seat3 (seat 3) raise to 15"))) \
        == "preflop 2+ raises"


def test_a_cbet_chance_needs_the_hero_to_have_made_the_last_preflop_raise() -> None:
    ours = spot("flop", "BTN", user("preflop: seat2 (seat 2) fold"), hero_call("raise"),
                user("preflop: seat0 (seat 0) call", "flop: seat0 (seat 0) check", street="flop"))
    assert ss.cbet_chance(ours) and ss.situation(ours) == "flop c-bet chance"
    three_bet = spot("flop", "BTN", user("preflop: seat2 (seat 2) fold"), hero_call("raise"),
                     user("preflop: seat0 (seat 0) raise to 15"), hero_call("call"),
                     user("flop: seat0 (seat 0) check", street="flop"))
    assert not ss.cbet_chance(three_bet) and ss.situation(three_bet) == "flop other"  # the villain raised last
    led = spot("flop", "BTN", user("preflop: seat2 (seat 2) fold"), hero_call("raise"),
               user("preflop: seat0 (seat 0) call", "flop: seat0 (seat 0) bet to 6", street="flop"))
    assert not ss.cbet_chance(led)


def test_opening_ranges_take_the_strongest_hands_to_the_asked_share() -> None:
    equity = json.loads(mer.EQUITY.read_text())
    btn = mer.opening_range(0.65, equity)
    share = sum(map(mer.combos, btn)) / 1326
    assert 0.62 < share < 0.68 and "AA" in btn and "72o" not in btn
    assert mer.opening_range(0.45, equity) < btn
    assert mer.hand_class(("Kd", "As")) == "AKo" and mer.hand_class(("9h", "Th")) == "T9s" and mer.hand_class(("5c", "5d")) == "55"


def test_first_in_rows_open_the_range_and_fold_the_rest() -> None:
    equity = json.loads(mer.EQUITY.read_text())
    rows = mer.first_in_rows(300, random.Random(1), equity)
    ranges = {pos: mer.opening_range(share, equity) for pos, share in mer.RANGE.items()}
    for r in rows:
        cls = mer.hand_class((r["hero_holding"][:2], r["hero_holding"][2:]))
        assert r["correct_decision"] == (mer.OPEN_TO[r["hero_pos"]] if cls in ranges[r["hero_pos"]] else "fold")
        assert r["prev_line"] == "" and r["pot_size"] == "1.5"
    assert {r["hero_pos"] for r in rows} == {"CO", "BTN", "SB"}


def test_the_weakest_share_by_flop_equity_checks_and_the_rest_bet_a_third() -> None:
    rows = [{"flop_equity": f"{e:.2f}", "pot_size": "12"} for e in (0.1, 0.9, 0.3, 0.5, 0.7)]
    mer.relabel_weakest(rows, 0.2)
    assert [r["correct_decision"] for r in rows] == ["Check", "Bet 4", "Bet 4", "Bet 4", "Bet 4"]
    mer.relabel_weakest(rows, 0.0)
    assert all(r["correct_decision"] == "Bet 4" for r in rows)
    pytest.importorskip("pokerkit")  # the harness's estimator: run from harness/ (its venv) to check it
    assert 0.8 < mer.flop_equity("AsAh", "Ad7c2h", 1) <= 1.0 and mer.flop_equity("3c2d", "AsKdQh", 1) < 0.3


def test_cbet_rows_are_flop_spots_where_the_preflop_aggressor_acts_first_or_is_checked_to() -> None:
    base = {"evaluation_at": "Flop", "aggressor_position": "IP", "hero_position": "IP", "postflop_action": "OOP_CHECK"}
    assert mer.cbet_chance(base)
    assert mer.cbet_chance(base | {"aggressor_position": "OOP", "hero_position": "OOP", "postflop_action": ""})
    assert not mer.cbet_chance(base | {"postflop_action": "OOP_BET_3"})
    assert not mer.cbet_chance(base | {"aggressor_position": "OOP"})
    assert not mer.cbet_chance(base | {"evaluation_at": "Turn"})
