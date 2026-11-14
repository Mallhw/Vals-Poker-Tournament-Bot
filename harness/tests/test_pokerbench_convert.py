"""pokerbench_convert.py: PokerBench spots replayed into the exact conversation the event table shows the trained seat."""

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pokerbench_convert as pc  # noqa: E402

from poker_harness.completions import table_rules, with_preset  # noqa: E402
from poker_harness.prompts import system_prompt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SOURCE = {"file": "test.csv", "row": 0}

# Row 0 of PokerBench's preflop test file: UTG opens, the button calls, the small blind (hero) 3-bets to 13 BB, the big
# blind shoves, the others fold; the solver calls with kings.
PREFLOP_SHOVE = {
    "prev_line": "UTG/2.0bb/BTN/call/SB/13.0bb/BB/allin/UTG/fold/BTN/fold",
    "hero_pos": "SB",
    "hero_holding": "KdKc",
    "correct_decision": "call",
    "pot_size": "117.0",
}
# First to act under the gun: the solver opens to 2 BB.
PREFLOP_OPEN = {"prev_line": "", "hero_pos": "UTG", "hero_holding": "AsKs", "correct_decision": "2.0bb", "pot_size": "1.5"}
# Row 1 of the postflop test file: the small blind opens, the big blind (hero, in position) 3-bets and is called, bets
# the flop and the turn, and faces a check-raise to 80 BB on the turn; the solver calls.
POSTFLOP_TURN = {
    "preflop_action": "SB/3.0bb/BB/10.0bb/SB/call",
    "board_flop": "Ks7h2d",
    "board_turn": "Jc",
    "board_river": "7c",
    "aggressor_position": "IP",
    "postflop_action": "OOP_CHECK/IP_BET_10/OOP_CALL/dealcards/Jc/OOP_CHECK/IP_BET_44/OOP_RAISE_80",
    "evaluation_at": "Turn",
    "available_moves": "['Fold', 'Call']",
    "pot_size": "164",
    "hero_position": "IP",
    "holding": "AhQd",
    "correct_decision": "Call",
}


async def convert(row: dict) -> dict:
    return await pc.convert_row(row, SOURCE)


def test_the_table_is_the_training_tasks_table() -> None:
    """The rules come from vals_tools/tasks.py's TABLE_RULES for a one-hand 6-seat task, and the tools match the
    ones opsd/tools.json records for every seat."""
    rules = table_rules(with_preset(pc.training_table_rules() | {"seats": 6, "total_hands": 1}))
    assert pc.RULES == rules
    assert rules.seats == 6 and rules.total_hands == 1 and rules.strict_actions and rules.scratchpad and rules.table_talk
    assert pc.TOOLS == json.loads((ROOT / "opsd" / "tools.json").read_text())


async def test_a_first_decision_is_rendered_as_the_harness_renders_it() -> None:
    record = await convert(PREFLOP_OPEN)
    system, scratchpad, prompt = record["messages"]
    assert system == {"role": "system", "content": system_prompt(pc.RULES)}
    assert scratchpad["role"] == "user" and "(empty)" in scratchpad["content"]
    assert prompt == {
        "role": "user",
        "content": "\n".join(
            [
                "## Hand 1 of 1",
                "Your seat: 2 (position 2; 0 = small blind)",
                "Street: preflop",
                "Your hole cards: As Ks",
                "Board: -",
                "Pot (including current bets): 3",
                "Table:",
                "  seat 0: seat0 stack 199, bet 1",
                "  seat 1: seat1 stack 198, bet 2",
                "  seat 2: seat2 (you) stack 200, bet 0",
                "  seat 3: seat3 stack 200, bet 0",
                "  seat 4: seat4 stack 200, bet 0",
                "  seat 5: seat5 [button] stack 200, bet 0",
                "Actions this hand:",
                "  (no actions yet)",
                "",
                "Legal actions: fold, call 2, raise to 4-200",
                "Call submit_action with your decision.",
            ]
        ),
    }
    assert record["label"] == {"action": "raise", "amount": 4} and record["solver_text"] == "raise to 4"
    assert record["hero_position"] == "UTG" and record["street"] == "preflop"


async def test_earlier_hero_actions_are_calls_the_table_answered() -> None:
    record = await convert(PREFLOP_SHOVE)
    roles = [m["role"] for m in record["messages"]]
    assert roles == ["system", "user", "user", "assistant", "tool", "user"]
    call = record["messages"][3]["tool_calls"][0]
    assert json.loads(call["function"]["arguments"]) == {"action": "raise", "amount": 26}  # 13 BB at blinds 1/2
    assert record["messages"][4] == {"role": "tool", "tool_call_id": call["id"], "content": "Applied: raise to 26."}
    last = record["messages"][-1]["content"]
    assert "preflop: seat1 (seat 1) raise to 200" in last and last.endswith("Legal actions: fold, call 174\nCall submit_action with your decision.")
    assert record["pot"] == 234 == 2 * 117  # PokerBench's pot, in event chips
    assert record["label"] == {"action": "call"} and record["solver_text"] == "call 174"


async def test_a_postflop_spot_replays_every_street_and_amounts_are_to_amounts() -> None:
    record = await convert(POSTFLOP_TURN)
    assert record["hero_seat"] == 1 and record["hero_side"] == "IP" and record["street"] == "turn"
    calls = [json.loads(m["tool_calls"][0]["function"]["arguments"]) for m in record["messages"] if m["role"] == "assistant"]
    assert calls == [{"action": "raise", "amount": 20}, {"action": "bet", "amount": 20}, {"action": "bet", "amount": 88}]
    last = record["messages"][-1]["content"]
    assert "turn: seat0 (seat 0) raise to 160" in last and "Board: Ks 7h 2d Jc\n" in last
    assert record["pot"] == 328 == 2 * 164
    assert record["label"] == {"action": "call"} and record["legal"]["call_amount"] == 72


async def test_the_hero_never_sees_another_seats_cards() -> None:
    record = await convert(POSTFLOP_TURN)
    shown = {c for m in record["messages"][1:] for c in re.findall(r"\b[2-9TJQKA][cdhs]\b", m.get("content") or "")}
    assert shown == {"Ah", "Qd", "Ks", "7h", "2d", "Jc"}


@pytest.mark.parametrize(
    ("row", "change", "reason"),
    [
        (POSTFLOP_TURN, {"holding": "KsQd"}, "duplicate_cards"),  # the hero's king is also on the flop
        (POSTFLOP_TURN, {"pot_size": "180"}, "pot_mismatch"),
        (POSTFLOP_TURN, {"correct_decision": "Raise 81"}, "label_raise_not_legal"),  # calling puts the hero all in
        (POSTFLOP_TURN, {"evaluation_at": "River"}, "street_mismatch"),
        (POSTFLOP_TURN, {"postflop_action": "OOP_CHECK/IP_BET_10/OOP_CALL/dealcards/Qc/OOP_CHECK"}, "board_mismatch"),
        (POSTFLOP_TURN, {"postflop_action": "OOP_CHECK/IP_BET_0.4/OOP_CALL/dealcards/Jc/OOP_CHECK"}, "amount_out_of_range"),
        (PREFLOP_OPEN, {"correct_decision": "1.0bb"}, "label_amount_out_of_range"),  # under the minimum open of 2 BB
        (PREFLOP_OPEN, {"prev_line": "HJ/2.0bb"}, "line_out_of_order"),  # the line skips the hero, who acts first
    ],
)
async def test_unfaithful_rows_are_dropped_with_their_reason(row: dict, change: dict, reason: str) -> None:
    with pytest.raises(pc.Dropped, match=f"^{reason}$"):
        await convert(row | change)


def test_labels_and_amounts() -> None:
    assert pc.label_move("14.0bb") == ("raise", 14.0) and pc.label_move("Bet 24") == ("bet", 24.0)
    assert pc.label_move("allin") == ("allin", None) and pc.label_move("Check") == ("check", None)
    assert pc.chips(2.3) == 5 and pc.chips(2.25) == 5 and pc.chips(13.0) == 26
