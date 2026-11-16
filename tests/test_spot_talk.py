"""rl/spot_talk.py: talk goes only on opponents' action lines, in the harness's format, and nothing else changes."""

import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rl"))
from spot_talk import ACTION_LINE, add_talk  # noqa: E402

RECORD = {
    "hero_seat": 3,
    "label": {"action": "call"},
    "legal": {"can_fold": True, "call_amount": 18},
    "messages": [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "## Hand 1 of 1\nActions this hand:\n  preflop: seat2 (seat 2) fold\n\nLegal actions: fold"},
        {"role": "assistant", "content": ""},
        {"role": "tool", "content": "Applied: raise to 4."},
        {"role": "user", "content": "Since your last action:\n  turn: seat1 (seat 1) raise to 28\n  turn: seat3 (seat 3) bet to 10\n"
                                    "\nLegal actions: fold, call 18, raise to 46-196"},
    ],
}


def test_only_opponent_lines_talk_and_the_rest_is_unchanged():
    out = add_talk(RECORD, random.Random(0), ["i have the nuts"], rate=1.0)
    assert out["label"] == RECORD["label"] and out["legal"] == RECORD["legal"] and out["hero_seat"] == 3
    lines = [l for m in out["messages"] if m["role"] == "user" for l in m["content"].split("\n")]
    assert '  turn: seat1 (seat 1) raise to 28 and says: "i have the nuts"' in lines
    assert '  preflop: seat2 (seat 2) fold and says: "i have the nuts"' in lines
    assert "  turn: seat3 (seat 3) bet to 10" in lines  # the hero's own line never talks
    assert [m["content"] for m in out["messages"] if m["role"] != "user"] == ["rules", "", "Applied: raise to 4."]
    assert RECORD["messages"][4]["content"].count("says") == 0  # the input record is not modified


def test_last_history_always_has_talk_even_at_rate_zero():
    out = add_talk(RECORD, random.Random(0), ["pure bluff, call me"], rate=0.0)
    last = out["messages"][4]["content"]
    assert last.count('and says: "pure bluff, call me"') == 1 and "seat1 (seat 1) raise to 28 and says" in last


def test_action_line_pattern_matches_harness_lines():
    assert ACTION_LINE.match("  river: seat5 (seat 5) check")
    assert ACTION_LINE.match("  flop: seat0 (seat 0) bet to 12").group(3) == "0"
    assert not ACTION_LINE.match("Legal actions: fold, call 18")
    assert re.search(r'and says: "', add_talk(RECORD, random.Random(1), ["x"], 1.0)["messages"][1]["content"])
