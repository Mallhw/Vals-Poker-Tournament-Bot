"""Situations of a converted PokerBench spot, read from the conversation the hero sees (no solver labels used).

- `raises_before_hero`: every raise before the hero's decision (others' history lines plus the hero's own raises).
- `first_in`: preflop, everyone before the hero folded (no raise, no limp): an open-or-fold decision.
- `cbet_chance`: flop, the hero made the last preflop raise and nobody has bet the flop yet.
- `situation`: a short name for the spot's situation, used to weight training data and to report frequencies.
"""

from __future__ import annotations

import json
import re
from typing import Any

HISTORY = re.compile(r"^\s*(preflop|flop|turn|river): .*?\) (fold|check|call|bet|raise)\b", re.M)
STREET = re.compile(r"^Street: (\w+)", re.M)
LATE = {"CO", "BTN", "SB"}


def actions(record: dict[str, Any]) -> list[tuple[str, str, bool]]:
    """(street, action, is_hero) in order: others' history lines and the hero's earlier submit_action calls."""
    seen: list[tuple[str, str, bool]] = []
    street = "preflop"
    for m in record["messages"]:
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            seen += [(s, a, False) for s, a in HISTORY.findall(m["content"])]
            if found := STREET.findall(m["content"]):
                street = found[-1]
        elif m.get("role") == "assistant":
            for call in m.get("tool_calls") or []:
                args = call.get("function", {}).get("arguments", "{}")
                try:
                    action = (json.loads(args) if isinstance(args, str) else args).get("action", "")
                except (json.JSONDecodeError, AttributeError):
                    action = ""
                seen.append((street, action, True))
    return seen


def raises_before_hero(record: dict[str, Any]) -> int:
    return sum(a == "raise" for s, a, _ in actions(record) if s == "preflop")


def first_in(record: dict[str, Any]) -> bool:
    return record.get("street") == "preflop" and all(a == "fold" for s, a, _ in actions(record) if s == "preflop")


def cbet_chance(record: dict[str, Any]) -> bool:
    if record.get("street") != "flop":
        return False
    seen = actions(record)
    pre = [(a, hero) for s, a, hero in seen if s == "preflop" and a in ("raise", "bet")]
    flop_bets = [a for s, a, _ in seen if s == "flop" and a in ("bet", "raise")]
    return bool(pre) and pre[-1][1] and not flop_bets


def situation(record: dict[str, Any]) -> str:
    street = record.get("street")
    if street == "preflop":
        if first_in(record):
            return f"first in {record.get('hero_position')}"
        return "preflop 0-1 raises" if raises_before_hero(record) <= 1 else "preflop 2+ raises"
    if cbet_chance(record):
        return "flop c-bet chance"
    return f"{street} other"
