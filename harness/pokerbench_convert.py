"""Turn PokerBench spots into the exact conversation the event harness shows the trained seat, with the solver's play.

PokerBench (https://huggingface.co/datasets/RZ412/PokerBench, Apache-2.0) holds 6-max no-limit hold'em spots at 100 BB
with a solver's decision. Its prompts are prose in its own format; a model trained on them would see a table it
never meets at the event. This script replays each spot through the harness's own engine and seat instead, so the
output is what the harness would send at that decision, message for message:

- the training tasks' rules (`vals_tools/tasks.py`'s `TABLE_RULES`, one 6-seat hand: the event preset, table talk,
  strict actions, the scratchpad and its tools), read from that file so the two can't drift apart;
- the hand dealt with the spot's cards: the hero's hole cards and the spot's board; the other seats get random
  unused cards, which the hero never sees;
- every action before the decision replayed in order: opponents follow the PokerBench line, players the line doesn't
  mention fold preflop (PokerBench's own rule), and the hero's earlier actions are `submit_action` calls in its
  conversation, answered by the table as usual;
- the conversation is captured when the hero is asked to act at the spot, and the solver's play becomes a
  `submit_action` call that the event's strict rules accept.

Chips: PokerBench plays blinds 0.5/1 with 100-chip stacks; the event plays 1/2 with 200, so every amount doubles (and
rounds to whole chips: 2.3 BB becomes 5). Every amount in a line is a "to" amount: preflop `13.0bb` is a raise to 13
BB, postflop `IP_BET_5` a bet of 5 and `OOP_RAISE_14` a raise to 14, as are the labels `Bet 24` and `Raise 29`. A
spot is kept only if the replayed pot matches PokerBench's `pot_size` and its label is a legal action; every other
row is counted by why it was dropped.

    uv run python pokerbench_convert.py preflop_1k_test_set_game_scenario_information.csv --out pb.jsonl
    uv run python pokerbench_convert.py postflop_10k_test_set_game_scenario_information.csv --out pb.jsonl --limit 500

Each output line: `messages` (the chat messages up to the decision, as the relay receives them), `tools` (the
seat's tools in AC2's form; `opsd/tools.json`), `label` (`{"action", "amount"}` in event chips), `solver_text`
(e.g. "raise to 6"), `source`, `street`, `hero_position`, `hero_seat`, `pot` and `legal`. Build AC2 tasks from it
with `opsd/pokerbench_tasks.py` (root project).
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import csv
import json
import math
import random
import re
import sys
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pokerkit import Card, Deck

from poker_harness.completions import table_rules, with_preset
from poker_harness.engine import Decision, DecisionContext, HandRunner, LegalActions
from poker_harness.prompts import SUBMIT_ACTION, table_tools, talk_fields
from poker_harness.seat import Completion, ModelSeat, invalid_action

REPO = Path(__file__).resolve().parents[1]
SEATS = 6
# Hand 1 with the button on the last seat, as the training tasks deal it (`vals_tools.tasks.hand_task`).
POSITIONS = {"SB": 0, "BB": 1, "UTG": 2, "HJ": 3, "CO": 4, "BTN": 5}
POSITION_NAMES = {seat: name for name, seat in POSITIONS.items()}
SCALE = 2  # PokerBench's blinds are 0.5/1 and stacks 100; the event's are 1/2 and 200
# Chips. PokerBench's postflop pots are whole BB with the small blind's dead 0.5 dropped (most spots replay 1 chip
# higher), fractional sizes round to whole chips, and some lines print bet sizes rounded to whole BB, so their pots
# replay a few chips high. `--pot-tolerance` changes it.
POT_TOLERANCE = 3
STREETS = {"Flop": "flop", "Turn": "turn", "River": "river"}


def training_table_rules(tasks_file: Path = REPO / "vals_tools" / "tasks.py") -> dict[str, Any]:
    """`TABLE_RULES` from `vals_tools/tasks.py`, read without importing it (that module needs the ac2 SDK)."""
    for node in ast.parse(tasks_file.read_text()).body:
        if isinstance(node, ast.AnnAssign | ast.Assign):
            targets = [node.target] if isinstance(node, ast.AnnAssign) else node.targets
            if any(isinstance(t, ast.Name) and t.id == "TABLE_RULES" for t in targets) and node.value is not None:
                return ast.literal_eval(node.value)
    raise RuntimeError(f"no TABLE_RULES in {tasks_file}")


def table_params() -> dict[str, Any]:
    """The env_params of a one-hand training task, as the harness server expands them."""
    params = with_preset(training_table_rules() | {"seats": SEATS, "total_hands": 1})
    if table_rules(params).table_talk:  # every seat's tool gains `say`, as on the harness server
        params = params | {"submit_fields": talk_fields(params.get("submit_fields"))}
    return params


PARAMS = table_params()
RULES = table_rules(PARAMS)
CHAT_TOOLS = table_tools(PARAMS.get("submit_fields"), RULES.scratchpad, RULES.scratchpad_chars)
TOOLS = [{"type": "function", **tool["function"]} for tool in CHAT_TOOLS]  # AC2's form, as in opsd/tools.json


def chips(bb: float) -> int:
    """PokerBench big blinds to event chips, rounded half up to a whole chip."""
    return math.floor(bb * SCALE + 0.5)


class Dropped(Exception):
    """The row can't be replayed faithfully; the message is the reason it's counted under."""


@dataclass(frozen=True)
class Move:
    seat: int
    street: str
    kind: str  # fold, check, call, bet, raise or allin
    to_bb: float | None = None


@dataclass
class Spot:
    source: dict[str, Any]
    hero_seat: int
    hole: list[str]
    board: list[str]
    moves: list[Move]
    in_line: set[int]  # seats the line mentions; the rest fold preflop
    street: str
    pot_bb: float
    label: str
    hero_side: str | None = None  # IP or OOP, postflop spots only


def _cards(text: str) -> list[str]:
    cards = re.findall(r"[2-9TJQKA][cdhs]", text)
    if "".join(cards) != text.strip():
        raise Dropped("bad_cards")
    return cards


def _preflop_moves(line: str) -> list[Move]:
    """`UTG/2.0bb/BTN/call/SB/13.0bb/BB/allin/...`: position and action pairs; `Xbb` is a raise to X BB."""
    tokens = [t for t in line.split("/") if t]
    if len(tokens) % 2:
        raise Dropped("bad_line")
    moves = []
    for position, action in zip(tokens[::2], tokens[1::2], strict=True):
        if position not in POSITIONS:
            raise Dropped("bad_line")
        if amount := re.fullmatch(r"(\d+(?:\.\d+)?)bb", action):
            moves.append(Move(POSITIONS[position], "preflop", "raise", float(amount[1])))
        elif action in ("fold", "check", "call", "allin"):
            moves.append(Move(POSITIONS[position], "preflop", action))
        else:
            raise Dropped("bad_line")
    return moves


def preflop_spot(row: dict[str, str], source: dict[str, Any]) -> Spot:
    moves = _preflop_moves(row["prev_line"])
    hero = POSITIONS.get(row["hero_pos"])
    if hero is None:
        raise Dropped("bad_position")
    return Spot(
        source=source,
        hero_seat=hero,
        hole=_cards(row["hero_holding"]),
        board=[],
        moves=moves,
        in_line={m.seat for m in moves} | {hero},
        street="preflop",
        pot_bb=float(row["pot_size"]),
        label=row["correct_decision"],
    )


def postflop_spot(row: dict[str, str], source: dict[str, Any]) -> Spot:
    moves = _preflop_moves(row["preflop_action"])
    folded = {m.seat for m in moves if m.kind == "fold"}
    live = sorted({m.seat for m in moves} - folded)
    if len(live) != 2:
        raise Dropped("not_heads_up")
    sides = {"OOP": live[0], "IP": live[1]}  # the small blind acts first after the flop, the button last
    board = _cards(row["board_flop"]) + _cards(row["board_turn"]) + _cards(row["board_river"])
    street = "flop"
    for token in row["postflop_action"].split("/"):
        if token == "dealcards" or not token:
            continue
        if re.fullmatch(r"[2-9TJQKA][cdhs]", token):
            street = {"flop": "turn", "turn": "river"}.get(street, "past_river")
            if street == "past_river" or token != board[{"turn": 3, "river": 4}[street]]:
                raise Dropped("board_mismatch")
            continue
        parts = token.split("_")
        if parts[0] not in sides or len(parts) not in (2, 3):
            raise Dropped("bad_line")
        kind = parts[1].lower()
        if kind not in ("check", "call", "fold", "bet", "raise") or (len(parts) == 3) != (kind in ("bet", "raise")):
            raise Dropped("bad_line")
        moves.append(Move(sides[parts[0]], street, kind, float(parts[2]) if len(parts) == 3 else None))
    target = STREETS.get(row["evaluation_at"])
    if target != street:
        raise Dropped("street_mismatch")
    hero_side = row["hero_position"]
    if hero_side not in sides:
        raise Dropped("bad_position")
    return Spot(
        source=source,
        hero_seat=sides[hero_side],
        hole=_cards(row["holding"]),
        board=board,
        moves=moves,
        in_line=set(live),
        street=street,
        pot_bb=float(row["pot_size"]),
        label=row["correct_decision"],
        hero_side=hero_side,
    )


def spot_from_row(row: dict[str, str], source: dict[str, Any]) -> Spot:
    if "prev_line" in row:
        return preflop_spot(row, source)
    if "postflop_action" in row:
        return postflop_spot(row, source)
    raise Dropped("unknown_format")


def decide(kind: str, to_bb: float | None, legal: LegalActions) -> Decision:
    """A line or label action as the event table applies it, or `Dropped` if the table wouldn't accept it as given."""
    if kind == "allin":
        if legal.raise_type:
            return Decision(legal.raise_type, legal.max_raise_to)
        return Decision("call")  # can't raise: all in is a call
    if kind in ("bet", "raise"):
        if legal.raise_type is None:
            raise Dropped("raise_not_legal")
        amount = chips(to_bb or 0)
        if not legal.min_raise_to <= amount <= legal.max_raise_to:
            raise Dropped("amount_out_of_range")
        return Decision(legal.raise_type, amount)
    if kind == "call":
        return Decision("check" if legal.can_check else "call")
    if kind == "check":
        if not legal.can_check:
            raise Dropped("check_facing_bet")
        return Decision("check")
    if kind == "fold":
        if not legal.can_fold:
            raise Dropped("fold_with_free_check")
        return Decision("fold")
    raise Dropped("bad_action")


def label_move(label: str) -> tuple[str, float | None]:
    """A PokerBench decision (`call`, `3.0bb`, `allin`, `Check`, `Bet 24`, `Raise 29`) as a kind and "to" amount."""
    text = label.strip()
    if amount := re.fullmatch(r"(\d+(?:\.\d+)?)bb", text):
        return "raise", float(amount[1])
    if sized := re.fullmatch(r"(Bet|Raise) (\d+(?:\.\d+)?)", text):
        return sized[1].lower(), float(sized[2])
    if text.lower() in ("fold", "check", "call", "allin"):
        return text.lower(), None
    raise Dropped("bad_label")


class Script:
    """The spot's actions in order. A seat the line doesn't mention folds at its first preflop turn."""

    def __init__(self, spot: Spot) -> None:
        self.moves = deque(spot.moves)
        self.in_line = spot.in_line

    def next_for(self, seat: int, ctx: DecisionContext) -> Move | None:
        """The seat's scripted action here, or None when the line is used up (the hero's decision)."""
        if self.moves and self.moves[0].seat == seat:
            move = self.moves.popleft()
            if move.street != ctx.street:
                raise Dropped("street_mismatch")
            return move
        if ctx.street == "preflop" and seat not in self.in_line:
            return Move(seat, "preflop", "fold")
        if self.moves:
            raise Dropped("line_out_of_order")
        return None


@dataclass
class Captured(BaseException):  # not an Exception: nothing between here and `replay` may swallow it
    messages: list[dict[str, Any]]
    ctx: DecisionContext


@dataclass
class LineSeat:
    """An opponent who plays the spot's line."""

    name: str
    seat: int
    script: Script

    async def decide(self, ctx: DecisionContext, history: str) -> Decision:
        move = self.script.next_for(self.seat, ctx)
        if move is None:
            raise Dropped("line_ends_before_hero")
        return decide(move.kind, move.to_bb, ctx.legal)


@dataclass
class HeroSeat:
    """The trained seat: a real `ModelSeat` whose replies are the line's earlier hero actions, as tool calls."""

    name: str
    seat: int
    script: Script
    model: ModelSeat = field(init=False)
    _plan: Decision | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        # As the harness server builds a policy seat (its relay call is replaced by the scripted reply).
        self.model = ModelSeat(self.name, self._reply, RULES, model="ac2/policy")

    async def _reply(self, messages: list[dict[str, Any]]) -> Completion:
        assert self._plan is not None
        arguments = {"action": self._plan.type}
        if self._plan.amount is not None:
            arguments["amount"] = self._plan.amount
        text = json.dumps(arguments)
        call = {"id": f"call_{self.model.turn}", "type": "function", "function": {"name": SUBMIT_ACTION, "arguments": text}}
        return Completion(text="", tool_arguments=text, tool_calls=[call])

    async def decide(self, ctx: DecisionContext, history: str) -> Decision:
        move = self.script.next_for(self.seat, ctx)
        if move is None:  # the spot: the messages the seat would send now
            raise Captured(self.model._next_messages(ctx), ctx)
        self._plan = decide(move.kind, move.to_bb, ctx.legal)
        decision = await self.model.decide(ctx, history)
        if decision.model_response.get("format_error"):
            raise Dropped("hero_line_rejected")
        return decision


def deal_spot(runner: HandRunner, spot: Spot, rng: random.Random) -> None:
    """Replace the runner's seeded dealing: the hero gets the spot's cards, the board comes from the spot (then spares),
    and every other seat gets random unused cards."""
    used = set(spot.hole) | set(spot.board)
    if len(used) != len(spot.hole) + len(spot.board):
        raise Dropped("duplicate_cards")
    spares = [card for card in Deck.STANDARD if repr(card) not in used]
    rng.shuffle(spares)
    hole = {seat: list(Card.parse("".join(spot.hole))) if seat == spot.hero_seat else [spares.pop(), spares.pop()] for seat in runner.order}
    board = deque(c for text in spot.board for c in Card.parse(text))
    state = runner.state

    def deal() -> None:
        while True:
            if state.can_deal_hole():
                i = state.hole_dealee_index
                state.deal_hole(hole[runner.order[i]][len(state.hole_cards[i])])
            elif state.can_burn_card():
                state.burn_card(spares.pop())
            elif state.can_deal_board():
                count = state.board_dealing_count or 0
                state.deal_board(tuple(board.popleft() if board else spares.pop() for _ in range(count)))
            else:
                return

    runner._deal = deal  # type: ignore[method-assign]


async def replay(spot: Spot, pot_tolerance: int = POT_TOLERANCE) -> dict[str, Any]:
    """The kept record for one spot, or `Dropped` with the reason."""
    script = Script(spot)
    players = {
        seat: HeroSeat(f"seat{seat}", seat, script) if seat == spot.hero_seat else LineSeat(f"seat{seat}", seat, script)
        for seat in range(SEATS)
    }
    runner = HandRunner(
        players=players,
        stacks=dict.fromkeys(range(SEATS), RULES.starting_stack),
        button_seat=SEATS - 1,
        hand_number=1,
        total_hands=RULES.total_hands,
        seed=0,
        small_blind=RULES.small_blind,
        big_blind=RULES.big_blind,
    )
    deal_spot(runner, spot, random.Random(json.dumps(spot.source, sort_keys=True)))
    try:
        await runner.play(dict.fromkeys(range(SEATS), ""), 0)
    except Captured as captured:
        messages, ctx = captured.messages, captured.ctx
    else:
        raise Dropped("hand_ended_before_hero")
    if ctx.street != spot.street:
        raise Dropped("street_mismatch")
    if abs(ctx.pot - spot.pot_bb * SCALE) > pot_tolerance:
        raise Dropped("pot_mismatch")
    try:
        kind, to_bb = label_move(spot.label)
        label = decide(kind, to_bb, ctx.legal)
    except Dropped as dropped:
        raise Dropped(f"label_{dropped}") from None
    if invalid_action(label, ctx):  # the event's strict rules must accept the label as given
        raise Dropped("label_rejected")
    if label.type == "call":
        solver_text = f"call {ctx.legal.call_amount}"
    elif label.amount is not None:
        solver_text = f"{label.type} to {label.amount}"
    else:
        solver_text = label.type
    return {
        "messages": messages,
        "tools": TOOLS,
        "label": {"action": label.type, **({"amount": label.amount} if label.amount is not None else {})},
        "solver_text": solver_text,
        "source": spot.source,
        "street": spot.street,
        "hero_position": POSITION_NAMES[spot.hero_seat],
        "hero_side": spot.hero_side,
        "hero_seat": spot.hero_seat,
        "pot": ctx.pot,
        "legal": {
            "can_fold": ctx.legal.can_fold,
            "can_check": ctx.legal.can_check,
            "call_amount": ctx.legal.call_amount,
            "raise_type": ctx.legal.raise_type,
            "min_raise_to": ctx.legal.min_raise_to,
            "max_raise_to": ctx.legal.max_raise_to,
        },
    }


async def convert_row(row: dict[str, str], source: dict[str, Any], pot_tolerance: int = POT_TOLERANCE) -> dict[str, Any]:
    return await replay(spot_from_row(row, source), pot_tolerance)


async def convert(
    paths: list[Path], out, limit: int | None = None, pot_tolerance: int = POT_TOLERANCE
) -> dict[str, Any]:
    stats: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    for path in paths:
        with path.open(newline="", encoding="utf-8") as rows:
            for index, row in enumerate(csv.DictReader(rows)):
                if limit is not None and index >= limit:
                    break
                stats["rows"] += 1
                try:
                    source = {"file": path.name, "row": int(row.get("") or index)}
                    record = await convert_row(row, source, pot_tolerance)
                except Dropped as dropped:
                    stats[f"dropped:{dropped}"] += 1
                    continue
                stats["kept"] += 1
                labels[f"{record['street']}:{record['label']['action']}"] += 1
                out.write(json.dumps(record) + "\n")
    return {**dict(sorted(stats.items())), "labels": dict(sorted(labels.items()))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="+", type=Path, help="PokerBench *_game_scenario_information.csv files")
    parser.add_argument("--out", type=Path, required=True, help="JSONL to write")
    parser.add_argument("--limit", type=int, help="rows per file")
    parser.add_argument(
        "--pot-tolerance", type=int, default=POT_TOLERANCE, help="chips the replayed pot may differ from PokerBench's"
    )
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as out:
        stats = asyncio.run(convert(args.csv, out, args.limit, args.pot_tolerance))
    print(json.dumps(stats, indent=1))
    if not stats.get("kept"):
        sys.exit("no spot was kept")


if __name__ == "__main__":
    main()
