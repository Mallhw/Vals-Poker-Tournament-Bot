"""Style bots: rule-based opponents with distinct, tunable play styles, for self-play training leagues.

A `StyleBot` plays one preset style (`STYLES`) with its numbers drawn around the preset per task, so no two tables
face exactly the same bot. It sees only what any player sees (its own cards, the board, the pot, the actions):

- **Preflop**, by Bill Chen's hand score (`spots.chen_score`) against thresholds that loosen with position: open-raise,
  limp, call a raise, 3-bet (sometimes light), 4-bet or shove.
- **Postflop**, by Monte Carlo equity against random hands for each live opponent (never their real cards): bet for
  value above a threshold, c-bet or bluff at set frequencies, call when equity beats the pot odds plus a margin,
  raise for value or as a bluff.

Every random choice is seeded by the bot's key (deal, policy seats and seat), the hand and the decision's index, so
every replay of a deal meets the same bot making the same draws in the same spots.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, fields, replace
from typing import Any

from ..spots import chen_score
from .table import Decision, DecisionContext

RANK_VALUE = {r: i for i, r in enumerate("23456789TJQKA", start=2)}
EQUITY_SAMPLES = 120


@dataclass(frozen=True)
class Style:
    # Preflop: Chen-score thresholds. `open` is for early position; each position later loosens it by `loosen`.
    open: float
    loosen: float  # Chen points per position step (UTG -> HJ -> CO -> BTN; SB as CO)
    limp: float  # unopened and not good enough to raise: limp with at least this (99 = never)
    call: float  # call an open raise with at least this
    three_bet: float
    four_bet: float  # facing a 3-bet or more: re-raise (or shove) with at least this, call with `four_bet - 2`
    light_three_bet: float  # chance to 3-bet a hand that would only call
    open_bb: float  # open-raise size in big blinds
    max_call_bb: float  # never call a preflop raise to more than this many big blinds without a 4-bet hand
    # Postflop: equity against random hands
    value_equity: float  # bet when checked to, or raise when bet into, with at least this
    raise_equity: float  # facing a bet: raise for value with at least this
    call_margin: float  # call when equity >= pot odds + this (negative calls light)
    cbet: float  # chance to bet the flop as the preflop raiser
    bluff: float  # chance to bet with a weak hand when checked to
    bluff_raise: float  # chance to raise as a bluff facing a bet
    bet_fraction: float  # bet size as a fraction of the pot
    raise_multiple: float  # raise to this multiple of the bet faced


STYLES: dict[str, Style] = {
    "nit": Style(open=10, loosen=0.5, limp=99, call=10, three_bet=12, four_bet=14, light_three_bet=0.0, open_bb=3,
                 max_call_bb=12, value_equity=0.72, raise_equity=0.85, call_margin=0.08, cbet=0.45, bluff=0.02,
                 bluff_raise=0.0, bet_fraction=0.5, raise_multiple=3),
    "tag": Style(open=9, loosen=1, limp=99, call=8, three_bet=11, four_bet=13, light_three_bet=0.05, open_bb=2.5,
                 max_call_bb=25, value_equity=0.6, raise_equity=0.76, call_margin=0.02, cbet=0.65, bluff=0.08,
                 bluff_raise=0.03, bet_fraction=0.6, raise_multiple=3),
    "lag": Style(open=7, loosen=1.2, limp=99, call=6, three_bet=9, four_bet=11, light_three_bet=0.12, open_bb=2.5,
                 max_call_bb=35, value_equity=0.55, raise_equity=0.68, call_margin=-0.02, cbet=0.78, bluff=0.2,
                 bluff_raise=0.08, bet_fraction=0.7, raise_multiple=3),
    "station": Style(open=11, loosen=0.5, limp=3, call=4, three_bet=15, four_bet=16, light_three_bet=0.0, open_bb=3,
                     max_call_bb=60, value_equity=0.75, raise_equity=0.92, call_margin=-0.2, cbet=0.1, bluff=0.0,
                     bluff_raise=0.0, bet_fraction=0.4, raise_multiple=2.5),
    "loose_passive": Style(open=11, loosen=0.5, limp=5, call=6, three_bet=14, four_bet=16, light_three_bet=0.0,
                           open_bb=3, max_call_bb=30, value_equity=0.68, raise_equity=0.88, call_margin=-0.08,
                           cbet=0.25, bluff=0.03, bluff_raise=0.01, bet_fraction=0.45, raise_multiple=2.5),
    "maniac": Style(open=3, loosen=0.5, limp=99, call=3, three_bet=7, four_bet=9, light_three_bet=0.3, open_bb=4,
                    max_call_bb=100, value_equity=0.45, raise_equity=0.55, call_margin=-0.05, cbet=0.9, bluff=0.45,
                    bluff_raise=0.25, bet_fraction=1.0, raise_multiple=3.5),
}

THRESHOLDS = ("open", "limp", "call", "three_bet", "four_bet")
FREQUENCIES = ("light_three_bet", "cbet", "bluff", "bluff_raise")
SIZES = ("open_bb", "bet_fraction", "raise_multiple", "max_call_bb")
EQUITIES = ("value_equity", "raise_equity")


def varied(style: Style, rng: random.Random) -> Style:
    """The preset with its numbers drawn around it: thresholds +-1 Chen point, frequencies x0.7-1.3, sizes x0.85-1.15,
    equity thresholds +-0.03, so each table's bot is a little different but keeps its style."""
    changes: dict[str, float] = {}
    for f in fields(style):
        value = getattr(style, f.name)
        if f.name in THRESHOLDS and value < 99:
            changes[f.name] = value + rng.choice((-1, 0, 0, 1))
        elif f.name in FREQUENCIES:
            changes[f.name] = min(1.0, value * rng.uniform(0.7, 1.3))
        elif f.name in SIZES:
            changes[f.name] = value * rng.uniform(0.85, 1.15)
        elif f.name in EQUITIES:
            changes[f.name] = min(0.97, value + rng.uniform(-0.03, 0.03))
    return replace(style, **changes)


# --- a small, fast 7-card evaluator (pokerkit's is ~0.4 ms a hand, too slow for Monte Carlo on the event loop) ---


def _straight_top(ranks: set[int]) -> int:
    """Highest card of the best straight in these ranks, or 0."""
    if 14 in ranks:
        ranks = ranks | {1}
    run, best = 0, 0
    for r in range(1, 15):
        run = run + 1 if r in ranks else 0
        if run >= 5:
            best = r
    return best


def hand_value(cards: list[str]) -> tuple[int, ...]:
    """Comparable value of the best five of these cards: (category, tiebreak ranks...)."""
    ranks = [RANK_VALUE[c[0]] for c in cards]
    suits: dict[str, list[int]] = {}
    for card, rank in zip(cards, ranks):
        suits.setdefault(card[1], []).append(rank)
    flush = next((sorted(s, reverse=True) for s in suits.values() if len(s) >= 5), None)
    if flush:
        top = _straight_top(set(flush))
        if top:
            return (8, top)
    counts: dict[int, int] = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    groups = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    if groups[0][1] == 4:
        kicker = max(r for r in counts if r != groups[0][0])
        return (7, groups[0][0], kicker)
    if groups[0][1] == 3 and len(groups) > 1 and groups[1][1] >= 2:
        return (6, groups[0][0], groups[1][0])
    if flush:
        return (5, *flush[:5])
    top = _straight_top(set(ranks))
    if top:
        return (4, top)
    if groups[0][1] == 3:
        return (3, groups[0][0], *sorted((r for r in counts if r != groups[0][0]), reverse=True)[:2])
    if groups[0][1] == 2 and len(groups) > 1 and groups[1][1] == 2:
        pairs = sorted((r for r, n in counts.items() if n == 2), reverse=True)[:2]
        return (2, *pairs, max(r for r in counts if r not in pairs))
    if groups[0][1] == 2:
        return (1, groups[0][0], *sorted((r for r in counts if r != groups[0][0]), reverse=True)[:3])
    return (0, *sorted(ranks, reverse=True)[:5])


DECK = [r + s for r in "23456789TJQKA" for s in "cdhs"]


def equity_vs_random(hole: list[str], board: list[str], opponents: int, rng: random.Random,
                     samples: int = EQUITY_SAMPLES) -> float:
    """Share of the pot `hole` wins against `opponents` random hands over random runouts."""
    if opponents <= 0:
        return 1.0
    dead = set(hole) | set(board)
    deck = [c for c in DECK if c not in dead]
    need = 5 - len(board)
    total = 0.0
    for _ in range(samples):
        drawn = rng.sample(deck, need + 2 * opponents)
        runout = board + drawn[:need]
        mine = hand_value(hole + runout)
        best, ties = True, 1
        for i in range(opponents):
            theirs = hand_value(drawn[need + 2 * i : need + 2 * i + 2] + runout)
            if theirs > mine:
                best = False
                break
            if theirs == mine:
                ties += 1
        if best:
            total += 1 / ties
    return total / samples


class StyleBot:
    """A rule-based opponent of one style (see the module docstring)."""

    def __init__(self, name: str, style: str, key: str, big_blind: int = 2) -> None:
        if style not in STYLES:
            raise ValueError(f"unknown style {style!r}; known: {sorted(STYLES)}")
        self.name = name
        self.style_name = style
        self.key = key
        self.big_blind = big_blind
        self.style = varied(STYLES[style], random.Random(f"style:{key}"))
        self._decisions = 0
        self._raised_preflop: dict[int, bool] = {}  # hand number -> this bot made the last preflop raise

    # -- helpers --

    def _mine(self, ctx: DecisionContext) -> int:
        return next((s.bet for s in ctx.seats if s.seat == ctx.seat), 0)

    def _raise(self, ctx: DecisionContext, to: float) -> Decision:
        legal = ctx.legal
        if not legal.raise_type:
            return Decision("check" if legal.can_check else "call")
        amount = int(round(to))
        if amount >= 0.6 * legal.max_raise_to:  # most of the stack: just shove
            amount = legal.max_raise_to
        return Decision(legal.raise_type, min(max(amount, legal.min_raise_to), legal.max_raise_to))

    def _passive(self, ctx: DecisionContext) -> Decision:
        return Decision("check" if ctx.legal.can_check else "call")

    def _give_up(self, ctx: DecisionContext) -> Decision:
        return Decision("check" if ctx.legal.can_check else "fold")

    # -- preflop --

    def _position_step(self, ctx: DecisionContext) -> int:
        """0 early .. 3 button. Positions are seats in action order from the small blind (0) to the button (n-1)."""
        live = sum(s.status != "out" for s in ctx.seats)
        from_button = live - 1 - ctx.position  # 0 = button
        if ctx.position == 0:  # small blind plays like the cutoff
            return 2
        return max(0, 3 - from_button)

    def _preflop(self, ctx: DecisionContext, rng: random.Random) -> Decision:
        s = self.style
        score = chen_score(ctx.hole_cards)
        bb = self.big_blind
        raises = sum(a["street"] == "preflop" and a["action"]["type"] in ("bet", "raise") for a in ctx.hand_actions)
        mine = self._mine(ctx)
        facing_to = mine + ctx.legal.call_amount  # the bet to match
        if raises == 0:
            limpers = sum(a["street"] == "preflop" and a["action"]["type"] == "call" for a in ctx.hand_actions)
            if score >= s.open - s.loosen * self._position_step(ctx):
                self._raised_preflop[ctx.hand_number] = True
                return self._raise(ctx, (s.open_bb + limpers) * bb)
            if ctx.legal.can_check:
                return Decision("check")
            if score >= s.limp:
                return Decision("call")
            return Decision("fold")
        if raises == 1:
            if score >= s.three_bet or (score >= s.call and rng.random() < s.light_three_bet):
                self._raised_preflop[ctx.hand_number] = True
                return self._raise(ctx, facing_to * 3)
            if score >= s.call and facing_to <= s.max_call_bb * bb:
                return self._passive(ctx)
            return self._give_up(ctx)
        if score >= s.four_bet:
            self._raised_preflop[ctx.hand_number] = True
            return self._raise(ctx, facing_to * 2.3)
        if score >= s.four_bet - 2 and facing_to <= s.max_call_bb * bb:
            return self._passive(ctx)
        return self._give_up(ctx)

    # -- postflop --

    def _postflop(self, ctx: DecisionContext, rng: random.Random) -> Decision:
        s = self.style
        legal = ctx.legal
        opponents = sum(s_.status in ("", "all-in") and s_.seat != ctx.seat for s_ in ctx.seats)
        equity = equity_vs_random(ctx.hole_cards, ctx.board, opponents, rng)
        mine = self._mine(ctx)
        if legal.can_check:
            aggressor = self._raised_preflop.get(ctx.hand_number, False) and ctx.street == "flop"
            if equity >= s.value_equity or (aggressor and rng.random() < s.cbet) or rng.random() < s.bluff:
                return self._raise(ctx, mine + s.bet_fraction * ctx.pot)
            return Decision("check")
        facing_to = mine + legal.call_amount
        odds = legal.call_amount / (ctx.pot + legal.call_amount)
        if equity >= s.raise_equity or rng.random() < s.bluff_raise:
            return self._raise(ctx, facing_to * s.raise_multiple)
        if equity >= odds + s.call_margin:
            return Decision("call")
        return Decision("fold")

    async def decide(self, ctx: DecisionContext, history: str) -> Decision:
        rng = random.Random(f"bot:{self.key}:{ctx.hand_number}:{self._decisions}")
        self._decisions += 1
        decision = self._preflop(ctx, rng) if ctx.street == "preflop" else self._postflop(ctx, rng)
        decision.model_response = {
            "model_name": f"bot/{self.style_name}",
            "provider": "scripted",
            "latency_seconds": 0.0,
            "in_tokens": 0,
            "out_tokens": 0,
            "reasoning_tokens": 0,
            "cost": 0.0,
            "reasoning": "",
            "fallback_used": False,
            "error": None,
        }
        return decision

    def describe(self) -> dict[str, Any]:
        return {"style": self.style_name} | {f.name: round(getattr(self.style, f.name), 3) for f in fields(self.style)}
