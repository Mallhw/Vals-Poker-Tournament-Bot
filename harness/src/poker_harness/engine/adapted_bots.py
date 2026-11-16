"""Rule bots adapted from two public bot repos, for self-play training leagues.

Reimplemented from the rules and numbers of the bots in these repos (their code is not copied; neither repo has a
license):

- Laosez/Texas-Holdem-AI (github.com/Laosez/Texas-Holdem-AI): the tournament hybrid bot's deep-stack play, in its
  survival (`hybrid_tag`) and aggro (`hybrid_lag`) modes, and `tag_punisher` / `wide_defender`.
- YosiefAbraham/pokerbot (github.com/YosiefAbraham/pokerbot): YOSIEF's play with no reads yet (`yosief`; each league
  table is one hand, so its opponent profiles never fill) and its range-based "GTO" bot (`range_gto`).

Their tournament-only parts (ICM, short-stack push/fold bands) don't apply at 100 BB and are left out. Where a source
gives no numbers (the hybrid's flat-calling charts and its middle-position opening range), the ranges here are ours.
Like `style_bot.StyleBot`, these bots see only what any player sees, and every random choice is seeded by the bot's
key, the hand and the decision's index.
"""

from __future__ import annotations

import json
import random
from functools import cache
from pathlib import Path
from typing import Any

from .style_bot import DECK, RANK_VALUE, equity_vs_random, hand_value
from .table import Decision, DecisionContext

RANKS = "AKQJT98765432"
STREET = {"flop": 0, "turn": 1, "river": 2}


# ---------------------------------------------------------------- hand classes, percentiles and ranges


def hand_class(cards: list[str]) -> str:
    """'AKs', 'AKo' or '77'."""
    (r1, s1), (r2, s2) = sorted(((c[0], c[1]) for c in cards), key=lambda c: RANKS.index(c[0]))
    if r1 == r2:
        return r1 + r2
    return r1 + r2 + ("s" if s1 == s2 else "o")


def _combos(cls: str) -> int:
    return 6 if len(cls) == 2 else 4 if cls.endswith("s") else 12


@cache
def percentiles() -> dict[str, float]:
    """Each class's start in the ranking by preflop equity against one random hand (0 = AA): 'top x' holds a class
    whose start is below x. The equities are precomputed (preflop_equity.json, 4,000 runouts per class)."""
    equity = json.loads((Path(__file__).with_name("preflop_equity.json")).read_text())
    start, out = 0, {}
    for cls in sorted(equity, key=lambda c: -equity[c]):
        out[cls] = start / 1326
        start += _combos(cls)
    return out


def in_top(cls: str, share: float) -> bool:
    return percentiles()[cls] < share


def _expand(token: str) -> set[str]:
    token = token.strip()
    if not token:
        return set()
    if "-" in token:  # 'A5s-A2s', 'TT-77', '98s-65s'
        a, b = (t.strip() for t in token.split("-"))
        if len(a) == 2:  # pairs
            hi, lo = sorted((RANKS.index(a[0]), RANKS.index(b[0])))
            return {RANKS[i] * 2 for i in range(hi, lo + 1)}
        suffix = a[2:]
        if a[0] == b[0]:  # fixed high card, kicker range
            hi, lo = sorted((RANKS.index(a[1]), RANKS.index(b[1])))
            return {a[0] + RANKS[i] + suffix for i in range(hi, lo + 1)}
        gap = RANKS.index(a[1]) - RANKS.index(a[0])  # connectors stepping down
        start, stop = RANKS.index(a[0]), RANKS.index(b[0])
        return {RANKS[i] + RANKS[i + gap] + suffix for i in range(start, stop + 1)}
    plus = token.endswith("+")
    token = token.rstrip("+")
    if len(token) == 2 and token[0] == token[1]:
        i = RANKS.index(token[0])
        return {RANKS[j] * 2 for j in range(0, i + 1)} if plus else {token}
    suffixes = [token[2:]] if len(token) == 3 else ["s", "o"]
    hi, kicker = token[0], token[1]
    if not plus:
        return {hi + kicker + s for s in suffixes}
    top = RANKS.index(hi) + 1
    return {hi + RANKS[k] + s for k in range(top, RANKS.index(kicker) + 1) for s in suffixes}


@cache
def hand_range(text: str) -> frozenset[str]:
    """'55+, ATs+, KJs+, QJs, A5s-A4s, AQ' as a set of hand classes."""
    out: set[str] = set()
    for token in text.split(","):
        out |= _expand(token)
    return frozenset(out)


# ---------------------------------------------------------------- what any player can read off the table


def order(ctx: DecisionContext) -> list[int]:
    """Seats in this hand's action order, small blind first and the button last."""
    live = sorted(s.seat for s in ctx.seats if s.status != "out")
    return [s for s in live if s > ctx.button_seat] + [s for s in live if s <= ctx.button_seat]


def position_name(position: int, players: int) -> str:
    if position == 0 and players > 2:
        return "SB"
    if position == 1 and players > 2:
        return "BB"
    from_button = players - 1 - position
    return {0: "BTN", 1: "CO", 2: "HJ", 3: "MP"}.get(from_button, "EP")


class Table:
    """The public facts a rule bot decides on."""

    def __init__(self, ctx: DecisionContext, big_blind: int) -> None:
        self.ctx = ctx
        self.bb = big_blind
        self.order = order(ctx)
        self.players = len(self.order)
        self.me = next(s for s in ctx.seats if s.seat == ctx.seat)
        pre = [a for a in ctx.hand_actions if a["street"] == "preflop"]
        at = [i for i, a in enumerate(pre) if a["action"]["type"] in ("bet", "raise")]
        self.raises = len(at)
        self.limpers = sum(a["action"]["type"] == "call" for a in pre[: at[0] if at else len(pre)])
        self.cold_callers = sum(a["action"]["type"] == "call" for a in pre[at[-1] + 1:]) if at else 0
        self.opener = pre[at[0]]["seat"] if at else None
        self.i_raised_preflop = bool(at) and pre[at[-1]]["seat"] == ctx.seat
        self.call = ctx.legal.call_amount
        self.to = self.me.bet + self.call  # the bet to match
        self.odds = self.call / (ctx.pot + self.call) if self.call else 0.0
        live_others = [s for s in ctx.seats if s.seat != ctx.seat and s.status in ("", "all-in")]
        self.opponents = len(live_others)
        deepest = max((s.stack + s.bet for s in live_others), default=0)
        self.effective = min(self.me.stack + self.me.bet, deepest)

    def position_of(self, seat: int) -> int:
        return self.order.index(seat)

    def name_of(self, seat: int) -> str:
        return position_name(self.position_of(seat), self.players)

    @property
    def position(self) -> str:
        return position_name(self.ctx.position, self.players)

    def in_position_of(self, seat: int) -> bool:
        """Acting after `seat` after the flop (the blinds act first)."""
        return self.ctx.position > self.position_of(seat) and self.position not in ("SB", "BB")

    def in_position(self) -> bool:
        live = [self.position_of(s.seat) for s in self.ctx.seats if s.seat != self.ctx.seat and s.status == ""]
        return all(self.ctx.position > p for p in live)


def draws(hole: list[str], board: list[str]) -> bool:
    """A flush draw or an open-ended straight draw that uses a hole card (before the river)."""
    if len(board) >= 5 or not board:
        return False
    cards = hole + board
    for suit in "cdhs":
        if sum(c[1] == suit for c in cards) == 4 and any(c[1] == suit for c in hole):
            return True
    ranks = {RANK_VALUE[c[0]] for c in cards}
    mine = {RANK_VALUE[c[0]] for c in hole}
    for low in range(2, 11):  # four in a row, open at both ends (not A-high)
        run = set(range(low, low + 4))
        if run <= ranks and run & mine and low + 3 < 14:
            return True
    return False


# ---------------------------------------------------------------- the bots


class RuleBot:
    style = "rule"

    def __init__(self, name: str, key: str, big_blind: int = 2) -> None:
        self.name = name
        self.key = key
        self.big_blind = big_blind
        self._decisions = 0

    def raise_to(self, ctx: DecisionContext, amount: float, shove_at: float = 0.7) -> Decision:
        legal = ctx.legal
        if not legal.raise_type:
            return Decision("check" if legal.can_check else "call")
        amount = int(round(amount))
        if amount >= shove_at * legal.max_raise_to:
            amount = legal.max_raise_to
        return Decision(legal.raise_type, min(max(amount, legal.min_raise_to), legal.max_raise_to))

    def jam(self, ctx: DecisionContext) -> Decision:
        legal = ctx.legal
        if legal.raise_type:
            return Decision(legal.raise_type, legal.max_raise_to)
        return Decision("call") if legal.call_amount else Decision("check")

    @staticmethod
    def passive(ctx: DecisionContext) -> Decision:
        return Decision("check" if ctx.legal.can_check else "call")

    @staticmethod
    def give_up(ctx: DecisionContext) -> Decision:
        return Decision("check" if ctx.legal.can_check else "fold")

    def preflop(self, ctx: DecisionContext, t: Table, rng: random.Random) -> Decision:
        raise NotImplementedError

    def postflop(self, ctx: DecisionContext, t: Table, rng: random.Random) -> Decision:
        raise NotImplementedError

    async def decide(self, ctx: DecisionContext, history: str) -> Decision:
        rng = random.Random(f"bot:{self.key}:{ctx.hand_number}:{self._decisions}")
        self._decisions += 1
        table = Table(ctx, self.big_blind)
        decision = self.preflop(ctx, table, rng) if ctx.street == "preflop" else self.postflop(ctx, table, rng)
        decision.model_response = {
            "model_name": f"bot/{self.style}", "provider": "scripted", "latency_seconds": 0.0, "in_tokens": 0,
            "out_tokens": 0, "reasoning_tokens": 0, "cost": 0.0, "reasoning": "", "fallback_used": False, "error": None,
        }
        return decision

    def describe(self) -> dict[str, Any]:
        return {"style": self.style}


class HybridBot(RuleBot):
    """Laosez's tournament hybrid, deep-stack play: tight-aggressive (survival) or loose-aggressive (aggro)."""

    EARLY = "55+, ATs+, KJs+, QJs, JTs, AJo+, KQo, A5s-A4s"
    MIDDLE = "44+, A8s+, KTs+, QTs+, JTs, T9s, 98s, ATo+, KJo+, A5s-A2s"  # ours: between the source's charts
    LATE = "22+, A2s+, K9s+, Q9s+, J9s+, T8s+, 98s-65s, A8o+, KTo+, QTo+, JTo"
    WIDER = LATE + ", A2o+, K8o+, Q8s, J8s, T7s, 97s, 86s, 75s, 54s, Q9o+, J9o, T9o"
    ISO = "88+, ATs+, AJo+, KJs+, QJs"
    ISO_AGGRO = ISO + ", 66-77, A9s, ATo, KQo, QTs, JTs"
    # Flat-calling an open, by the opener's position (ours; the source names a chart without numbers)
    FLAT = {"early": "JJ-77, AQs-ATs, KQs, AQo", "middle": "JJ-55, AQs-A9s, KQs-KJs, QJs, JTs, AQo-AJo, KQo",
            "late": "JJ-22, AQs-A2s, KQs-K9s, QJs-Q9s, JTs-J9s, T9s, 98s, 87s, AQo-ATo, KQo-KJo"}

    def __init__(self, name: str, key: str, big_blind: int = 2, aggro: bool = False) -> None:
        super().__init__(name, key, big_blind)
        self.aggro = aggro
        self.style = "hybrid_lag" if aggro else "hybrid_tag"

    def open_range(self, position: str) -> frozenset[str]:
        charts = [self.EARLY, self.MIDDLE, self.LATE, self.WIDER]
        step = {"EP": 0, "MP": 0, "HJ": 1, "SB": 1, "CO": 2, "BTN": 2, "BB": 2}[position] + (1 if self.aggro else 0)
        return hand_range(charts[min(step, 3)])

    def flat_range(self, opener: str) -> frozenset[str]:
        band = {"EP": "early", "MP": "early", "HJ": "middle", "CO": "late", "BTN": "late", "SB": "late"}.get(opener, "late")
        if self.aggro:
            band = {"early": "middle", "middle": "late"}.get(band, band)
        return hand_range(self.FLAT[band])

    def preflop(self, ctx, t, rng):
        hand = hand_class(ctx.hole_cards)
        bb = self.big_blind
        if t.raises == 0:
            if t.limpers == 0 and hand in self.open_range(t.position):
                size = {"CO": 2.2, "BTN": 2.2, "SB": 2.7}.get(t.position, 2.5)
                return self.raise_to(ctx, size * bb)
            if t.limpers and hand in hand_range(self.ISO_AGGRO if self.aggro else self.ISO):
                return self.raise_to(ctx, ((3.5 if t.position in ("CO", "BTN") else 4.5) + t.limpers) * bb)
            return self.give_up(ctx)
        opener = t.name_of(t.opener)
        if t.raises == 1:
            three = "QQ+, AK" + (", JJ, AQs" if opener in ("CO", "BTN", "SB") else "")
            three += ", TT, AJs, AQo, KQs" if self.aggro else ""
            if hand in hand_range(three):
                size = t.to * (3.0 if t.in_position_of(t.opener) else 3.8) + t.to * t.cold_callers
                if size >= 0.3 * t.effective:
                    return self.jam(ctx)
                return self.raise_to(ctx, size)
            flat = self.flat_range(opener)
            if t.position == "BB" and t.to <= 3 * bb:
                flat = flat | hand_range(self.LATE + ", 76s, 65s")
            return self.passive(ctx) if hand in flat else self.give_up(ctx)
        if t.raises == 2:
            if hand in hand_range("QQ+, AK"):
                return self.raise_to(ctx, t.to * 2.3)
            if hand in hand_range("JJ, AQs") and t.effective >= 40 * bb and t.position != "SB":
                return self.passive(ctx)
            return self.give_up(ctx)
        return self.jam(ctx) if hand in hand_range("AA, KK, AKs") else self.give_up(ctx)

    def postflop(self, ctx, t, rng):
        s = STREET[ctx.street]
        hu = t.opponents == 1
        eq = equity_vs_random(ctx.hole_cards, ctx.board, t.opponents, rng, samples=200)
        shift = -0.04 if self.aggro else 0.0
        cushion = [0.07, 0.06, 0.05][s] + (0.0 if hu else 0.04 if t.opponents == 2 else 0.08) + shift
        extra = 0.03 if t.opponents >= 3 else 0.0
        value_raise = ([0.69, 0.71, 0.74] if hu else [0.76, 0.79, 0.82])[s] + shift + extra
        value_bet = ([0.61, 0.64, 0.60] if hu else [0.70, 0.72, 0.68])[s] + shift + extra
        scale = 1.0 if self.aggro else 0.85  # survival sizes are smaller
        value_size, semi_size, bluff_size = ([0.55, 0.70, 0.75][s] * scale, [0.50, 0.65, 0.65][s] * scale,
                                             [0.35, 0.55, 0.75][s] * scale)
        semi = False
        if hu and s < 2:
            low, high = [(0.28, 0.52), (0.18, 0.40)][s]
            chance = ([0.65, 0.50] if self.aggro else [0.35, 0.22])[s] * (1.25 if draws(ctx.hole_cards, ctx.board) else 1)
            semi = low <= eq <= high and rng.random() < chance
        pot_after_call = ctx.pot + t.call
        if t.call:
            if eq >= value_raise:
                return self.raise_to(ctx, t.to + value_size * pot_after_call)
            if semi:
                return self.raise_to(ctx, t.to + semi_size * pot_after_call)
            return Decision("call") if eq >= t.odds + cushion else Decision("fold")
        if eq >= value_bet:
            return self.raise_to(ctx, value_size * ctx.pot)
        if semi:
            return self.raise_to(ctx, semi_size * ctx.pot)
        multiway = 0.5 if not hu else 1.0
        cbet = ([0.70, 0.45, 0.30] if self.aggro else [0.55, 0.30, 0.18])[s] * multiway
        if t.i_raised_preflop and rng.random() < cbet:
            return self.raise_to(ctx, semi_size * ctx.pot)
        bluff = ([0.18, 0.12, 0.22] if self.aggro else [0.06, 0.04, 0.06])[s] * multiway
        if rng.random() < bluff:
            return self.raise_to(ctx, bluff_size * ctx.pot)
        return Decision("check")


class PunisherBot(RuleBot):
    """Laosez's tag_punisher (and its wider cousin wide_defender): percentile ranges preflop, made hands after."""

    def __init__(self, name: str, key: str, big_blind: int = 2, wide: bool = False) -> None:
        super().__init__(name, key, big_blind)
        self.wide = wide
        self.style = "wide_defender" if wide else "tag_punisher"
        self.open_top = 0.26 if wide else 0.22
        self.continue_top = (0.48, 0.45, 0.12) if wide else (0.34, 0.26, 0.12)
        self.margin = 0.03 if wide else 0.05

    def pot_raise(self, ctx: DecisionContext, t: Table) -> Decision:
        return self.raise_to(ctx, t.to + ctx.pot + t.call)

    def preflop(self, ctx, t, rng):
        hand = hand_class(ctx.hole_cards)
        bb = self.big_blind
        if t.raises == 0:
            if in_top(hand, self.open_top):
                size = (3.5 + t.limpers) * bb
                if t.effective <= 1.5 * (ctx.pot + size):
                    return self.jam(ctx)
                return self.raise_to(ctx, size)
            return self.give_up(ctx)
        if in_top(hand, 0.08):
            return self.passive(ctx) if rng.random() < 0.15 else self.pot_raise(ctx, t)  # sometimes a trap
        top = self.continue_top[0 if t.odds < 0.25 else 1 if t.odds < 0.40 else 2]
        eq = equity_vs_random(ctx.hole_cards, [], t.opponents, rng)
        if in_top(hand, top) and eq >= t.odds + self.margin:
            return self.passive(ctx)
        if self.wide and t.raises == 1 and in_top(hand, 0.5) and rng.random() < 0.12:
            return self.pot_raise(ctx, t)  # a re-steal
        return self.give_up(ctx)

    def postflop(self, ctx, t, rng):
        category = hand_value(ctx.hole_cards + ctx.board)[0]  # the board's own pairs count, as in the source
        if category >= 2:
            return self.pot_raise(ctx, t) if t.call else self.raise_to(ctx, 0.75 * ctx.pot)
        if category == 1 and t.call:
            return Decision("call") if t.odds <= 0.30 else Decision("fold")
        return self.give_up(ctx)


class YosiefBot(RuleBot):
    """YOSIEF with no reads yet: position ranges, set-mining, big value bets with strong hands."""

    style = "yosief"
    OPEN = {"EP": "88+, AQ+", "MP": "88+, AJ+", "HJ": "88+, AJ+", "CO": "88+, AT+", "BTN": "88+, AT+", "SB": "88+, AT+"}
    STEAL_TOP = {"CO": 0.27, "BTN": 0.35}
    SB_STEAL = "KQ, KJs, QJs, A9s-A2s"
    BB_DEFEND = "22+, AT+, A9s-A2s, KQ, KJs, QJs, JTs"
    PREMIUM = "AA, KK, AKs"

    def preflop(self, ctx, t, rng):
        hand = hand_class(ctx.hole_cards)
        bb = self.big_blind
        if t.raises == 0:
            opens = hand in hand_range(self.OPEN.get(t.position, "88+, AT+"))
            opens = opens or (t.position in self.STEAL_TOP and t.limpers == 0 and in_top(hand, self.STEAL_TOP[t.position]))
            opens = opens or (t.position == "SB" and t.limpers == 0 and hand in hand_range(self.SB_STEAL))
            if opens:
                return self.capped(ctx, t, hand, (4 + t.limpers) * bb)
            if hand in hand_range("22-77") and not ctx.legal.can_check:
                return Decision("call")  # limp
            return self.give_up(ctx)
        if t.raises == 1:
            if hand in hand_range("QQ+, AK"):
                return self.capped(ctx, t, hand, t.to * (3.5 if t.position in ("SB", "BB") else 3))
            if hand in hand_range("TT-JJ, AQ"):
                return self.passive(ctx)
            if hand in hand_range("99-22") and 10 * t.call <= t.effective:
                return self.passive(ctx)  # set-mining
            if t.position == "BB" and t.to <= 2.5 * bb and hand in hand_range(self.BB_DEFEND):
                return self.passive(ctx)
            return self.give_up(ctx)
        if hand in hand_range("KK+, AKs"):
            return self.jam(ctx)
        if hand in hand_range("QQ, AKo") and t.call <= 0.35 * (t.me.stack + t.me.bet):
            return self.passive(ctx)
        return self.give_up(ctx)

    def capped(self, ctx: DecisionContext, t: Table, hand: str, amount: float) -> Decision:
        """Raises add at most twice the pot (premiums exempt); within 30% of all-in it shoves."""
        if hand not in hand_range(self.PREMIUM):
            amount = min(amount, t.to + 2 * ctx.pot)
        return self.raise_to(ctx, amount)

    def bucket(self, hole: list[str], board: list[str]) -> str:
        value = hand_value(hole + board)
        category = value[0]
        mine = sorted((RANK_VALUE[c[0]] for c in hole), reverse=True)
        top = sorted({RANK_VALUE[c[0]] for c in board}, reverse=True)
        pocket = mine[0] == mine[1]
        if category >= 4 or (pocket and mine[0] in top) or (category == 2 and not pocket and set(mine) == set(top[:2])):
            return "monster"
        if category == 3 or (pocket and mine[0] > top[0]) or (category == 2 and not pocket and set(mine) <= set(top)):
            return "strong"
        if top[0] in mine and not pocket:
            kicker = mine[1] if mine[0] == top[0] else mine[0]
            return "strong" if kicker >= RANK_VALUE["T"] else "medium"
        if pocket:
            return "small"  # a pocket pair under the board: check-fold
        if set(mine) & set(top):
            return "medium"
        return "draw" if draws(hole, board) else "air"

    def range_equity(self, ctx: DecisionContext, opponents: int, rng: random.Random, top: float = 0.30,
                     samples: int = 150) -> float:
        """Equity against opponents dealt from the top `top` of hands (YOSIEF's default VPIP is 30%)."""
        if opponents <= 0:
            return 1.0
        dead = set(ctx.hole_cards) | set(ctx.board)
        deck = [c for c in DECK if c not in dead]
        need = 5 - len(ctx.board)
        total = 0.0
        for _ in range(samples):
            used: set[str] = set()
            hands = []
            for _ in range(opponents):
                free = [c for c in deck if c not in used]
                for tries in range(40):  # a hand from the top of the range (after 40 misses, any hand)
                    a, b = rng.sample(free, 2)
                    if in_top(hand_class([a, b]), top) or tries == 39:
                        break
                hands.append([a, b])
                used |= {a, b}
            left = [c for c in deck if c not in used]
            runout = ctx.board + rng.sample(left, need)
            mine = hand_value(ctx.hole_cards + runout)
            values = [hand_value(h + runout) for h in hands]
            if all(mine > v for v in values):
                total += 1
            elif all(mine >= v for v in values):
                total += 1 / (1 + sum(mine == v for v in values))
        return total / samples

    def postflop(self, ctx, t, rng):
        s = STREET[ctx.street]
        kind = self.bucket(ctx.hole_cards, ctx.board)
        category = hand_value(ctx.hole_cards + ctx.board)[0]
        no_shove = 0.69 if category < 3 else None  # all-in only with trips or better

        def bet(fraction_of_pot: float, base: int = 0) -> Decision:
            amount = base + fraction_of_pot * (ctx.pot + t.call)
            if no_shove and ctx.legal.raise_type:
                amount = min(amount, no_shove * ctx.legal.max_raise_to)
            return self.raise_to(ctx, amount, shove_at=0.7 if not no_shove else 1.01)

        if not t.call:
            if kind == "monster":
                return bet([1.5, 1.0, 0.75][s])
            if kind == "strong":
                return bet([1.0, 0.75, 0.6][s])
            return Decision("check")
        if kind == "monster":
            return self.raise_to(ctx, t.to * 3)
        if kind == "small":
            return Decision("fold")
        eq = self.range_equity(ctx, t.opponents, rng)
        return Decision("call") if eq >= t.odds + 0.05 else Decision("fold")


class RangeGTOBot(RuleBot):
    """Yosief's range-based "GTO" bot: percentile opening widths, 3-bets and calls as shares of the opener's width."""

    style = "range_gto"
    WIDTH = {"EP": 0.12, "MP": 0.17, "HJ": 0.21, "CO": 0.28, "BTN": 0.45, "SB": 0.40, "BB": 0.40}

    def mixed(self, hand: str, share: float, rng: random.Random) -> bool:
        """In the top `share`, mixing 50/50 within 3 points of the edge."""
        start = percentiles()[hand]
        if abs(start - share) < 0.03:
            return rng.random() < 0.5
        return start < share

    def preflop(self, ctx, t, rng):
        hand = hand_class(ctx.hole_cards)
        bb = self.big_blind
        if t.raises == 0:
            if t.position != "BB" and self.mixed(hand, self.WIDTH[t.position], rng):
                return self.raise_to(ctx, ((3 if t.position == "SB" else 2.5) + t.limpers) * bb)
            return self.give_up(ctx)
        if t.raises == 1:
            width = self.WIDTH.get(t.name_of(t.opener), 0.3)
            ip = t.in_position_of(t.opener)
            call_share = 0.75 * width if ip else 0.4 * width
            suited_bluff = hand.endswith("s") and not in_top(hand, call_share) and in_top(hand, width) and rng.random() < 0.35
            if self.mixed(hand, 0.22 * width, rng) or suited_bluff:
                return self.raise_to(ctx, t.to * (3 if ip else 4))
            return self.passive(ctx) if self.mixed(hand, call_share, rng) else self.give_up(ctx)
        if in_top(hand, 0.035):
            return self.raise_to(ctx, t.to * 2.3)
        return self.give_up(ctx)

    def postflop(self, ctx, t, rng):
        s = STREET[ctx.street]
        size = [0.33, 0.66, 0.75][s]
        eq = equity_vs_random(ctx.hole_cards, ctx.board, t.opponents, rng, samples=200)
        if t.call:
            if eq >= 0.80 or (eq >= 0.70 and rng.random() < 0.5):
                return self.raise_to(ctx, t.to * 3)
            return Decision("call") if eq >= t.odds + 0.03 else Decision("fold")
        if eq >= (0.62 if t.opponents == 1 else 0.72):
            return self.raise_to(ctx, size * ctx.pot)
        if eq < 0.25 and rng.random() < 0.85 * size / (1 + size):
            return self.raise_to(ctx, size * ctx.pot)
        return Decision("check")


def bot_styles() -> set[str]:
    """Every style a league entry may name: our presets (style_bot.STYLES) and the adapted bots."""
    from .style_bot import STYLES

    return set(STYLES) | set(ADAPTED)


def make_bot(style: str, name: str, key: str, big_blind: int = 2) -> Any:
    from .style_bot import STYLES, StyleBot

    if style in STYLES:
        return StyleBot(name, style, key, big_blind)
    if style in ADAPTED:
        return ADAPTED[style](name, key, big_blind)
    raise ValueError(f"unknown bot style {style!r}; known: {sorted(bot_styles())}")


ADAPTED: dict[str, Any] = {
    "hybrid_tag": lambda name, key, bb: HybridBot(name, key, bb, aggro=False),
    "hybrid_lag": lambda name, key, bb: HybridBot(name, key, bb, aggro=True),
    "tag_punisher": lambda name, key, bb: PunisherBot(name, key, bb, wide=False),
    "wide_defender": lambda name, key, bb: PunisherBot(name, key, bb, wide=True),
    "yosief": lambda name, key, bb: YosiefBot(name, key, bb),
    "range_gto": lambda name, key, bb: RangeGTOBot(name, key, bb),
}
