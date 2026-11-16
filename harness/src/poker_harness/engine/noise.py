"""Small seeded noise on an opponent's decisions, for self-play training (league tables only, never evals).

`NoisyPlayer` wraps any player (a model seat or a style bot) and, after it decides:

- with probability `action_epsilon`, swaps the action for a neighboring reasonable one: facing a bet, fold <-> call
  and call -> min-raise, raise -> call; with a free check, check <-> a small bet. It never folds when it could check.
- scales a bet or raise by a log-uniform factor in [1 - size_jitter, 1 / (1 - size_jitter)] (0.75 to 1.33 at 0.25),
  clamped to the legal range. All-ins are left alone, and so are replies that already failed (format errors).

The draws are seeded by the wrapper's key (deal, policy seats and seat) and the decision's index, so every replay of
a deal makes the same draws in the same spots. Every change is recorded in `model_response["noise"]`.
"""

from __future__ import annotations

import math
import random
from typing import Any

from .table import Decision, DecisionContext, resolve


def _raise_to(ctx: DecisionContext, fraction: float) -> int:
    """Bet or raise to `fraction` of the pot (after calling) on top of the current bet, within the legal range."""
    legal = ctx.legal
    mine = next((s.bet for s in ctx.seats if s.seat == ctx.seat), 0)
    top = mine + legal.call_amount
    target = top + round(fraction * (ctx.pot + legal.call_amount))
    return min(max(target, legal.min_raise_to), legal.max_raise_to)


def swap(decision: Decision, ctx: DecisionContext, rng: random.Random) -> Decision:
    """A neighboring reasonable action (see the module docstring)."""
    legal = ctx.legal
    kind = decision.type
    if legal.can_check:
        if kind in ("bet", "raise"):
            return Decision("check")
        if legal.raise_type:
            return Decision(legal.raise_type, _raise_to(ctx, 1 / 3))
        return decision
    if kind == "fold":
        return Decision("call")
    if kind == "call":
        if legal.raise_type and rng.random() < 0.5:
            return Decision(legal.raise_type, legal.min_raise_to)
        return Decision("fold")
    return Decision("call")  # a raise


def perturb(
    decision: Decision, ctx: DecisionContext, rng: random.Random, size_jitter: float, action_epsilon: float
) -> Decision:
    if decision.model_response.get("format_error"):
        return decision
    applied, _ = resolve(decision, ctx.legal)
    roll, coin, factor_draw = rng.random(), random.Random(rng.random()), rng.random()  # fixed draw order
    out = swap(applied, ctx, coin) if roll < action_epsilon else applied
    legal = ctx.legal
    if out.type in ("bet", "raise") and out.amount is not None and size_jitter > 0 and out.amount < legal.max_raise_to:
        mine = next((s.bet for s in ctx.seats if s.seat == ctx.seat), 0)
        top = mine + legal.call_amount
        span = -math.log(1 - size_jitter)
        factor = math.exp(-span + 2 * span * factor_draw)
        amount = top + round((out.amount - top) * factor)
        out = Decision(out.type, min(max(amount, legal.min_raise_to), legal.max_raise_to))
    if (out.type, out.amount) == (applied.type, applied.amount):
        return decision
    response = dict(decision.model_response)
    response["noise"] = {
        "from": {"type": applied.type, "amount": applied.amount},
        "to": {"type": out.type, "amount": out.amount},
    }
    return Decision(out.type, out.amount, response)


class NoisyPlayer:
    """Wraps a player; everything but `decide` (name, cleanup, scratchpad, ...) is the wrapped player's."""

    def __init__(self, inner: Any, key: str, size_jitter: float = 0.0, action_epsilon: float = 0.0) -> None:
        self.inner = inner
        self.key = key
        self.size_jitter = size_jitter
        self.action_epsilon = action_epsilon
        self._decisions = 0

    def __getattr__(self, name: str) -> Any:
        if name == "inner":  # not set yet: avoid recursing
            raise AttributeError(name)
        return getattr(self.inner, name)

    async def decide(self, ctx: DecisionContext, history: str) -> Decision:
        decision = await self.inner.decide(ctx, history)
        rng = random.Random(f"noise:{self.key}:{ctx.hand_number}:{self._decisions}")
        self._decisions += 1
        return perturb(decision, ctx, rng, self.size_jitter, self.action_epsilon)
