"""Opponent noise (`engine/noise.py`): small, legal, seeded, never a fold when a check is free, always recorded."""

import asyncio
import math
import random

from poker_harness.engine.noise import NoisyPlayer, perturb
from poker_harness.engine.table import Decision, DecisionContext, LegalActions, SeatView


def context(rng: random.Random) -> DecisionContext:
    call = rng.choice([0, 0, 2, 6, 20])
    top = 200
    raise_ok = rng.random() < 0.8
    min_to = max(4, 2 * call) if raise_ok else 0
    legal = LegalActions(can_fold=call > 0, can_check=call == 0, call_amount=call,
                         raise_type=("raise" if call else "bet") if raise_ok else None,
                         min_raise_to=min_to, max_raise_to=top if raise_ok else 0)
    seats = [SeatView(i, f"seat{i}", 200, 0, "") for i in range(6)]
    return DecisionContext(1, 1, "flop", 2, 2, 5, ["As", "Kd"], ["2c", "7h", "9s"], rng.choice([6, 20, 60]),
                           seats, [], legal)


def decision(ctx: DecisionContext, rng: random.Random) -> Decision:
    legal = ctx.legal
    options = ["check"] if legal.can_check else ["fold", "call"]
    if legal.raise_type:
        options.append(legal.raise_type)
    kind = rng.choice(options)
    amount = rng.randint(legal.min_raise_to, legal.max_raise_to) if kind in ("bet", "raise") else None
    return Decision(kind, amount, {"model_name": "opp"})


def test_noise_stays_legal_and_never_folds_for_free() -> None:
    rng = random.Random(0)
    changed = 0
    for i in range(3000):
        ctx = context(rng)
        before = decision(ctx, rng)
        after = perturb(before, ctx, random.Random(i), 0.25, 0.2)
        legal = ctx.legal
        if legal.can_check:
            assert after.type != "fold"
        if after.type in ("bet", "raise"):
            assert legal.raise_type == after.type and legal.min_raise_to <= after.amount <= legal.max_raise_to
        if after is not before:
            changed += 1
            assert after.model_response["noise"]["from"]["type"] == before.type
    assert changed > 300


def test_jitter_scales_the_raise_within_bounds_and_leaves_all_ins_alone() -> None:
    legal = LegalActions(False, True, 0, "bet", 2, 400)
    ctx = DecisionContext(1, 1, "turn", 0, 0, 5, ["As", "Kd"], ["2c", "7h", "9s", "Td"], 100,
                          [SeatView(i, f"seat{i}", 400, 0, "") for i in range(6)], [], legal)
    for i in range(500):
        out = perturb(Decision("bet", 60), ctx, random.Random(i), 0.25, 0.0)
        assert 0.75 * 60 - 1 <= out.amount <= 60 / 0.75 + 1
    assert perturb(Decision("bet", 400), ctx, random.Random(1), 0.25, 0.0).amount == 400
    failed = Decision("check", None, {"format_error": True})
    assert perturb(failed, ctx, random.Random(1), 0.25, 1.0) is failed


def test_the_same_key_makes_the_same_draws_and_attributes_pass_through() -> None:
    class Inner:
        name = "seat4"
        scratchpad = "notes"

        async def decide(self, ctx, history):
            return Decision("call", None, {})

    rng = random.Random(5)
    contexts = [context(rng) for _ in range(50)]

    def run(key: str) -> list:
        player = NoisyPlayer(Inner(), key, 0.25, 0.3)
        return [asyncio.run(player.decide(c, "")).type for c in contexts]

    assert run("7:[0]:4") == run("7:[0]:4") != run("8:[0]:4")
    wrapped = NoisyPlayer(Inner(), "k")
    assert wrapped.name == "seat4" and wrapped.scratchpad == "notes" and not hasattr(wrapped, "cleanup")
    assert math.isclose(wrapped.size_jitter, 0.0)
