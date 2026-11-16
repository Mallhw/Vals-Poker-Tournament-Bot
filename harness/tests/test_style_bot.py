"""Style bots: a correct fast evaluator, sensible equity, legal moves, and styles that really differ."""

import asyncio
import random

from pokerkit import StandardHighHand

from poker_harness.completions import table_rules
from poker_harness.engine.adapted_bots import ADAPTED, bot_styles, hand_class, hand_range, in_top, make_bot, percentiles
from poker_harness.engine.style_bot import DECK, StyleBot, equity_vs_random, hand_value
from poker_harness.game import play_session


def test_the_fast_evaluator_orders_hands_like_pokerkit() -> None:
    rng = random.Random(3)
    for _ in range(400):
        cards = rng.sample(DECK, 9)
        board, a, b = cards[:5], cards[5:7], cards[7:9]
        ours = (hand_value(a + board) > hand_value(b + board)) - (hand_value(a + board) < hand_value(b + board))
        pa, pb = StandardHighHand.from_game("".join(a), "".join(board)), StandardHighHand.from_game("".join(b), "".join(board))
        theirs = (pa > pb) - (pa < pb)
        assert ours == theirs, (a, b, board)


def test_equity_against_random_hands_is_about_right() -> None:
    rng = random.Random(1)
    assert 0.78 < equity_vs_random(["As", "Ah"], [], 1, rng, samples=600) < 0.92
    assert 0.25 < equity_vs_random(["7c", "2d"], [], 1, rng, samples=600) < 0.42
    assert equity_vs_random(["As", "Ah"], [], 4, rng, samples=600) < 0.7  # more opponents, less equity
    assert equity_vs_random(["As", "Ks"], ["Qs", "Js", "Ts"], 3, rng) == 1.0  # a royal flush


def bot_tables(styles: list[str], games: int, hands: int = 10) -> list[dict]:
    rules = table_rules({"seats": len(styles), "total_hands": hands, "reset_stacks": True})

    async def one(seed: int) -> dict:
        bots = [make_bot(style, f"seat{i}", f"{seed}:{i}") for i, style in enumerate(styles)]
        return await play_session(f"bots-{seed}", seed, bots, rules)

    async def every() -> list[dict]:
        return list(await asyncio.gather(*(one(seed) for seed in range(games))))

    return asyncio.run(every())


def test_bots_only_make_legal_moves() -> None:
    styles = sorted(bot_styles())
    for start in range(0, len(styles), 6):  # every style, six to a table
        for log in bot_tables(styles[start:start + 6], games=8):
            for hand in log["hands"]:
                for action in hand["actions"]:
                    assert not action["model_response"].get("fallback_used"), action


def test_ranges_classes_and_percentiles() -> None:
    assert hand_range("55+") == {"AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55"}
    assert hand_range("ATs+") == {"AKs", "AQs", "AJs", "ATs"}
    assert hand_range("A5s-A3s, KQ") == {"A5s", "A4s", "A3s", "KQs", "KQo"}
    assert hand_range("98s-65s") == {"98s", "87s", "76s", "65s"}
    assert hand_range("TT-88") == {"TT", "99", "88"}
    assert hand_class(["Kd", "As"]) == "AKo" and hand_class(["7h", "7c"]) == "77" and hand_class(["9s", "Ts"]) == "T9s"
    assert len(percentiles()) == 169 and percentiles()["AA"] == 0.0
    assert in_top("AKs", 0.05) and not in_top("72o", 0.9) and in_top("72o", 1.0)


def test_adapted_bots_keep_their_sources_shape() -> None:
    styles = ["tag_punisher", "wide_defender", "hybrid_tag", "hybrid_lag", "yosief", "range_gto"]
    logs = bot_tables(styles, games=40)
    vpip = {}
    for i, style in enumerate(styles):
        hands = [[a["action"]["type"] for a in h["actions"] if a["player"] == f"seat{i}" and a["street"] == "preflop"]
                 for log in logs for h in log["hands"]]
        vpip[style] = sum(any(m in ("call", "raise", "bet") for m in moves) for moves in hands) / len(hands)
    assert vpip["tag_punisher"] < vpip["wide_defender"] and vpip["hybrid_tag"] < vpip["hybrid_lag"], vpip
    assert all(0.05 < v < 0.6 for v in vpip.values()), vpip
    assert set(ADAPTED) <= bot_styles()


def test_the_styles_really_differ() -> None:
    styles = ["nit", "tag", "lag", "maniac", "station", "loose_passive"]
    logs = bot_tables(styles, games=40)
    vpip, pfr = {}, {}
    for i, style in enumerate(styles):
        seat = f"seat{i}"
        hands = [[a["action"]["type"] for a in h["actions"] if a["player"] == seat and a["street"] == "preflop"]
                 for log in logs for h in log["hands"]]
        vpip[style] = sum(any(m in ("call", "raise", "bet") for m in moves) for moves in hands) / len(hands)
        pfr[style] = sum("raise" in moves for moves in hands) / len(hands)
    assert vpip["nit"] < vpip["tag"] < vpip["lag"] < vpip["maniac"], vpip
    assert pfr["station"] < pfr["tag"] and vpip["station"] > vpip["tag"], (vpip, pfr)
    assert vpip["loose_passive"] > vpip["tag"] and pfr["loose_passive"] < pfr["tag"], (vpip, pfr)


def test_a_bot_repeats_itself_on_the_same_deal_and_varies_across_tables() -> None:
    first, again = bot_tables(["tag"] * 6, games=2), bot_tables(["tag"] * 6, games=2)
    moves = lambda logs: [a["action"] for log in logs for h in log["hands"] for a in h["actions"]]  # noqa: E731
    assert moves(first) == moves(again)
    assert StyleBot("s", "tag", "1:0").describe() != StyleBot("s", "tag", "2:0").describe()
