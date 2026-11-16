"""vals_tools.ladder: the league weights, slot rotation, the delete guard and the watchdog's rules."""

from datetime import datetime, timedelta

import pytest

from vals_tools import ladder as ld

BASELINE = {"failed": 0.02, "vpip": 0.22, "allin": 0.02}


def copy(cid: str, n: int, kind: str = "ckpt", passed: bool | None = None, diff: float = 0.0, bad: bool = False) -> dict:
    out = {"id": cid, "n": n, "kind": kind, "bad": bad}
    if passed is not None:
        out["yardstick"] = {"passed": passed, "diff": diff}
    return out


def test_league_weights_sum_to_100_and_favor_the_newest_checkpoint() -> None:
    none = ld.league_entries([], "p/base", "p/base")
    assert sum(e["weight"] for e in none) == pytest.approx(100, abs=0.1)
    assert sum(e["weight"] for e in none if "bot" in e) == pytest.approx(ld.BOT_SHARE, abs=0.1)
    assert not any(e["id"] == "house" for e in none)  # the shared house model is off by default
    live = [{"id": "sp-a-step20", "deployment": "p/sp-a-step20", "kind": "ckpt"},
            {"id": "sp-a-step10", "deployment": "p/sp-a-step10", "kind": "ckpt"},
            {"id": "sp-start1-step0", "deployment": "p/sp-start1-step0", "kind": "start"}]
    some = {e["id"]: e for e in ld.league_entries(live, "p/base", "p/base", "p/house", 5.0, bot_share=45.0)}
    assert sum(e["weight"] for e in some.values()) == pytest.approx(100, abs=0.1)
    assert some["sp-a-step20"]["weight"] > some["sp-a-step10"]["weight"] > 0 and some["house"]["weight"] > 0
    assert sum(e["weight"] for e in some.values() if "bot" in e) == pytest.approx(45.0, abs=0.1)
    assert {e["bot"] for e in some.values() if "bot" in e} == set(ld.BOT_STYLES)


def test_the_league_file_the_ladder_writes_is_one_the_harness_accepts() -> None:
    harness_league = pytest.importorskip("poker_harness.league")
    live = [{"id": "sp-a-step20", "deployment": "p/sp-a-step20", "kind": "ckpt"}]
    harness_league.check_league({"version": 1, "noise": ld.NOISE,
                                 "entries": ld.league_entries(live, "p/base", "p/base")})


def test_rotation_keeps_the_newest_the_second_newest_and_the_best() -> None:
    deployed = [copy("old-best", 0, passed=True, diff=0.5), copy("mid", 1, passed=True, diff=-0.2), copy("new", 2)]
    assert ld.choose_evictee(deployed, copy("newer", 3))["id"] == "mid"
    starts = [copy("s1", -1, "start"), copy("s2", -2, "start"), copy("s3", -3, "start")]
    assert ld.choose_evictee(starts, copy("c1", 0))["id"] == "s3"  # start copies make way first, oldest first
    assert ld.choose_evictee(starts[:2] + [copy("bad", 0, bad=True)], copy("c2", 1))["id"] == "bad"
    assert ld.choose_evictee(deployed, copy("s4", -4, "start")) is None  # a start copy never displaces a checkpoint


def test_only_our_own_copies_may_be_deleted() -> None:
    ours = {"train_4f13366072d4"}
    assert ld.may_delete("sp-6072d4-step20", ours) and ld.may_delete("sp-start2-step0", ours)
    for name in ("base", "base-h1", "opsd-pb-6072d4-step9", "sp-abcdef-step10", "sp-6072d4-step20-x", "entry"):
        assert not ld.may_delete(name, ours), name


def test_gpu_budget() -> None:
    assert ld.free_gpus(1, True) == 3 and ld.free_gpus(4, True) == 0 and ld.free_gpus(9, False) == 1
    assert ld.free_gpus(8, True) < 0  # e.g. the base-h copies still up: the ladder waits


def test_the_yardstick_judges_only_a_clear_loss() -> None:
    assert ld.yardstick_passed({"diff": -0.4, "ci95": 0.6})
    assert ld.yardstick_passed({"diff": -1.3, "ci95": 1.5})  # the interval reaches 0: not clear
    assert not ld.yardstick_passed({"diff": -1.3, "ci95": 0.8})
    assert ld.yardstick_passed(None) and ld.yardstick_passed({"diff": -2.0, "ci95": float("nan")}) is False


def test_watchdog_rules() -> None:
    calm = {"decisions": 400, "failed": 0.02, "vpip": 0.25, "allin": 0.02, "opponent_decisions": 600,
            "opponent_failed": 0.03, "tables": 100, "rollout_failed_share": 0.0}
    assert ld.tripped(calm, BASELINE) == []
    assert ld.tripped(calm | {"vpip": 0.05}, BASELINE) == ["vpip"]
    assert ld.tripped(calm | {"vpip": 0.6, "allin": 0.2}, BASELINE) == ["vpip", "allin"]
    assert ld.tripped(calm | {"failed": 0.08}, BASELINE) == ["failed"]
    assert ld.tripped(calm | {"opponent_failed": 0.4}, BASELINE) == ["opponent_failures"]
    assert ld.tripped(calm | {"decisions": 20, "vpip": 0.0}, BASELINE) == []  # too few hands to judge the seat
    assert ld.tripped(calm | {"rollout_failed_share": 0.3}, BASELINE) == ["rollout_failures"]


def test_window_stats_read_policy_and_opponent_seats() -> None:
    def act(player: str, model: str, kind: str, call: int, stack_after: int = 150, **flags) -> dict:
        return {"player": player, "street": "preflop", "action": {"type": kind}, "call_amount": call,
                "stack_after": stack_after, "model_response": {"model_name": model, **flags}}

    log = {"hands": [{"actions": [act("seat1", "opponent/p/base", "raise", 2),
                                   act("seat2", "opponent/p/x", "fold", 6, afk=True),
                                   act("seat3", "bot/nit", "fold", 6),
                                   act("seat0", "ac2/policy", "raise", 6, stack_after=0)]}]}
    stats = ld.window_stats([log], failures=1)
    assert stats["allin"] == 1.0 and stats["opponent_decisions"] == 2 and stats["opponent_failed"] == 0.5
    assert stats["tables"] == 2 and stats["rollout_failed_share"] == 0.5 and stats["vpip"] == 1.0


def test_final_candidates_and_deadlines() -> None:
    cps = {c["id"]: c | {"check_passed": True} for c in [
        copy("a", 0, passed=True, diff=0.3), copy("b", 1, passed=True, diff=0.9), copy("c", 2, passed=False, diff=-2),
        copy("d", 3, passed=True, diff=0.1), copy("e", 4, passed=True, diff=-0.1), copy("f", 5, bad=True),
        copy("s", -1, "start")]}
    assert ld.final_candidates(cps) == ["b", "a", "d", "e"]
    now = datetime(2026, 11, 14, 22, 30).astimezone()
    assert ld.deadline("08:00", now) == now.replace(hour=8, minute=0) + timedelta(days=1)
    assert ld.deadline("23:15", now) == now.replace(hour=23, minute=15)
