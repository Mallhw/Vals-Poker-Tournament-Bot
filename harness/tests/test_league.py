"""Self-play leagues (`league.py`): per-seat opponents drawn from a league file, pinned per task, logged per table."""

import json
from pathlib import Path

import httpx
import pytest
from test_server import fake_relay, play

from poker_harness.league import League, LeagueError, check_league, draw
from poker_harness.personas import PERSONAS
from poker_harness.server import create_app

BOTS = [{"id": "nit", "bot": "nit", "weight": 1}, {"id": "lag", "bot": "lag", "weight": 1},
        {"id": "station", "bot": "station", "weight": 1}]


def league_file(path: Path, entries: list[dict], version: int = 1, noise: dict | None = None) -> Path:
    path.write_text(json.dumps({"version": version, "entries": entries, **({"noise": noise} if noise else {})}))
    return path


def test_bad_leagues_are_refused() -> None:
    for bad in (
        {"entries": BOTS},  # no version
        {"version": 1, "entries": [{"id": "a", "bot": "nit"}]},  # no weight
        {"version": 1, "entries": [{"id": "a", "bot": "nope", "weight": 1}]},
        {"version": 1, "entries": [{"id": "a", "model": "p/m", "bot": "nit", "weight": 1}]},
        {"version": 1, "entries": [{"id": "a", "bot": "nit", "persona": "trappy", "weight": 1}]},
        {"version": 1, "entries": [{"id": "a", "bot": "nit", "weight": 0}]},  # nothing to draw
        {"version": 1, "entries": [{"id": "a", "bot": "nit", "weight": 1}, {"id": "a", "bot": "lag", "weight": 1}]},
        {"version": 1, "entries": BOTS, "noise": {"size_jitter": 1.5}},
    ):
        with pytest.raises(LeagueError):
            check_league(bad)
    assert check_league({"version": 2, "entries": BOTS, "noise": {"action_epsilon": 0.03}})


def test_draws_follow_the_weights_skip_retired_entries_and_repeat() -> None:
    entries = [{"id": "a", "bot": "nit", "weight": 3}, {"id": "b", "bot": "lag", "weight": 1},
               {"id": "gone", "bot": "maniac", "weight": 0}]
    picks = [spec["id"] for seed in range(2000) for spec in draw(entries, seed, [0], 6, 1).values()]
    assert "gone" not in picks and 0.7 < picks.count("a") / len(picks) < 0.8
    assert draw(entries, 5, [2], 6, 1) == draw(entries, 5, [2], 6, 1)
    assert set(draw(entries, 5, [2], 6, 1)) == {0, 1, 3, 4, 5}


def test_a_tasks_lineup_is_pinned_across_league_changes(tmp_path: Path) -> None:
    path = league_file(tmp_path / "league.json", [{"id": "nit", "bot": "nit", "weight": 1}])
    league = League(path, tmp_path / "pins")
    first = league.lineup(11, [0], 6)
    league_file(path, [{"id": "lag", "bot": "lag", "weight": 1}], version=2)
    assert League(path, tmp_path / "pins").lineup(11, [0], 6) == first  # another shard reads the same pin
    assert {s["id"] for s in league.lineup(12, [0], 6)["seats"].values()} == {"lag"}  # a new task sees version 2
    expired = League(path, tmp_path / "pins", pin_seconds=0)
    assert expired.lineup(11, [0], 6)["version"] == 2


def test_a_bad_edit_keeps_the_last_good_league(tmp_path: Path) -> None:
    path = league_file(tmp_path / "league.json", BOTS)
    league = League(path, tmp_path / "pins")
    assert league.current()["version"] == 1
    path.write_text('{"version": 2, "entries": [')
    import os
    os.utime(path, (1, 1))  # a different mtime than the good file's
    assert league.current()["version"] == 1
    with pytest.raises(LeagueError):
        League(path).current()


async def test_league_tables_seat_bots_and_models_and_log_the_lineup(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AC2_API_KEY", "gateway-key")
    monkeypatch.setenv("POKER_OPPONENT_BASE_URL", "http://relay/job/trace/v1")
    entries = [{"id": "copy", "model": "p/copy", "persona": "trappy", "weight": 1}, *BOTS]
    league = League(league_file(tmp_path / "league.json", entries, noise={"action_epsilon": 0.5}), tmp_path / "pins")
    opponents: list[dict] = []
    app = create_app(
        relay_transport=httpx.ASGITransport(app=fake_relay([])),
        opponent_transport=httpx.ASGITransport(app=fake_relay(opponents)),
        runs_dir=tmp_path,
        league=league,
    )
    params = {"seed": 7, "seats": 6, "total_hands": 1, "policy_seats": [3], "opponent_league": True, "ev_labels": True}
    status, outputs = await play(app, {"input": [], "env_params": params})
    assert status["status"] == "completed", status
    log = outputs["log"]
    lineup = log["lineup"]
    assert lineup["version"] == 1 and set(lineup["seats"]) == {"seat0", "seat1", "seat2", "seat4", "seat5"}
    for seat, spec in lineup["seats"].items():
        if "bot" in spec:
            assert spec["params"]["style"] == spec["bot"]
    models = {r["body"]["model"] for r in opponents}
    systems = [r["body"]["messages"][0]["content"] for r in opponents]
    if any("model" in s for s in lineup["seats"].values()):
        assert models == {"p/copy"} and all(PERSONAS["trappy"] in s for s in systems)
    names = {a["player"]: a["model_response"]["model_name"] for h in log["hands"] for a in h["actions"]}
    assert names.get("seat3", "ac2/policy") == "ac2/policy"
    noisy = [a for h in log["hands"] for a in h["actions"] if a["model_response"].get("noise")]
    assert all(a["player"] != "seat3" for a in noisy)  # the policy seat is never perturbed


async def test_explicit_opponent_seats_and_the_missing_league_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AC2_API_KEY", "gateway-key")
    monkeypatch.setenv("POKER_OPPONENT_BASE_URL", "http://relay/job/trace/v1")
    monkeypatch.delenv("POKER_LEAGUE_FILE", raising=False)
    opponents: list[dict] = []
    app = create_app(
        relay_transport=httpx.ASGITransport(app=fake_relay([])),
        opponent_transport=httpx.ASGITransport(app=fake_relay(opponents)),
        runs_dir=tmp_path,
    )
    seats = {"1": {"bot": "maniac"}, "2": {"model": "p/lag", "persona": "loose_aggressive"}}
    params = {"seed": 9, "seats": 4, "total_hands": 2, "policy_seats": [0], "opponent_model": "p/opp",
              "opponent_seats": seats}
    status, outputs = await play(app, {"input": [], "env_params": params})
    assert status["status"] == "completed", status
    assert set(outputs["log"]["lineup"]["seats"]) == {"seat1", "seat2"}
    assert {r["body"]["model"] for r in opponents} <= {"p/lag", "p/opp"}  # seat 3 keeps the task's opponent_model
    fresh = create_app(relay_transport=httpx.ASGITransport(app=fake_relay([])), runs_dir=tmp_path)  # `play` reuses one id
    status, _ = await play(fresh, {"input": [], "env_params": {"seed": 9, "policy_seats": [0], "opponent_league": True}})
    assert status["status"] == "failed" and "POKER_LEAGUE_FILE" in status["error_message"]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://harness") as client:
        for bad in ({"opponent_seats": {"x": {"bot": "nit"}}}, {"opponent_seats": {"1": {"bot": "nope"}}},
                    {"opponent_league": "yes"}, {"opponent_seats": {"1": {"bot": "nit"}}, "opponent_noise": {"x": 1}}):
            task = {"input": [], "env_params": {"seed": 1, **bad}}
            response = await client.post("/submit", json={"ac_api_url": "http://r/v1", "ac_api_key": "k", "task": task})
            assert response.status_code == 422, bad
