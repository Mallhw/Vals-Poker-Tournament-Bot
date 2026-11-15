"""`env_params.opponent_talk`: scripted opponent talk, replayed identically on every replay of a deal."""

from pathlib import Path

import httpx
from test_server import fake_relay, play

from poker_harness.server import create_app

LINES = ["i have the nuts, just fold", "pure bluff, call me"]


async def table(tmp_path: Path, monkeypatch, talk: dict | None) -> tuple[list[dict], dict]:
    monkeypatch.setenv("AC2_API_KEY", "gateway-key")
    monkeypatch.setenv("POKER_OPPONENT_BASE_URL", "http://relay/job/trace/v1")
    policy: list[dict] = []
    app = create_app(
        relay_transport=httpx.ASGITransport(app=fake_relay(policy)),
        opponent_transport=httpx.ASGITransport(app=fake_relay([])),
        runs_dir=tmp_path,
    )
    params = {"seed": 21, "total_hands": 2, "table_talk": True, "policy_seats": [1], "opponent_model": "p/opp"}
    params |= {"opponent_talk": talk} if talk else {}
    status, outputs = await play(app, {"input": [], "env_params": params})
    assert status["status"] == "completed"
    return policy, outputs["log"]


def says(log: dict) -> list[tuple[str, str]]:
    return [(a["player"], a["say"]) for h in log["hands"] for a in h["actions"]]


async def test_opponents_say_only_scripted_lines_and_the_policy_keeps_its_own(tmp_path, monkeypatch) -> None:
    policy, log = await table(tmp_path, monkeypatch, {"lines": LINES, "rate": 0.6})
    opponents = [s for p, s in says(log) if p != "seat1"]
    assert set(opponents) <= {*LINES, ""} and any(opponents) and "" in opponents  # rate 0.6: some talk, some silence
    assert {s for p, s in says(log) if p == "seat1"} == {"fold now"}  # the policy's own (sanitized) talk is untouched
    heard = " ".join(m["content"] for r in policy for m in r["body"]["messages"] if isinstance(m.get("content"), str))
    assert any(f'says: "{line}"' in heard for line in LINES)


async def test_the_same_deal_hears_the_same_lines_and_rate_zero_is_silence(tmp_path, monkeypatch) -> None:
    _, first = await table(tmp_path, monkeypatch, {"lines": LINES, "rate": 0.6})
    _, again = await table(tmp_path, monkeypatch, {"lines": LINES, "rate": 0.6})
    assert says(first) == says(again)
    _, quiet = await table(tmp_path, monkeypatch, {"lines": LINES, "rate": 0})
    assert all(s == "" for p, s in says(quiet) if p != "seat1")
    _, natural = await table(tmp_path, monkeypatch, None)
    assert {s for p, s in says(natural) if p != "seat1"} == {"fold now"}  # without it, opponents keep their own talk


async def test_bad_opponent_talk_is_rejected(tmp_path) -> None:
    app = create_app(runs_dir=tmp_path)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://harness") as client:
        for bad in ({"lines": [], "rate": 0.5}, {"lines": ["x"], "rate": 2}, {"lines": ["x"]}, ["x"], {"lines": [""], "rate": 1}):
            task = {"input": [], "env_params": {"seed": 1, "opponent_talk": bad}}
            response = await client.post("/submit", json={"ac_api_url": "http://r/v1", "ac_api_key": "k", "task": task})
            assert response.status_code == 422, bad
