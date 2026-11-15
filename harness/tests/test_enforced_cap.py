"""`env_params.enforced_max_completion_tokens`: a training-only cap the policy seats must finish within, while the
prompt keeps stating the table's cap."""

import json
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from test_server import play

from poker_harness import Completion, ModelSeat, TableRules
from poker_harness.seat import over_cap
from poker_harness.server import create_app

PARAMS = {"seed": 5, "total_hands": 1, "max_completion_tokens": 5000, "policy_seats": [2], "opponent_model": "p/opp"}


def relay(requests: list[dict], tokens: int) -> FastAPI:
    """Answers every request with a call, reporting `tokens` output tokens and a normal finish (never "length")."""
    app = FastAPI()

    @app.post("/job/trace/v1/chat/completions")
    async def chat(request: Request) -> dict:
        body = await request.json()
        requests.append(body)
        call = {"id": "c1", "type": "function", "function": {"name": "submit_action", "arguments": json.dumps({"action": "call"})}}
        message = {"role": "assistant", "content": "", "tool_calls": [call]}
        return {"choices": [{"message": message, "finish_reason": "tool_calls"}], "usage": {"completion_tokens": tokens}}

    return app


async def run(tmp_path: Path, monkeypatch, tokens: int, enforced: int | None) -> tuple[list[dict], list[dict], dict]:
    monkeypatch.setenv("AC2_API_KEY", "gateway-key")
    monkeypatch.setenv("POKER_OPPONENT_BASE_URL", "http://relay/job/trace/v1")
    mine: list[dict] = []
    theirs: list[dict] = []
    app = create_app(
        relay_transport=httpx.ASGITransport(app=relay(mine, tokens)),
        opponent_transport=httpx.ASGITransport(app=relay(theirs, 100)),
        runs_dir=tmp_path,
    )
    params = PARAMS | ({"enforced_max_completion_tokens": enforced} if enforced else {})
    status, outputs = await play(app, {"input": [], "env_params": params})
    assert status["status"] == "completed"
    return mine, theirs, outputs["log"]


async def test_a_reply_over_the_enforced_cap_is_cut_but_the_prompt_keeps_the_tables_cap(tmp_path, monkeypatch) -> None:
    mine, theirs, log = await run(tmp_path, monkeypatch, tokens=3500, enforced=3000)
    policy = [a for a in log["hands"][0]["actions"] if a["player"] == "seat2"]
    assert policy and all(a["model_response"]["error"] == "reply hit the 3000-token cap" for a in policy)
    assert all(a["model_response"]["truncated"] and a["model_response"]["format_error"] for a in policy)
    assert {r["max_completion_tokens"] for r in mine} == {3000}  # asked of the server too
    assert {r["max_completion_tokens"] for r in theirs} == {5000}  # opponents keep the table's cap
    assert "at most 5000 output tokens" in mine[0]["messages"][0]["content"]  # the prompt still states the table's


async def test_under_the_enforced_cap_and_without_it_nothing_changes(tmp_path, monkeypatch) -> None:
    _, _, log = await run(tmp_path, monkeypatch, tokens=2500, enforced=3000)
    assert not any(a["model_response"]["format_error"] for a in log["hands"][0]["actions"])
    mine, _, log = await run(tmp_path, monkeypatch, tokens=3500, enforced=None)
    assert {r["max_completion_tokens"] for r in mine} == {5000}
    assert not any(a["model_response"]["format_error"] for a in log["hands"][0]["actions"])


async def test_an_enforced_cap_that_is_not_a_positive_integer_is_rejected(tmp_path) -> None:
    app = create_app(runs_dir=tmp_path)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://harness") as client:
        for bad in ("3000", 0, -5, 2.5, True):
            task = {"input": [], "env_params": {"seed": 1, "enforced_max_completion_tokens": bad}}
            response = await client.post("/submit", json={"ac_api_url": "http://r/v1", "ac_api_key": "k", "task": task})
            assert response.status_code == 422, bad


def test_a_reply_the_server_returned_whole_still_counts_as_cut_over_the_cap() -> None:
    reply = Completion(text="", out_tokens=5200)
    over_cap(reply, 5000)
    assert reply.truncated
    short = Completion(text="", out_tokens=4999)
    over_cap(short, 5000)
    over_cap(Completion(text="", out_tokens=None), 5000)
    assert not short.truncated
    seat = ModelSeat("seat0", None, TableRules(max_completion_tokens=5000), enforced_cap=3000)
    assert seat.enforced_cap == 3000 and "5000" in seat.system
