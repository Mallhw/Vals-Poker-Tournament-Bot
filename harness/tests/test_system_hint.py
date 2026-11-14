"""`env_params.system_hint`: a hint under test reaches only the policy seats' system prompt, as OPSD's teacher sees it."""

import hashlib
import json
import runpy
import sys
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Request
from test_server import fake_relay, play

from poker_harness import ModelSeat, TableRules, system_prompt
from poker_harness.grading import grade_log
from poker_harness.prompts import HINT_SEPARATOR
from poker_harness.server import create_app

HARNESS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS))
import play_local

HINT = "Strategy notes: raise more from the button."


def system_messages(requests: list[dict]) -> set[str]:
    return {r["body"]["messages"][0]["content"] for r in requests}


def test_the_hint_follows_the_system_prompt_with_the_opsd_separator() -> None:
    rules = TableRules()
    assert ModelSeat("seat0", None, rules).system == system_prompt(rules)
    assert ModelSeat("seat0", None, rules, system_hint=HINT).system == system_prompt(rules) + HINT_SEPARATOR + HINT
    # OPSD's teacher gets the hint appended to the system message with opsd/hint.py's SEPARATOR: they must agree.
    assert runpy.run_path(str(HARNESS.parent / "opsd" / "hint.py"))["SEPARATOR"] == HINT_SEPARATOR


async def test_only_policy_seats_see_the_hint_and_the_score_names_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AC2_API_KEY", "gateway-key")
    monkeypatch.setenv("POKER_OPPONENT_BASE_URL", "http://relay/job/trace/v1")
    relay_requests: list[dict] = []
    opponent_requests: list[dict] = []
    app = create_app(
        relay_transport=httpx.ASGITransport(app=fake_relay(relay_requests)),
        opponent_transport=httpx.ASGITransport(app=fake_relay(opponent_requests)),
        runs_dir=tmp_path,
    )
    params = {"seed": 99, "total_hands": 1, "policy_seats": [3], "opponent_model": "p/opp", "system_hint": HINT}
    status, outputs = await play(app, {"input": [], "env_params": params})
    log = outputs["log"]

    assert status["status"] == "completed"
    assert system_messages(relay_requests) == {system_prompt(TableRules(total_hands=1, reset_stacks=True)) + "\n\n" + HINT}
    assert opponent_requests and not any(HINT in m for m in system_messages(opponent_requests))
    assert not any(HINT in json.dumps(r["body"]["messages"][1:]) for r in relay_requests + opponent_requests)
    assert log["system_hint"] == HINT
    artifacts = grade_log(log, ["seat3"])["artifacts"]
    assert artifacts["deal"] == "99:seat3"
    assert artifacts["hint"] == hashlib.sha256(HINT.encode()).hexdigest()[:12]


async def test_without_a_hint_nothing_changes(tmp_path: Path) -> None:
    requests: list[dict] = []
    app = create_app(relay_transport=httpx.ASGITransport(app=fake_relay(requests)), runs_dir=tmp_path)
    status, outputs = await play(app, {"input": [], "env_params": {"seed": 7, "total_hands": 1}})
    assert status["status"] == "completed" and "system_hint" not in outputs["log"]
    assert system_messages(requests) == {system_prompt(TableRules(total_hands=1, reset_stacks=True))}
    artifacts = grade_log(outputs["log"], ["seat0"])["artifacts"]
    assert artifacts["deal"] == "7:seat0" and "hint" not in artifacts


async def test_a_hint_that_is_not_text_is_rejected(tmp_path: Path) -> None:
    app = create_app(runs_dir=tmp_path)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://harness") as client:
        body = {"ac_api_url": "http://relay/v1", "ac_api_key": "k", "task": {"input": [], "env_params": {"seed": 1, "system_hint": 3}}}
        response = await client.post("/submit", json=body)
    assert response.status_code == 422 and "system_hint" in response.text


def recording_gateway(systems: list[tuple[str, str]]) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> dict:
        body = await request.json()
        systems.append((body["model"], body["messages"][0]["content"]))
        last = str(body["messages"][-1].get("content", ""))
        name, arguments = ("update_scratchpad", {"contents": "n"}) if "is over" in last else ("submit_action", {"action": "call"})
        call = {"id": "c", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
        return {"choices": [{"finish_reason": "tool_calls", "message": {"content": "", "tool_calls": [call]}}], "usage": {}}

    return app


@pytest.mark.parametrize("suffix", [".txt", ".py"])
def test_play_local_compares_the_same_seed_with_and_without_the_hint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture, suffix: str
) -> None:
    hint_file = tmp_path / f"hint{suffix}"
    hint_file.write_text(f"HINT = {HINT!r}\n" if suffix == ".py" else HINT + "\n", encoding="utf-8")
    systems: list[tuple[str, str]] = []
    real = httpx.AsyncClient

    class Client(real):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.ASGITransport(app=recording_gateway(systems))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(play_local.httpx, "AsyncClient", Client)
    monkeypatch.setattr(play_local, "GATEWAY_URL", "http://gw/v1")
    monkeypatch.setenv("AC2_API_KEY", "k")
    argv = ["play_local.py", "--model", "p/entry", "--hands", "2", "--seed", "5", "--out", str(tmp_path / "runs"),
            "--opponents", "bots", "--hint-file", str(hint_file), "--compare"]
    monkeypatch.setattr(sys, "argv", argv)
    play_local.main()

    mine = [content for model, content in systems if model == "p/entry"]
    plain, hinted = [c for c in mine if not c.endswith(HINT)], [c for c in mine if c.endswith(HINT)]
    assert plain and hinted and all(c == plain[0] for c in plain) and hinted[0] == plain[0] + HINT_SEPARATOR + HINT
    assert sorted(p.name for p in (tmp_path / "runs").glob("*.json")) == ["local-5-hint.json", "local-5.json"]
    out = capsys.readouterr().out
    assert "=== without the hint" in out and "=== with the hint" in out and "hint - no hint:" in out
    assert out.count(", 0 failed") == 2 and "VPIP" in out


def test_compare_needs_a_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["play_local.py", "--model", "p/entry", "--compare"])
    with pytest.raises(SystemExit):
        play_local.main()
