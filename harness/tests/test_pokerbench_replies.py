"""pokerbench_replies.py: a deployment's replies to converted spots, sent as the harness sends them and judged."""

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pokerbench_convert as pc
import pokerbench_replies as pr
from poker_harness.seat import Completion

# First to act under the gun with AsKs: the solver opens to 2 BB, a raise to 4 chips at the event's 1/2.
OPEN = {"prev_line": "", "hero_pos": "UTG", "hero_holding": "AsKs", "correct_decision": "2.0bb", "pot_size": "1.5"}


SPOT = asyncio.run(pc.convert_row(OPEN, {"file": "test.csv", "row": 7}))


@pytest.fixture
def spot() -> dict:
    assert SPOT["label"] == {"action": "raise", "amount": 4}
    return json.loads(json.dumps(SPOT))


def call(arguments: dict, name: str = "submit_action", call_id: str = "c1") -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


@pytest.mark.parametrize(
    ("calls", "truncated", "clean", "agrees", "played"),
    [
        ([call({"action": "raise", "amount": 4})], False, True, True, {"action": "raise", "amount": 4}),
        ([call({"action": "raise", "amount": 5, "say": "gl"})], False, True, True, {"action": "raise", "amount": 5}),
        ([call({"action": "raise", "amount": 12})], False, True, False, {"action": "raise", "amount": 12}),
        ([call({"action": "fold"})], False, True, False, {"action": "fold"}),
        ([call({"action": "check"})], False, False, False, None),  # facing the big blind: illegal under strict rules
        ([call({"action": "raise", "amount": 4})], True, False, False, None),  # cut at the cap never counts
        ([], False, False, False, None),
        ([call({"action": "call"}), call({"action": "fold"}, call_id="c2")], False, False, False, None),
        ([call({"contents": "x"}, name="update_scratchpad")], False, False, False, None),
    ],
)
def test_replies_are_judged_by_the_event_rules_and_the_solver(spot, calls, truncated, clean, agrees, played) -> None:
    verdict = pr.judge(spot, Completion(text="", tool_calls=calls, truncated=truncated))
    assert (verdict["clean"], verdict["agrees"], verdict["decision"]) == (clean, agrees, played)
    assert (verdict["error"] is None) == clean


def gateway(seen: list[dict]) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> dict:
        body = await request.json()
        seen.append(body)
        message = {"content": "", "reasoning_content": "AK suited opens.", "tool_calls": [call({"action": "raise", "amount": 4})]}
        usage = {"completion_tokens": 30, "completion_tokens_details": {"reasoning_tokens": 25}}
        return {"choices": [{"finish_reason": "tool_calls", "message": message}], "usage": usage}

    return app


async def test_replies_are_asked_as_the_harness_asks_and_reruns_resume(spot, tmp_path, monkeypatch) -> None:
    seen: list[dict] = []
    real = httpx.AsyncClient

    class Client(real):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.ASGITransport(app=gateway(seen))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(pr.httpx, "AsyncClient", Client)
    out = tmp_path / "replies.jsonl"
    stats = await pr.collect([spot], out, "p/base", "k", samples=2, base_url="http://gw/v1")
    assert stats == {"skipped": 0, "written": 2, "clean": 2, "agrees": 2, "failed": 0}
    body = seen[0]
    assert body["model"] == "p/base" and body["messages"] == spot["messages"]
    assert body["max_completion_tokens"] == 5000 and body["chat_template_kwargs"] == {"enable_thinking": True}
    assert [t["function"]["name"] for t in body["tools"]] == ["submit_action", "update_scratchpad"]
    assert [{"type": "function", **t["function"]} for t in body["tools"]] == spot["tools"]  # what the converter saw
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["sample"] for r in rows] == [0, 1] and rows[0]["reply"]["reasoning"] == "AK suited opens."
    again = await pr.collect([spot], out, "p/base", "k", samples=2, base_url="http://gw/v1")
    assert again["skipped"] == 2 and again["written"] == 0 and len(seen) == 2


def test_spots_are_sampled_per_file_and_street(tmp_path) -> None:
    rows = [{"street": s, "n": i} for i, s in enumerate(["preflop", "flop", "turn", "river"] * 5)]
    path = tmp_path / "spots.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert len(pr.sample_records([path], 3, 0, None)) == 3
    assert {r["street"] for r in pr.sample_records([path], None, 0, {"flop", "turn"})} == {"flop", "turn"}
    assert pr.sample_records([path], 4, 1, None) == pr.sample_records([path], 4, 1, None)


def test_a_hint_is_appended_to_the_system_message_as_opsds_teacher_reads_it(spot) -> None:
    template = pr.opsd_hint_template()
    assert template.startswith("Reference play for this exact spot") and "{solver_text}" in template
    hinted = pr.with_hint(spot, template)
    system = hinted["messages"][0]["content"]
    assert system == spot["messages"][0]["content"] + "\n\n" + template.format(solver_text="raise to 4")
    assert hinted["messages"][1:] == spot["messages"][1:] and hinted["hint"] in system
    assert "raise to 4" not in spot["messages"][0]["content"]  # the original record is untouched
