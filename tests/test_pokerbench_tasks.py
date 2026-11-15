"""opsd/pokerbench_tasks.py: converted PokerBench spots as hinted (one-step OPSD) or SFT tasks."""

import json
import sys
from pathlib import Path

import pytest
from ac2.runtime import FunctionCall, FunctionCallOutput, HintedTask, Message, Task

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "opsd")]

import pokerbench_tasks as pb  # noqa: E402

TOOLS = json.loads((ROOT / "opsd" / "tools.json").read_text())
CALL = {"id": "call_1", "type": "function", "function": {"name": "submit_action", "arguments": '{"action": "raise", "amount": 5}'}}
# A converted postflop spot, shortened: the hero opened preflop and now acts on the flop.
RECORD = {
    "messages": [
        {"role": "system", "content": "You are a professional poker player..."},
        {"role": "user", "content": "## Your scratchpad (private: only you can see it)\n(empty)"},
        {"role": "user", "content": "## Hand 1 of 1\nLegal actions: fold, call 2, raise to 4-200"},
        {"role": "assistant", "content": "", "tool_calls": [CALL]},
        {"role": "tool", "tool_call_id": "call_1", "content": "Applied: raise to 5."},
        {"role": "user", "content": "Since your last action:\n...\nLegal actions: check, bet to 2-195"},
    ],
    "tools": TOOLS,
    "label": {"action": "bet", "amount": 8},
    "solver_text": "bet to 8",
    "source": {"file": "postflop.csv", "row": 10},
    "street": "flop",
    "hero_position": "CO",
}


def test_chat_messages_become_ac2_items() -> None:
    items = pb.to_items(RECORD["messages"])
    assert [type(i) for i in items] == [Message, Message, Message, FunctionCall, FunctionCallOutput, Message]
    assert items[3].call_id == items[4].call_id == "call_1" and items[3].arguments == CALL["function"]["arguments"]


def test_a_hinted_task_appends_the_solver_play_to_the_system_message_for_the_teacher() -> None:
    task = pb.hinted_task(RECORD)
    assert HintedTask.model_validate(task.model_dump(mode="json")) == task  # survives the registry's round trip
    (selection,) = task.opsd_hint_selections
    assert selection.rollout_prefix_len == len(task.input) and isinstance(task.input[-1], Message)
    assert selection.mutation.item_boundary == 1 and selection.mutation.action == "append_to_previous_item"
    assert task.input[0].role == "system" and "bet to 8" in selection.hint
    assert task.env_params["tools"] == TOOLS and task.env_params["pokerbench"]["label"] == RECORD["label"]


def test_an_sft_task_ends_with_the_solver_play_as_a_submit_action_call() -> None:
    task = pb.sft_task(RECORD)
    assert Task.model_validate(task.model_dump(mode="json")) == task
    last = task.input[-1]
    assert isinstance(last, FunctionCall) and last.name == "submit_action"
    assert json.loads(last.arguments) == {"action": "bet", "amount": 8}
    assert task.input[:-1] == pb.to_items(RECORD["messages"])


def test_tools_that_differ_from_the_recorded_ones_are_refused() -> None:
    with pytest.raises(SystemExit, match="differ from opsd/tools.json"):
        pb.hinted_task(RECORD | {"tools": TOOLS[:1]})
    with pytest.raises(ValueError):
        pb.build([RECORD], "dpo")


THINKING = "Pot odds and a dry board: a small bet takes it down often."
REPLY = RECORD | {
    "model": "vals-hackathon/base",
    "sample": 0,
    "reply": {
        "reasoning": THINKING,
        "text": "",
        "tool_calls": [{"id": "r1", "type": "function", "function": {"name": "submit_action", "arguments": '{"action": "check"}'}}],
        "out_tokens": 40,
        "reasoning_tokens": 30,
        "truncated": False,
    },
    "clean": True,
    "error": None,
    "decision": {"action": "check"},
    "agrees": False,
}


def test_an_offline_task_replays_the_reply_with_the_solver_play_for_the_teacher() -> None:
    from ac2.runtime import Reasoning

    task = pb.offline_task(REPLY)
    assert HintedTask.model_validate(task.model_dump(mode="json")) == task
    prompt = pb.to_items(RECORD["messages"])
    (selection,) = task.opsd_hint_selections
    assert task.input[: len(prompt)] == prompt and selection.rollout_prefix_len == len(prompt)
    reasoning, call = task.input[len(prompt) :]
    assert isinstance(reasoning, Reasoning) and reasoning.content == THINKING
    assert isinstance(call, FunctionCall) and json.loads(call.arguments) == {"action": "check"}
    assert "bet to 8" in selection.hint and "bet to 8" not in json.dumps(task.model_dump(mode="json")["input"])
    assert task.env_params["reply"] == {"model": "vals-hackathon/base", "sample": 0, "decision": {"action": "check"}, "agrees": False}


def test_reply_modes_keep_clean_replies_and_rft_only_those_that_agree() -> None:
    agreeing = REPLY | {"agrees": True, "decision": {"action": "bet", "amount": 8}}
    unclean = REPLY | {"clean": False, "error": "reply hit the token cap", "decision": None}
    replies = [REPLY, agreeing, unclean]
    assert len(pb.build(replies, "offline")) == 2
    (rft,) = pb.build(replies, "rft")
    assert Task.model_validate(rft.model_dump(mode="json")) == rft and rft.env_params["reply"]["agrees"]
    with pytest.raises(SystemExit, match="needs replies"):
        pb.build([RECORD], "offline")


def test_max_tokens_drops_longer_replies_in_the_reply_modes() -> None:
    long = REPLY | {"agrees": True, "reply": REPLY["reply"] | {"out_tokens": 2500}}
    short = REPLY | {"agrees": True, "reply": REPLY["reply"] | {"out_tokens": 1800}}
    assert len(pb.build([long, short], "rft")) == 2
    (kept,) = pb.build([long, short], "rft", max_tokens=2000)
    assert kept.env_params["reply"]["agrees"] and len(pb.build([long, short], "offline", max_tokens=2000)) == 1
