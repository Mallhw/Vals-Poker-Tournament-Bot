"""rl/make_spot_dataset.py: solver-spot tasks for GRPO, with the training tables' rules."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_spot_dataset", ROOT / "rl" / "make_spot_dataset.py")
msd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(msd)

TOOLS = json.loads((ROOT / "opsd" / "tools.json").read_text())


def record(row: int, street: str = "preflop", **extra: object) -> dict:
    return {"messages": [{"role": "system", "content": "rules"}, {"role": "user", "content": "decide"}],
            "legal": {"can_fold": True, "can_check": False, "call_amount": 2}, "label": {"action": "fold"},
            "solver_text": "fold", "source": {"file": "f.csv", "row": row}, "pot": 3, "street": street,
            "hero_position": "UTG", "tools": TOOLS, **extra}


def test_spot_tasks_carry_the_spot_and_the_training_tables_rules() -> None:
    task = msd.spot_task(record(4, reply={"text": "x"}, clean=True), seed=9)
    params = task.env_params
    assert params["preset"] == "event" and params["seats"] == 6 and params["total_hands"] == 1 and params["seed"] == 9
    assert set(params["pokerbench_spot"]) == set(msd.SPOT_FIELDS)  # reply fields of a held-out record are dropped
    with pytest.raises(SystemExit, match="differ"):
        msd.spot_task(record(5, tools=TOOLS[:1]), seed=1)


def history(*lines: str, hero_raised: bool = False) -> list[dict]:
    hero = [{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "submit_action",
             "arguments": json.dumps({"action": "raise", "amount": 4})}}]}] if hero_raised else []
    return [{"role": "system", "content": "rules"}, *hero,
            {"role": "user", "content": "Actions:\n" + "\n".join(f"  preflop: {x}" for x in lines)
             + "\nLegal actions: fold, call 40, raise to 69-200"}]


def test_raises_before_the_hero_count_the_history_and_its_own_raises_not_the_legal_actions() -> None:
    assert msd.raises_before_hero({"messages": history("seat2 (seat 2) fold")}) == 0
    assert msd.raises_before_hero({"messages": history("seat4 (seat 4) raise to 5", "seat5 (seat 5) call")}) == 1
    assert msd.raises_before_hero({"messages": history("seat0 (seat 0) raise to 12", hero_raised=True)}) == 2


def test_low_raise_weighting_picks_preflop_spots_by_situation_and_carries_the_reward(tmp_path) -> None:
    low = [record(i, messages=history("seat4 (seat 4) raise to 5")) for i in range(6)]
    high = [record(50 + i, messages=history("seat4 (seat 4) raise to 5", "seat5 (seat 5) raise to 15")) for i in range(6)]
    pre, post = tmp_path / "pre.jsonl", tmp_path / "post.jsonl"
    pre.write_text("".join(json.dumps(r) + "\n" for r in low + high))
    post.write_text("".join(json.dumps(record(100 + i, "river")) + "\n" for i in range(10)))
    tasks = msd.train_tasks([pre, post], count=2, seed=3, low_raise_count=4, reward="action_first")
    spots = [t.env_params["pokerbench_spot"] for t in tasks]
    rows = [s["source"]["row"] for s in spots if s["street"] == "preflop"]
    assert len(tasks) == 8 and sum(r < 50 for r in rows) == 4 and sum(r >= 50 for r in rows) == 2
    assert all(s["reward"] == "action_first" for s in spots)
    assert "reward" not in msd.spot_task(record(1), seed=1).env_params["pokerbench_spot"]  # off unless asked


def test_per_file_counts_take_a_number_or_a_low_plus_high_raise_split(tmp_path) -> None:
    low = [record(i, messages=history("seat4 (seat 4) raise to 5")) for i in range(6)]
    high = [record(50 + i, messages=history("seat4 (seat 4) raise to 5", "seat5 (seat 5) raise to 15")) for i in range(6)]
    pre, post = tmp_path / "pre.jsonl", tmp_path / "post.jsonl"
    pre.write_text("".join(json.dumps(r) + "\n" for r in low + high))
    post.write_text("".join(json.dumps(record(100 + i, "river")) + "\n" for i in range(10)))
    tasks = msd.train_tasks([pre, post], count=99, seed=3, counts=["3+1", "5"])
    rows = [t.env_params["pokerbench_spot"]["source"]["row"] for t in tasks]
    assert len(tasks) == 9 and sum(r < 50 for r in rows) == 3 and sum(50 <= r < 100 for r in rows) == 1
    assert sum(r >= 100 for r in rows) == 5


def test_training_spots_are_sampled_per_file(tmp_path) -> None:
    pre, post = tmp_path / "pre.jsonl", tmp_path / "post.jsonl"
    pre.write_text("".join(json.dumps(record(i)) + "\n" for i in range(10)))
    post.write_text("".join(json.dumps(record(100 + i, "river")) + "\n" for i in range(10)))
    tasks = msd.train_tasks([pre, post], count=3, seed=1)
    streets = [t.env_params["pokerbench_spot"]["street"] for t in tasks]
    assert len(tasks) == 6 and streets.count("river") == 3 and len({t.env_params["seed"] for t in tasks}) == 6
