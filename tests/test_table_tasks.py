"""Multi-hand tables and spreading opponents over our own deployments: deals stay paired across every copy of a split."""

import importlib.util
from collections import Counter
from pathlib import Path

import pytest

from vals_tools.tasks import (
    SEATS,
    SPLITS,
    TABLE_SPLITS,
    Opponents,
    hand_tasks,
    split_tasks,
    spread_opponents,
    table_smoke_tasks,
    table_task,
)

ROOT = Path(__file__).resolve().parents[1]
OPPONENT = Opponents("vals-hackathon/ac-vals-model", "chat", None)
COPIES = ["vals-hackathon/base-h1", "vals-hackathon/base-h2"]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "rl" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_table_task_is_an_event_game_with_the_policy_in_one_seat() -> None:
    task = table_task(123, 4, OPPONENT, 12)
    params = task.env_params
    assert params["total_hands"] == 12 and params["policy_seats"] == [4] and params["preset"] == "event"
    assert params["seats"] == SEATS and params["opponent_model"] == "vals-hackathon/ac-vals-model"
    assert "reset_stacks" not in params  # the event preset carries chips between hands
    assert "12 hands from seat 4" in task.input[0].content


def test_table_splits_cycle_every_seat_and_share_no_seed_with_single_hands() -> None:
    tables = split_tasks("select", OPPONENT, total_hands=12)
    assert len(tables) == TABLE_SPLITS["select"][0] and tables[0].env_params["total_hands"] == 12
    assert {t.env_params["policy_seats"][0] for t in tables[:SEATS]} == set(range(SEATS))
    singles = {t.env_params["seed"] for s in SPLITS for t in hand_tasks(*SPLITS[s], OPPONENT)}
    assert not singles & {t.env_params["seed"] for s in TABLE_SPLITS for t in split_tasks(s, OPPONENT, total_hands=12)}
    assert len(split_tasks("select", OPPONENT, total_hands=12, limit=7)) == 7
    with pytest.raises(ValueError, match="no 12-hand tables for split 'eval'"):
        split_tasks("eval", OPPONENT, total_hands=12)
    assert split_tasks("select", OPPONENT)[0].env_params["total_hands"] == 1  # single hands by default


def test_opponents_are_spread_by_seed_unrelated_to_the_policy_seat() -> None:
    plain = split_tasks("select", OPPONENT)
    spread = spread_opponents(plain, COPIES)
    assert [t.env_params["seed"] for t in spread] == [t.env_params["seed"] for t in plain]
    assert {t.env_params["opponent_model"] for t in spread} == set(COPIES)
    assert all(t.content_hash != p.content_hash for t, p in zip(spread, plain))
    for seat in range(SEATS):  # every policy seat meets both copies, so seat and opponent aren't confounded
        met = Counter(t.env_params["opponent_model"] for t in spread if t.env_params["policy_seats"] == [seat])
        assert set(met) == set(COPIES) and min(met.values()) > 0.3 * sum(met.values()) / 2
    assert spread_opponents(plain, None) is plain and spread_opponents(plain, []) is plain
    smoke = table_smoke_tasks()  # every seat is the policy: no opponent to replace
    assert spread_opponents(smoke, COPIES) == smoke


def test_every_copy_of_a_split_gives_a_deal_the_same_opponent() -> None:
    hint = load("make_hint_dataset")
    talk = load("make_talk_dataset")
    plain = spread_opponents(split_tasks("select", OPPONENT), COPIES)
    hinted = hint.hinted_tasks("select", "Raise more.", OPPONENT, models=COPIES)
    strong = talk.talk_tasks("select", "strong", 0.7, OPPONENT, models=COPIES)
    for p, h, s in zip(plain, hinted, strong):
        assert h.env_params["opponent_model"] == s.env_params["opponent_model"] == p.env_params["opponent_model"]
        assert h.env_params["seed"] == s.env_params["seed"] == p.env_params["seed"]
    assert "opponent_talk" in strong[0].env_params and "system_hint" in hinted[0].env_params


def test_table_datasets_are_named_for_their_length_and_split() -> None:
    make_dataset = load("make_dataset")
    built = make_dataset.table_datasets("pokerbots-bo", OPPONENT, 12, ["train", "select"])
    assert list(built) == ["pokerbots-bo-t12-train", "pokerbots-bo-t12-select"]
    assert [len(t) for t in built.values()] == [TABLE_SPLITS["train"][0], TABLE_SPLITS["select"][0]]
    hinted = load("make_hint_dataset").hinted_tasks("select", "x", OPPONENT, total_hands=12, limit=6)
    assert len(hinted) == 6 and hinted[0].env_params["total_hands"] == 12 and hinted[0].env_params["system_hint"] == "x"
    talking = load("make_talk_dataset").talk_tasks("select", "weak", 0.5, OPPONENT, total_hands=12)
    assert talking[0].env_params["total_hands"] == 12 and talking[0].env_params["opponent_talk"]["rate"] == 0.5
