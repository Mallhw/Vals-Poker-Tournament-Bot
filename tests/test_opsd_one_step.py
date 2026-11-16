"""opsd/train.py --rollout-mode one_step: the student samples its own reply; offline stays as it was."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "opsd")]

import train as opsd_train  # noqa: E402


def config(*argv: str) -> dict:
    return opsd_train.run_config(opsd_train.parser().parse_args(["--dataset", "d", *argv]), "n").to_json()


def test_one_step_samples_with_one_engine_at_the_event_cap() -> None:
    cfg = config("--rollout-mode", "one_step", "--steps", "30")
    exp = cfg["experiment_cfg"]
    assert exp["method"]["rollout_mode"] == "one_step"
    assert exp["rollout"] == {"n_inference_replicas": 1, "max_response_len": 5000}
    assert exp["step"]["samples_per_problem"] == 1 and exp["step"]["num_train_steps"] == 30
    assert exp["eval"]["mode"] == "off" and "learning_rate" not in json.dumps(cfg)
    assert config("--rollout-mode", "one_step", "--samples", "2", "--max-response-len", "4000")["experiment_cfg"][
        "rollout"
    ]["max_response_len"] == 4000


def test_one_step_refuses_a_total_length_and_offline_is_unchanged() -> None:
    with pytest.raises(SystemExit, match="max-total-len"):
        config("--rollout-mode", "one_step", "--max-total-len", "9000")
    offline = config()["experiment_cfg"]
    assert offline["method"]["rollout_mode"] == "offline"
    assert offline["rollout"] == {"n_inference_replicas": 0} and offline["step"]["samples_per_problem"] == 1
