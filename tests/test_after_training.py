"""vals_tools.after_training: names, the tool-call check, and waiting on AC2 states."""

import json

from vals_tools import after_training as at


def test_deployment_names_name_the_run_and_step() -> None:
    assert at.deployment_name("ckpt", "train_4f13366072d4", 20) == "ckpt-6072d4-step20"
    assert at.deployment_name("opsd", "train_abc", 10) == "opsd-abc-step10"


def test_the_local_game_summary_is_parsed() -> None:
    out = "hand 1 done\n\n6 hands, 23 decisions by vals-hackathon/ckpt-x-step20, 2 failed\n  2 x reply hit the cap\n"
    assert at.parse_check(out) == (23, 2)
    assert at.parse_check("Traceback (most recent call last): ...") is None


def test_wait_returns_the_first_finished_state(monkeypatch) -> None:
    monkeypatch.setattr(at.time, "sleep", lambda s: None)
    states = iter(["queued", "running", "running", "succeeded"])
    assert at.wait("x", lambda: next(states)) == "succeeded"
    assert at.wait("y", lambda: "FAILED") == "failed"


def test_the_talk_check_reads_what_a_checkpoint_said_in_its_local_game(tmp_path) -> None:
    out = "hand 1 done\n\n6 hands, 23 decisions by p/ckpt, 0 failed\nseat0 net: +2.0 BB; log: runs/checks/local-5.json\n"
    assert at.parse_log_path(out) == "runs/checks/local-5.json" and at.parse_log_path("no log") is None

    def decision(say: str) -> dict:
        action = {"type": "fold"}
        return {"street": "preflop", "action": action, "call_amount": 2, "player": "seat0", "say": say,
                "model_response": {"model_name": "p/ckpt"}}

    hands = [{"hand_number": n, "actions": [decision(line)]} for n, line in
             [(0, "i like this spot"), (1, "got nothing but ace-high"), (2, "")]]
    path = tmp_path / "local-5.json"
    path.write_text(json.dumps({"hands": hands}))
    check = at.talk_check(path, "p/ckpt")
    assert check["talk_rate"] == round(2 / 3, 3) and check["say_hand_talk"] == 0.5
    assert check["say_rule"] == round(1 / 3, 3)  # only hand 0 said its rule line ("not much here" was due on hand 1)
    assert at.talk_check(tmp_path / "missing.json", "p/ckpt") is None
