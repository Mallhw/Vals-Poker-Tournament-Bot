"""vals_tools.after_training: names, the tool-call check, and waiting on AC2 states."""

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
