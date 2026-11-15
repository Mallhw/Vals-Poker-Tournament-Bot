"""vals_tools.hint_report: the trained seat's behavior from game logs, matched to evals by job."""

import json

from vals_tools import hint_report as hr

POLICY = {"model_name": "ac2/policy"}
OTHER = {"model_name": "opponent/p/opp"}


def act(street: str, kind: str, call: int, response: dict | None = None, player: str = "seat0") -> dict:
    return {"street": street, "action": {"type": kind}, "call_amount": call, "player": player,
            "model_response": response or POLICY}


def game(*hands: list[dict], job: str = "job_a") -> dict:
    return {"relay_job_id": job, "hands": [{"actions": list(h)} for h in hands]}


def test_policy_behavior_is_counted_from_its_own_decisions_only() -> None:
    cut = POLICY | {"format_error": True, "truncated": True, "out_tokens": 5000}
    logs = [
        game([act("preflop", "raise", 2, POLICY | {"out_tokens": 400}), act("preflop", "call", 4, OTHER, "seat1"),
              act("flop", "bet", 0, POLICY | {"out_tokens": 800})]),
        game([act("preflop", "call", 2, POLICY | {"out_tokens": 600}), act("flop", "fold", 6, POLICY | {"out_tokens": 300})]),
        game([act("preflop", "fold", 2, cut)]),  # forced fold after a cut reply: counted as failed, not as a fold
        game([act("preflop", "raise", 4, OTHER, "seat1")]),  # the trained seat never acted: not a hand it played
    ]
    s = hr.policy_stats(logs)
    assert s["games"] == 4 and s["decisions"] == 5 and s["failed"] == 0.2 and s["cut_at_cap"] == 0.2
    assert s["vpip"] == round(2 / 3, 3) and s["pfr"] == round(1 / 3, 3)
    assert s["facing_bet"] == 3 and s["facing_fold"] == round(1 / 3, 3) and s["facing_raise"] == round(1 / 3, 3)
    assert s["can_check"] == 1 and s["bets_when_can_check"] == 1.0
    assert s["tokens_median"] == 600


def test_logs_are_matched_to_evals_by_job(tmp_path) -> None:
    (tmp_path / "a.json").write_text(json.dumps(game([], job="job_a")))
    (tmp_path / "b.json").write_text(json.dumps(game([], job="job_b")))
    (tmp_path / "partial.json").write_text('{"hands": [')
    found = hr.logs_by_job(tmp_path, {"job_a", "job_c"})
    assert len(found["job_a"]) == 1 and found["job_c"] == [] and "job_b" not in found


def test_ids_file_and_table(tmp_path) -> None:
    ids = tmp_path / "ids.txt"
    ids.write_text("plain=eval_1\ncalling=eval_2\n")
    assert hr.read_ids(ids) == {"plain": "eval_1", "calling": "eval_2"}
    report = {"plain": {"games": 10, "vpip": 0.5, "bb_per_hand": -0.31},
              "calling": {"games": 9, "vs_base": {"diff": 0.42, "ci95": 0.8, "paired_hands": 9}}}
    text = hr.table(report)
    assert "50%" in text and "-0.31" in text and "+0.42 ±0.80 (n=9)" in text
