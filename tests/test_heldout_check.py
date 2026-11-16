"""vals_tools.heldout_check: solver agreement on the spots every file answered."""

import json

from vals_tools import heldout_check as hc


def reply(row: int, solver: str, played: str | None, agrees: bool, tokens: int = 500) -> dict:
    return {"source": {"file": "t.csv", "row": row}, "label": {"action": solver}, "clean": played is not None,
            "decision": {"action": played} if played else None, "agrees": agrees, "reply": {"out_tokens": tokens}}


def write(path, rows) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_agreement_is_compared_on_shared_spots_only(tmp_path) -> None:
    base, ckpt = tmp_path / "heldout_base.jsonl", tmp_path / "heldout_ckpt.jsonl"
    write(base, [reply(0, "raise", "call", False), reply(1, "fold", "fold", True), reply(2, "call", "call", True)])
    write(ckpt, [reply(0, "raise", "raise", True, 900), reply(1, "fold", None, False)])  # row 2 missing
    s = hc.summarize({"base": base, "ckpt": ckpt, "absent": tmp_path / "nope.jsonl"})
    assert set(s) == {"base", "ckpt"} and s["base"]["spots"] == s["ckpt"]["spots"] == 2
    assert s["base"]["agree"] == 0.5 and s["ckpt"]["agree"] == 0.5
    assert s["base"]["agree_when_bet/raise"] == 0.0 and s["ckpt"]["agree_when_bet/raise"] == 1.0
    assert s["ckpt"]["clean"] == 0.5 and s["ckpt"]["plays_bet/raise"] == 1.0 and s["ckpt"]["agree_when_check"] is None
    text = hc.table(s)
    assert "solver folds" in text and "50%" in text


def test_keep_best_keeps_the_highest_agreement_and_names_the_one_to_delete(tmp_path) -> None:
    path = tmp_path / "r.jsonl"
    write(path, [reply(0, "fold", "fold", True), reply(1, "call", "fold", False), reply(2, "check", "check", True)])
    assert hc.agreement(path) == 2 / 3 and hc.agreement(tmp_path / "missing.jsonl") == 0.0
    best, loser = hc.keep_best(None, 0.50, "step10")
    assert best == (0.50, "step10") and loser is None
    best, loser = hc.keep_best(best, 0.58, "step20")
    assert best == (0.58, "step20") and loser == "step10"  # a better one replaces the old best
    best, loser = hc.keep_best(best, 0.55, "step30")
    assert best == (0.58, "step20") and loser == "step30"  # a worse one is deleted right away
    best, loser = hc.keep_best(best, 0.58, "step40")
    assert best == (0.58, "step20") and loser == "step40"  # ties keep the earlier, shorter-trained checkpoint


def test_hold_best_flag_and_docs() -> None:
    import subprocess
    import sys
    out = subprocess.run([sys.executable, "-m", "vals_tools.heldout_check", "--help"], capture_output=True, text=True,
                         cwd=hc.ROOT).stdout
    assert "--hold-best" in out and "_best.json" in out
