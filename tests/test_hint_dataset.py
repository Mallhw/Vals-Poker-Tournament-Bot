"""Hinted copies of a split play the same deals, and `score` pairs them with the plain split by deal."""

import importlib.util
from pathlib import Path

import pytest

from vals_tools.score import summarize
from vals_tools.tasks import SPLITS, Opponents, hand_tasks

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_hint_dataset", ROOT / "rl" / "make_hint_dataset.py")
make_hint_dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_hint_dataset)

OPPONENT = Opponents("vals-hackathon/ac-vals-model", "chat", None)


def test_a_hinted_split_has_the_plain_splits_deals_plus_the_hint() -> None:
    plain = hand_tasks(*SPLITS["select"], OPPONENT)
    hinted = make_hint_dataset.hinted_tasks("select", "Raise more.", OPPONENT)
    assert len(hinted) == len(plain) == SPLITS["select"][0]
    for p, h in zip(plain, hinted):
        assert h.env_params == {**p.env_params, "system_hint": "Raise more."}
        assert h.content_hash != p.content_hash  # rebuilt, so the hash covers the hint
    assert len(make_hint_dataset.hinted_tasks("select", "x", OPPONENT, limit=12)) == 12


def test_the_hint_comes_from_text_or_from_a_hint_module(tmp_path: Path) -> None:
    (tmp_path / "h.txt").write_text("  Fold more.\n", encoding="utf-8")
    assert make_hint_dataset.read_hint(tmp_path / "h.txt") == "Fold more."
    assert make_hint_dataset.read_hint(ROOT / "opsd" / "hint.py").startswith("Strategy notes")
    (tmp_path / "empty.txt").write_text("\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        make_hint_dataset.read_hint(tmp_path / "empty.txt")


def row(ev: float, deal: str | None = None) -> dict:
    return {"ev_bb": ev, "realized_bb": ev, "policy_format_errors": 0, **({"deal": deal} if deal else {})}


def test_scores_pair_by_deal_across_datasets_and_by_task_otherwise() -> None:
    hinted = {"t9": row(3.0, "11:seat2"), "t8": row(-1.0, "12:seat3"), "t7": row(0.0, "99:seat0")}
    plain = {"a1": row(1.0, "11:seat2"), "a2": row(-2.0, "12:seat3")}
    out = summarize(hinted, plain)["vs_base"]
    assert out["paired_by"] == "deal" and out["paired_hands"] == 2 and out["diff"] == pytest.approx(1.5)
    legacy = {"a1": row(1.0), "a2": row(-2.0)}  # scores from before the grader named the deal
    out = summarize({"a1": row(2.0, "11:seat2"), "a2": row(0.0, "12:seat3")}, legacy)["vs_base"]
    assert out["paired_by"] == "task" and out["paired_hands"] == 2 and out["diff"] == pytest.approx(1.5)
