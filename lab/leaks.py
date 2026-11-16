"""Leak report for the policy seat over harness game logs (evals and training games played on this machine).

    uv run --project harness python lab/leaks.py --eval eval_xxx        # games of one eval (via its compute job)
    uv run --project harness python lab/leaks.py --job job_xxx
    uv run --project harness python lab/leaks.py --files harness/runs/*.json

Measures what the arena showed matters: busts and bust risk, big calls / all-ins with weak hands, format errors,
VPIP/PFR, EV per hand and the current grader's score.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness" / "src"))

from common import RUNS, ac2, field  # noqa: E402
from poker_harness import grading  # noqa: E402
from poker_harness.spots import chen_score  # noqa: E402

WEAK_CHEN = 8  # below ~8 on the Chen scale: worse than roughly TT / AQ, too weak to call off a stack with


def policy_seats(log: dict) -> list[str]:
    seats = set()
    for h in log.get("hands", []):
        seats |= set((h.get("ev") or {}).keys())
    return sorted(seats)


def logs_for_job(job: str) -> list[dict]:
    out = []
    for f in glob.glob(str(RUNS / "*.json")):
        try:
            d = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue
        if d.get("relay_job_id") == job:
            out.append(d)
    return out


def analyze(logs: list[dict]) -> dict:
    s = collections.Counter()
    errs = collections.Counter()
    evs, scores, weak_calls = [], [], []
    for log in logs:
        seats = policy_seats(log)
        if not seats:
            continue
        try:
            scores.append(grading.grade_log(log, seats)["score"])
        except Exception:  # noqa: BLE001 - a log the grader can't read
            pass
        s["games"] += 1
        for h in log["hands"]:
            for seat in seats:
                if seat not in h["starting_state"]:
                    continue
                stack = h["starting_state"][seat]["starting_stack"]
                hole = h["starting_state"][seat]["hole_cards"]
                chen = chen_score(hole)
                lab = (h.get("ev") or {}).get(seat) or {}
                s["hands"] += 1
                evs.append(lab.get("ev_bb", 0.0))
                s["bust_prob"] += lab.get("bust_prob", 0.0)
                s["busted"] += h["ending_stacks"].get(seat, 1) <= 0
                mine = [a for a in h["actions"] if a["player"] == seat]
                pre = [a for a in mine if a["street"] == "preflop"]
                s["vpip"] += any(a["action"]["type"] in ("call", "raise", "bet") for a in pre)
                s["pfr"] += any(a["action"]["type"] in ("raise", "bet") for a in pre)
                for a in mine:
                    s["decisions"] += 1
                    mr = a.get("model_response") or {}
                    if mr.get("format_error"):
                        s["format_errors"] += 1
                        errs[str(mr.get("error") or mr.get("format_error"))[:60]] += 1
                    before = a["stack_after"] - a["chip_change"]
                    big = before > 0 and (-a["chip_change"]) >= 0.4 * before
                    if a["action"]["type"] in ("call", "raise", "bet") and big:
                        s["big_commits"] += 1
                        if chen < WEAK_CHEN and a["street"] == "preflop":
                            s["weak_big_commits"] += 1
                            weak_calls.append({"hole": hole, "action": a["action"]["type"],
                                               "chips": -a["chip_change"], "stack": before,
                                               "result_bb": round(lab.get("realized_bb", 0.0), 1)})
    h = max(1, s["hands"])
    return {
        "games": s["games"],
        "hands": s["hands"],
        "decisions": s["decisions"],
        "format_error_rate": round(s["format_errors"] / max(1, s["decisions"]), 4),
        "format_errors": dict(errs.most_common(5)),
        "vpip": round(s["vpip"] / h, 3),
        "pfr": round(s["pfr"] / h, 3),
        "ev_bb_per_hand": round(statistics.fmean(evs), 3) if evs else None,
        "ev_ci95": round(1.96 * statistics.stdev(evs) / len(evs) ** 0.5, 3) if len(evs) > 1 else None,
        "mean_score": round(statistics.fmean(scores), 3) if scores else None,
        "bust_prob_per_100_hands": round(100 * s["bust_prob"] / h, 2),
        "busts_per_100_hands": round(100 * s["busted"] / h, 2),
        "big_commits_per_100": round(100 * s["big_commits"] / h, 2),
        "weak_preflop_stackoffs_per_100": round(100 * s["weak_big_commits"] / h, 2),
        "weak_stackoff_examples": weak_calls[:8],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--eval")
    g.add_argument("--job")
    g.add_argument("--files", nargs="+")
    p.add_argument("--project", default="vals-hackathon-matt")
    a = p.parse_args()
    if a.files:
        logs = [json.load(open(f)) for f in a.files]
    else:
        job = a.job or field(ac2("eval", "get", a.eval, project=a.project), "compute job")
        logs = logs_for_job(job)
    print(json.dumps(analyze(logs), indent=1))


if __name__ == "__main__":
    main()
