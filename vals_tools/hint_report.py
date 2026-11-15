"""Compare evals of the same deals (e.g. base with and without each hint) by behavior and by chips.

    uv run python -m vals_tools.hint_report --eval plain=eval_a --eval calling=eval_b --eval preflop=eval_c
    uv run python -m vals_tools.hint_report --ids data/launches/hint_evals.txt          # label=eval_id per line

For each eval, from the game logs your harness wrote (harness/runs/, matched to the eval by its AC2 job), the trained
seat's behavior:

- failed decisions (format errors) and how many were replies cut at the token cap; output tokens per decision;
- preflop: VPIP (voluntarily put chips in) and PFR (raised), per hand dealt;
- facing a bet (something to call): how often it folds, calls or raises;
- able to check: how often it bets instead.

And from AC2's scores, BB per hand and the paired difference from the `--base` label (default `plain`), deal by deal,
with a 95% interval. Run it any time: an eval still running reports the hands played so far.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
POLICY = "ac2/policy"


def policy_stats(logs: list[dict[str, Any]]) -> dict[str, Any]:
    """The trained seat's behavior over these games."""
    decisions = [a for log in logs for h in log["hands"] for a in h["actions"]
                 if a["model_response"].get("model_name") == POLICY]
    failed = [a for a in decisions if a["model_response"].get("format_error")]
    clean = [a for a in decisions if not a["model_response"].get("format_error")]
    tokens = sorted(a["model_response"]["out_tokens"] for a in decisions if a["model_response"].get("out_tokens"))
    hands = [
        [a for a in h["actions"] if a["model_response"].get("model_name") == POLICY and a["street"] == "preflop"]
        for log in logs for h in log["hands"]
        if any(a["model_response"].get("model_name") == POLICY for a in h["actions"])
    ]
    facing = Counter(a["action"]["type"] for a in clean if a["call_amount"] > 0)
    free = Counter(a["action"]["type"] for a in clean if a["call_amount"] == 0)
    share = lambda part, whole: round(part / whole, 3) if whole else None  # noqa: E731
    return {
        "games": len(logs),
        "decisions": len(decisions),
        "failed": share(len(failed), len(decisions)),
        "cut_at_cap": share(sum(bool(a["model_response"].get("truncated")) for a in failed), len(decisions)),
        "tokens_median": tokens[len(tokens) // 2] if tokens else None,
        "tokens_p90": tokens[int(len(tokens) * 0.9)] if tokens else None,
        "vpip": share(sum(any(a["action"]["type"] in ("call", "bet", "raise") for a in h) for h in hands), len(hands)),
        "pfr": share(sum(any(a["action"]["type"] in ("bet", "raise") for a in h) for h in hands), len(hands)),
        "facing_bet": sum(facing.values()),
        "facing_fold": share(facing["fold"], sum(facing.values())),
        "facing_call": share(facing["call"], sum(facing.values())),
        "facing_raise": share(facing["raise"] + facing["bet"], sum(facing.values())),
        "can_check": sum(free.values()),
        "bets_when_can_check": share(free["bet"] + free["raise"], sum(free.values())),
    }


def logs_by_job(runs: Path, jobs: set[str]) -> dict[str, list[dict[str, Any]]]:
    found: dict[str, list[dict[str, Any]]] = {job: [] for job in jobs}
    for path in runs.glob("*.json"):
        try:
            log = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue  # a log still being written
        if log.get("relay_job_id") in found:
            found[log["relay_job_id"]].append(log)
    return found


def read_ids(path: Path) -> dict[str, str]:
    pairs = [line.strip().split("=", 1) for line in path.read_text().splitlines() if "=" in line]
    return {label: eval_id for label, eval_id in pairs}


def table(report: dict[str, dict[str, Any]]) -> str:
    columns = [("games", "games"), ("failed", "failed"), ("cut_at_cap", "cut"), ("tokens_median", "tok med"),
               ("tokens_p90", "tok p90"), ("vpip", "VPIP"), ("pfr", "PFR"), ("facing_fold", "fold"),
               ("facing_call", "call"), ("facing_raise", "raise"), ("bets_when_can_check", "bet/chk"),
               ("bb_per_hand", "BB/hand"), ("vs_base", "vs base (95%)")]
    rows = [[label] + [cell(r.get(key), key in PERCENT) for key, _ in columns] for label, r in report.items()]
    head = ["eval"] + [name for _, name in columns]
    widths = [max(len(str(x)) for x in col) for col in zip(head, *rows)]
    fmt = lambda row: "  ".join(str(x).rjust(w) for x, w in zip(row, widths))  # noqa: E731
    return "\n".join([fmt(head), *map(fmt, rows)])


PERCENT = {"failed", "cut_at_cap", "vpip", "pfr", "facing_fold", "facing_call", "facing_raise", "bets_when_can_check"}


def cell(value: Any, percent: bool = False) -> str:
    if isinstance(value, dict):  # vs_base
        return f"{value['diff']:+.2f} ±{value['ci95']:.2f} (n={value['paired_hands']})"
    if value is None:
        return "-"
    if percent:
        return f"{value:.0%}"
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default="vals-hackathon")
    parser.add_argument("--eval", action="append", default=[], help="label=eval_id (repeat)")
    parser.add_argument("--ids", type=Path, help="a file of label=eval_id lines")
    parser.add_argument("--base", default="plain", help="the label to pair the others against (default plain)")
    parser.add_argument("--runs", type=Path, default=ROOT / "harness" / "runs", help="the harness's game logs")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "launches" / "hint_report.json")
    args = parser.parse_args()

    from ac2.sdk import Client

    from vals_tools.score import score_rows, summarize

    evals = read_ids(args.ids) if args.ids else {}
    evals |= dict(pair.split("=", 1) for pair in args.eval)
    if not evals:
        parser.error("pass --eval label=eval_id or --ids FILE")
    client = Client(project=args.project)
    info = {label: client.eval.get(eval_id) for label, eval_id in evals.items()}
    logs = logs_by_job(args.runs, {i.job_id for i in info.values() if i.job_id})
    scores = {label: score_rows(client, eval_id=eval_id) for label, eval_id in evals.items()}
    base = scores.get(args.base)
    report: dict[str, dict[str, Any]] = {}
    for label, eval_id in evals.items():
        summary = summarize(scores[label], base if label != args.base else None)
        report[label] = {"eval": eval_id, "status": str(info[label].status), **policy_stats(logs.get(info[label].job_id, [])),
                         "scored_hands": summary.get("hands", 0), "bb_per_hand": summary.get("ev_bb_per_hand"),
                         "vs_base": summary.get("vs_base")}
    print(table(report))
    print("\nBB/hand: the grader's EV (all-ins before the river at expected value). vs base: paired by deal.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
