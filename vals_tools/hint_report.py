"""Compare evals of the same deals (e.g. base with and without each hint) by behavior and by chips.

    uv run python -m vals_tools.hint_report --eval plain=eval_a --eval calling=eval_b --eval preflop=eval_c
    uv run python -m vals_tools.hint_report --ids data/launches/hint_evals.txt          # label=eval_id per line

For each eval, from the game logs your harness wrote (harness/runs/, matched to the eval by its AC2 job), the trained
seat's behavior:

- opponent health (opp fail, opp p90s): the share of opponent decisions that failed (a timed-out opponent checks or
  folds) and their 90th-percentile reply time. Read it first: a run with many failures is not poker;
- failed decisions (format errors) and how many were replies cut at the token cap; output tokens per decision;
- preflop: VPIP (voluntarily put chips in) and PFR (raised), per hand dealt;
- facing a bet (something to call): how often it folds, calls or raises;
- able to check: how often it bets instead;
- table talk: how often it talks itself, and how it answers bets after another player spoke this hand (t-fold,
  t-call, t-raise), which is where scripted talk (rl/make_talk_dataset.py) would move it if talk moves it.

And from AC2's scores, BB per hand and the paired difference from the `--base` label (default `plain`), deal by deal,
with a 95% interval. Run it any time: an eval still running reports the hands played so far.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
POLICY = "ac2/policy"
TALK_RULE = ROOT / "opsd" / "hints" / "talk_rule.txt"
# Table talk that says something about the speaker's hand (the table lowercases talk first): what our own talk must not.
HAND_TALK = re.compile(
    r"\b(aces?|kings?|queens?|jacks?|tens?|deuces?|pairs?|pocket|suited|offsuit|flush|straight|sets?|trips|nuts|monster|"
    r"air|nothing|bluff\w*|strong|weak|draws?|junk|trash|premium|top pair)\b"
)


def load_talk_rule(path: Path = TALK_RULE) -> dict[int, str]:
    """The hand-number rule's lines by remainder, from lines like `0: "i like this spot"`."""
    rule = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        found = re.fullmatch(r'\s*(\d+):\s*"([^"]*)"\s*', line)
        if found:
            rule[int(found[1])] = found[2]
    return rule


def prop_diff(k1: int, n1: int, k2: int, n2: int) -> dict[str, Any] | None:
    """Difference of two proportions (k1/n1 minus k2/n2) with a 95% normal-approximation interval."""
    if not n1 or not n2:
        return None
    p1, p2 = k1 / n1, k2 / n2
    half = 1.96 * (p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2) ** 0.5
    return {"diff": round(p1 - p2, 3), "ci95": round(half, 3), "n": [n1, n2]}


def talk_effects(report: dict[str, dict[str, Any]], base: str = "plain", threshold: float = 0.05) -> dict[str, Any]:
    """How each eval's answers to bets differ from the base eval's, with the plan's call: an effect counts when it is
    at least `threshold` and its interval excludes zero. Only large effects show at a few hundred decisions."""
    ref = report.get(base, {}).get("facing_counts")
    effects: dict[str, Any] = {}
    for label, row in report.items():
        mine = row.get("facing_counts")
        if label == base or not mine or not ref:
            continue
        effects[label] = {}
        for action in ("fold", "call", "raise"):
            diff = prop_diff(mine[action], mine["n"], ref[action], ref["n"])
            if diff:
                diff["significant"] = abs(diff["diff"]) >= threshold and abs(diff["diff"]) > diff["ci95"]
                effects[label][action] = diff
    return effects


def effects_text(effects: dict[str, Any], base: str = "plain") -> str:
    lines = [f"Answers to bets vs {base} (95% interval; * = at least 5 points and the interval excludes 0):"]
    for label, by_action in effects.items():
        cells = [f"{a} {d['diff']:+.0%} ±{d['ci95']:.0%}{'*' if d['significant'] else ''}" for a, d in by_action.items()]
        n = next(iter(by_action.values()))["n"] if by_action else ["?", "?"]
        lines.append(f"  {label:>10}: " + " | ".join(cells) + f"   (n={n[0]} vs {n[1]})")
    return "\n".join(lines)


def policy_stats(
    logs: list[dict[str, Any]], rule: dict[int, str] | None = None, policy: str = POLICY
) -> dict[str, Any]:
    """The trained seat's behavior over these games. With `rule`, how often its talk is the rule's line for the hand.
    `policy` is the seat's model name in the logs: the harness names it `ac2/policy`, `play_local.py` by deployment."""
    decisions = [a for log in logs for h in log["hands"] for a in h["actions"]
                 if a["model_response"].get("model_name") == policy]
    failed = [a for a in decisions if a["model_response"].get("format_error")]
    clean = [a for a in decisions if not a["model_response"].get("format_error")]
    tokens = sorted(a["model_response"]["out_tokens"] for a in decisions if a["model_response"].get("out_tokens"))
    hands = [
        [a for a in h["actions"] if a["model_response"].get("model_name") == policy and a["street"] == "preflop"]
        for log in logs for h in log["hands"]
        if any(a["model_response"].get("model_name") == policy for a in h["actions"])
    ]
    facing = Counter(a["action"]["type"] for a in clean if a["call_amount"] > 0)
    free = Counter(a["action"]["type"] for a in clean if a["call_amount"] == 0)
    # Facing a bet after someone else spoke this hand: where talk would move the decision, if it does.
    after_talk = Counter(
        a["action"]["type"]
        for log in logs
        for h in log["hands"]
        for i, a in enumerate(h["actions"])
        if a["model_response"].get("model_name") == policy and not a["model_response"].get("format_error")
        and a["call_amount"] > 0 and any(b.get("say") and b["player"] != a["player"] for b in h["actions"][:i])
    )
    share = lambda part, whole: round(part / whole, 3) if whole else None  # noqa: E731
    # Our own talk: how often it follows the rule, and how often it says something about our hand.
    said = [a for a in decisions if a.get("say")]
    by_hand = [(h["hand_number"], a) for log in logs for h in log["hands"] for a in h["actions"]
               if a["model_response"].get("model_name") == policy and not a["model_response"].get("format_error")
               ] if rule else []
    compliant = sum(a.get("say", "") == rule.get(n % len(rule)) for n, a in by_hand) if rule else 0
    # How opponents answer our bets: right after we spoke, or after we bet in silence.
    answers = {"talk": Counter(), "silent": Counter()}
    for log in logs:
        for h in log["hands"]:
            last = None
            for a in h["actions"]:
                if a["model_response"].get("model_name") == policy:
                    last = ("talk" if a.get("say") else "silent") if a["action"]["type"] in ("bet", "raise") else None
                elif last and a["call_amount"] > 0 and not a["model_response"].get("format_error"):
                    answers[last][a["action"]["type"]] += 1
    both = answers["talk"] + answers["silent"]
    # Opponent health: a timed-out opponent checks or folds, so a run where many fail is not poker. Check this first.
    opponents = [a for log in logs for h in log["hands"] for a in h["actions"]
                 if a["model_response"].get("model_name", "").startswith("opponent/")]
    opp_latency = sorted(a["model_response"]["latency_seconds"] for a in opponents
                         if a["model_response"].get("latency_seconds") is not None)
    return {
        "opp_decisions": len(opponents),
        "opp_failed": share(sum(bool(a["model_response"].get("format_error")) for a in opponents), len(opponents)),
        "opp_latency_p90": round(opp_latency[int(len(opp_latency) * 0.9)], 1) if opp_latency else None,
        "games": len(logs),
        "decisions": len(decisions),
        "failed": share(len(failed), len(decisions)),
        "cut_at_cap": share(sum(bool(a["model_response"].get("truncated")) for a in failed), len(decisions)),
        "tokens_median": tokens[len(tokens) // 2] if tokens else None,
        "tokens_p90": tokens[int(len(tokens) * 0.9)] if tokens else None,
        "vpip": share(sum(any(a["action"]["type"] in ("call", "bet", "raise") for a in h) for h in hands), len(hands)),
        "pfr": share(sum(any(a["action"]["type"] in ("bet", "raise") for a in h) for h in hands), len(hands)),
        "facing_bet": sum(facing.values()),
        "facing_counts": {"n": sum(facing.values()), "fold": facing["fold"], "call": facing["call"],
                          "raise": facing["raise"] + facing["bet"]},
        "facing_fold": share(facing["fold"], sum(facing.values())),
        "facing_call": share(facing["call"], sum(facing.values())),
        "facing_raise": share(facing["raise"] + facing["bet"], sum(facing.values())),
        "can_check": sum(free.values()),
        "bets_when_can_check": share(free["bet"] + free["raise"], sum(free.values())),
        "talk_rate": share(sum(bool(a.get("say")) for a in decisions), len(decisions)),
        "after_talk_facing": sum(after_talk.values()),
        "after_talk_fold": share(after_talk["fold"], sum(after_talk.values())),
        "after_talk_call": share(after_talk["call"], sum(after_talk.values())),
        "after_talk_raise": share(after_talk["raise"] + after_talk["bet"], sum(after_talk.values())),
        "say_rule": share(compliant, len(by_hand)) if rule else None,
        "say_hand_talk": share(sum(bool(HAND_TALK.search(a["say"])) for a in said), len(said)),
        "opp_answers_to_our_bet": sum(both.values()),
        "opp_fold_to_bet": share(both["fold"], sum(both.values())),
        "opp_fold_to_talk_bet": share(answers["talk"]["fold"], sum(answers["talk"].values())),
        "opp_fold_to_silent_bet": share(answers["silent"]["fold"], sum(answers["silent"].values())),
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
    columns = [("games", "games"), ("opp_failed", "opp fail"), ("opp_latency_p90", "opp p90s"),
               ("failed", "failed"), ("cut_at_cap", "cut"), ("tokens_median", "tok med"),
               ("tokens_p90", "tok p90"), ("vpip", "VPIP"), ("pfr", "PFR"), ("facing_fold", "fold"),
               ("facing_call", "call"), ("facing_raise", "raise"), ("bets_when_can_check", "bet/chk"),
               ("talk_rate", "talks"), ("after_talk_facing", "n talk"), ("after_talk_fold", "t-fold"),
               ("after_talk_call", "t-call"), ("after_talk_raise", "t-raise"),
               ("say_rule", "rule"), ("say_hand_talk", "leaks"), ("opp_fold_to_bet", "opp folds"),
               ("bb_per_hand", "BB/hand"), ("vs_base", "vs base (95%)")]
    rows = [[label] + [cell(r.get(key), key in PERCENT) for key, _ in columns] for label, r in report.items()]
    head = ["eval"] + [name for _, name in columns]
    widths = [max(len(str(x)) for x in col) for col in zip(head, *rows)]
    fmt = lambda row: "  ".join(str(x).rjust(w) for x, w in zip(row, widths))  # noqa: E731
    return "\n".join([fmt(head), *map(fmt, rows)])


PERCENT = {"opp_failed", "failed", "cut_at_cap", "vpip", "pfr", "facing_fold", "facing_call", "facing_raise", "bets_when_can_check",
           "talk_rate", "after_talk_fold", "after_talk_call", "after_talk_raise", "say_rule", "say_hand_talk",
           "opp_fold_to_bet"}


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
    parser.add_argument("--talk-rule", type=Path, default=TALK_RULE, help="the hand-number talk rule to check compliance with")
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
    rule = load_talk_rule(args.talk_rule) if args.talk_rule.exists() else None
    report: dict[str, dict[str, Any]] = {}
    for label, eval_id in evals.items():
        summary = summarize(scores[label], base if label != args.base else None)
        stats = policy_stats(logs.get(info[label].job_id, []), rule)
        report[label] = {"eval": eval_id, "status": str(info[label].status), **stats,
                         "scored_hands": summary.get("hands", 0), "bb_per_hand": summary.get("ev_bb_per_hand"),
                         "vs_base": summary.get("vs_base")}
    effects = talk_effects(report, args.base)
    print(table(report))
    print("\nBB/hand: the grader's EV (all-ins before the river at expected value). vs base: paired by deal.")
    print("rule: share of our decisions saying the talk rule's line for the hand; leaks: share of our talk about our hand;")
    print("opp folds: opponents' folds to our bets.")
    if effects:
        print("\n" + effects_text(effects, args.base))
    report["_talk_effects"] = effects
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
