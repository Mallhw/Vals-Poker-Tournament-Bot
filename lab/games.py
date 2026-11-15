"""Compare multi-hand A/B variants (lab/make_games.py) from the harness's local game logs.

    uv run --project harness python lab/games.py --evals control=eval_a talk=eval_b ... [--project vals-hackathon-matt2]

Per variant: games, mean place (1-6, elimination ranking), mean final chips, bust rate, policy fallbacks, how often
the policy talked, and how often opponents folded to the policy's bets/raises with vs without a say line.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import statistics
import subprocess
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent / "harness" / "runs"


def job_of(eval_id: str, project: str) -> str:
    out = subprocess.run(["ac2", "-p", project, "eval", "get", eval_id], capture_output=True, text=True, cwd="/tmp").stdout
    for line in out.splitlines():
        if line.strip().startswith("compute job:"):
            return line.split(":", 1)[1].strip()
    raise SystemExit(f"no compute job for {eval_id}")


def places(log: dict) -> dict[str, int]:
    """Elimination ranking: survivors by final chips, then busted players by bust hand (later is better)."""
    r = log.get("ranking")
    if isinstance(r, dict) and r and all(isinstance(v, int) for v in r.values()):
        return r
    final = log["final_stacks"]
    elim = log.get("eliminated_on_hand") or {}
    order = sorted(final, key=lambda s: (s not in elim or elim.get(s) is None, elim.get(s) or 0, final[s]), reverse=True)
    return {s: i + 1 for i, s in enumerate(order)}


def analyze(logs: list[dict]) -> dict:
    c = collections.Counter()
    pl, chips = [], []
    for log in logs:
        pol = set()
        for h in log["hands"]:
            pol |= set((h.get("ev") or {}).keys())
        if not pol:
            continue
        seat = sorted(pol)[0]
        c["games"] += 1
        p = places(log)
        pl.append(p[seat])
        chips.append(log["final_stacks"][seat])
        c["busts"] += log["final_stacks"][seat] <= 0
        for h in log["hands"]:
            acts = h["actions"]
            for i, a in enumerate(acts):
                if a["player"] != seat:
                    continue
                c["decisions"] += 1
                mr = a.get("model_response") or {}
                c["fallbacks"] += bool(mr.get("format_error"))
                if a.get("say"):
                    c["says"] += 1
                if a["action"]["type"] in ("bet", "raise"):
                    nxt = [b for b in acts[i + 1:] if b["player"] != seat and b["street"] == a["street"]]
                    if nxt:
                        folded = sum(b["action"]["type"] == "fold" for b in nxt) / len(nxt)
                        key = "talk" if a.get("say") else "silent"
                        c[f"agg_{key}"] += 1
                        c[f"foldrate_{key}"] += folded
    g = max(1, c["games"])
    return {
        "games": c["games"],
        "mean_place": round(statistics.fmean(pl), 2) if pl else None,
        "place_ci95": round(1.96 * statistics.stdev(pl) / len(pl) ** 0.5, 2) if len(pl) > 1 else None,
        "mean_final_chips": round(statistics.fmean(chips), 1) if chips else None,
        "bust_rate": round(c["busts"] / g, 3),
        "fallback_rate": round(c["fallbacks"] / max(1, c["decisions"]), 4),
        "talk_rate": round(c["says"] / max(1, c["decisions"]), 3),
        "opp_fold_after_talk": round(c["foldrate_talk"] / c["agg_talk"], 3) if c["agg_talk"] else None,
        "opp_fold_after_silent": round(c["foldrate_silent"] / c["agg_silent"], 3) if c["agg_silent"] else None,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--evals", nargs="+", required=True, help="variant=eval_id ...")
    p.add_argument("--project", default="vals-hackathon-matt2")
    a = p.parse_args()
    by_job: dict[str, list[dict]] = collections.defaultdict(list)
    for f in glob.glob(str(RUNS / "*.json")):
        try:
            d = json.load(open(f))
        except (OSError, json.JSONDecodeError):
            continue
        by_job[d.get("relay_job_id")].append(d)
    for item in a.evals:
        name, eid = item.split("=", 1)
        print(name, json.dumps(analyze(by_job.get(job_of(eid, a.project), []))))


if __name__ == "__main__":
    main()
