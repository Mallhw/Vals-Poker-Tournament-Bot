"""Does opponents' talk move a model's decisions, and does it leak its own hand? Measured on solver spots, paired.

Score one model with `harness/pokerbench_replies.py` on the plain held-out spots and on talk copies of the same spots
(`rl/spot_talk.py --share 1`, same record order, same `--count/--seed`), then:

    uv run python -m vals_tools.spot_talk_report --plain data/pokerbench/heldout_X.jsonl \
        --talk strong=data/pokerbench/heldout_X_talk-strong.jsonl --talk weak=data/pokerbench/heldout_X_talk-weak.jsonl

For each condition: solver agreement, and the fold / call / raise shares, all on the spots every file has, plus the
paired change against plain with a 95% interval (* = at least 5 points and the interval excludes 0). Then the model's
own talk: how often it sets `say`, and how often that line names its cards or hand strength (hint_report's HAND_TALK).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from vals_tools.hint_report import HAND_TALK


def load(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            rows[json.dumps(r.get("source"), sort_keys=True)] = r
    return rows


def move(r: dict) -> str:
    action = ((r.get("decision") or {}).get("action") or "failed").lower()
    return {"bet": "raise", "check": "call"}.get(action, action)  # passive vs aggressive vs fold


def paired(a: list[float], b: list[float]) -> tuple[float, float]:
    d = [y - x for x, y in zip(a, b)]
    mean = sum(d) / len(d)
    var = sum((x - mean) ** 2 for x in d) / max(len(d) - 1, 1)
    return mean, 1.96 * math.sqrt(var / len(d))


def say_of(r: dict) -> str:
    """The `say` the model put in its submit_action call. The judged `decision` drops it, so read the call itself."""
    for call in (r.get("reply") or {}).get("tool_calls") or []:
        args = (call.get("function") or {}).get("arguments") if isinstance(call, dict) else None
        try:
            args = json.loads(args) if isinstance(args, str) else (args or {})
        except ValueError:
            args = {}
        if isinstance(args, dict) and str(args.get("say") or "").strip():
            return str(args["say"]).strip().lower()
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plain", type=Path, required=True)
    parser.add_argument("--talk", action="append", default=[], help="label=replies file (repeat)")
    args = parser.parse_args()
    files = {"plain": load(args.plain)} | {k: load(Path(v)) for k, v in (t.split("=", 1) for t in args.talk)}
    keys = sorted(set.intersection(*(set(f) for f in files.values())))
    print(f"{len(keys)} spots in every file")
    print(f"{'condition':10} {'agree':>7} {'fold':>6} {'call':>6} {'raise':>6} {'failed':>7} {'talks':>6} {'leaks':>6}")
    for name, rows in files.items():
        rs = [rows[k] for k in keys]
        share = lambda m: sum(move(r) == m for r in rs) / len(rs)
        said = [say_of(r) for r in rs if say_of(r)]
        leaks = sum(bool(HAND_TALK.search(s)) for s in said)
        print(f"{name:10} {sum(r['agrees'] for r in rs) / len(rs):7.1%} {share('fold'):6.1%} {share('call'):6.1%} "
              f"{share('raise'):6.1%} {share('failed'):7.1%} {len(said) / len(rs):6.1%} {leaks:6}")
    base = [files["plain"][k] for k in keys]
    print("\npaired change vs plain (points, 95% interval):")
    for name, rows in files.items():
        if name == "plain":
            continue
        rs = [rows[k] for k in keys]
        parts = []
        for label, f in [("agree", lambda r: float(r["agrees"]))] + [(m, (lambda m: lambda r: float(move(r) == m))(m)) for m in ("fold", "call", "raise")]:
            mean, half = paired([f(r) for r in base], [f(r) for r in rs])
            star = "*" if abs(mean) >= 0.05 and abs(mean) > half else ""
            parts.append(f"{label} {100 * mean:+.1f} ±{100 * half:.1f}{star}")
        print(f"  {name:8} " + " | ".join(parts))
    examples = [(n, say_of(files[n][k])) for n in files for k in keys if HAND_TALK.search(say_of(files[n][k]) or "~")]
    if examples:
        print("\nleaking lines:", "; ".join(f"[{n}] {s}" for n, s in examples[:8]))


if __name__ == "__main__":
    main()
