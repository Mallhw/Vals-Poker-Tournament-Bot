"""After a training run: deploy its checkpoints, check each still calls submit_action, evaluate, and score vs base.

    uv run python -m vals_tools.after_training --train train_xxx --base-eval eval_xxx
    uv run python -m vals_tools.after_training --train train_xxx --base-eval eval_xxx --steps 20 --dry-run

For each chosen checkpoint (default: every checkpoint the run saved), once the run has finished:

1. deploy it as `<prefix>-<run>-step<N>` (1 GPU each; `vals_tools.deploy` waits until it answers with a tool call);
2. play a short local game against scripted bots under the event rules (`harness/play_local.py`) and count failed
   decisions: a checkpoint whose decisions mostly fail is not evaluated, since the arena would pause it;
3. run it on the eval dataset through your harness (the harness and connector must be up), as the base eval did;
4. when every eval (and the base eval) has finished, score each checkpoint paired by deal against base.

The summary is printed and saved to data/launches/after_<run>.json. Deployments are left running (each holds a GPU):
delete the ones you won't enter with `python -m vals_tools.deploy --name <name> --delete`. Needs AC2_API_KEY for the
deploy probe and the local game.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ac2.sdk import Client

from vals_tools.eval import eval_config
from vals_tools.score import score_rows, summarize

ROOT = Path(__file__).resolve().parents[1]
DONE = ("succeeded", "failed", "stopped", "cancelled", "canceled", "error")
CHECK = re.compile(r"(\d+) decisions by \S+, (\d+) failed")
MAX_FAILED_SHARE = 0.5  # more than this in the local game: don't spend an eval on it (the arena pauses such entries)


def deployment_name(prefix: str, train_id: str, step: int) -> str:
    return f"{prefix}-{train_id.removeprefix('train_')[-6:]}-step{step}"


def parse_check(output: str) -> tuple[int, int] | None:
    """(decisions, failed) from play_local's summary line, or None if it never printed one."""
    found = CHECK.findall(output)
    return (int(found[-1][0]), int(found[-1][1])) if found else None


def status(value: Any) -> str:
    return str(getattr(value, "value", value) or "?").lower()


def wait(label: str, poll: callable, every: int = 60) -> str:
    """Poll until a finished state; print each change."""
    last = None
    while True:
        state = status(poll())
        if state != last:
            print(f"{label}: {state}", flush=True)
            last = state
        if any(word in state for word in DONE):
            return state
        time.sleep(every)


def run(args: argparse.Namespace) -> dict[str, Any]:
    client = Client(project=args.project)
    state = wait(f"training {args.train}", lambda: client.train.get(args.train).status, args.poll)
    if "succeeded" not in state:
        raise SystemExit(f"{args.train} ended {state}: nothing to evaluate")
    checkpoints = sorted(client.train.checkpoints(args.train), key=lambda c: c.iter_num)
    if args.steps:
        checkpoints = [c for c in checkpoints if c.iter_num in args.steps]
    if not checkpoints:
        raise SystemExit(f"no checkpoints of {args.train} match --steps {args.steps}")
    deployed = {getattr(d, "name", None) for d in client.inference.list()}
    results: dict[str, Any] = {"train": args.train, "base_eval": args.base_eval, "dataset": args.dataset, "checkpoints": []}
    for ckpt in checkpoints:
        name = deployment_name(args.prefix, args.train, ckpt.iter_num)
        model = f"{args.project}/{name}"
        entry: dict[str, Any] = {"step": ckpt.iter_num, "checkpoint": ckpt.id, "deployment": model}
        results["checkpoints"].append(entry)
        print(f"\n=== step {ckpt.iter_num}: {ckpt.id} as {model}", flush=True)
        if args.dry_run:
            continue
        if name not in deployed:
            deploy = [sys.executable, "-m", "vals_tools.deploy", "--project", args.project, "--name", name,
                      "--checkpoint", ckpt.id]
            if subprocess.run(deploy, cwd=ROOT).returncode != 0:
                entry["error"] = "deploy failed"
                continue
        game = ["uv", "run", "python", "play_local.py", "--model", model, "--hands", str(args.check_hands),
                "--opponents", "bots", "--out", "runs/checks"]
        played = subprocess.run(game, cwd=ROOT / "harness", capture_output=True, text=True)
        print(played.stdout[-600:], flush=True)
        check = parse_check(played.stdout)
        entry["tool_check"] = {"decisions": check[0], "failed": check[1]} if check else {"error": played.stderr[-300:]}
        if not check or not check[0] or check[1] / check[0] > MAX_FAILED_SHARE:
            entry["error"] = "most decisions failed in the local game: not evaluated"
            continue
        launched = asyncio.run(client.eval.run(eval_config(args.dataset, deployment=model, max_parallel=args.max_parallel)))
        entry["eval"] = getattr(launched, "eval_id", str(launched))
        print(f"eval {entry['eval']} launched on {args.dataset}", flush=True)
    if args.dry_run:
        return results
    for entry in [e for e in results["checkpoints"] if "eval" in e]:
        entry["eval_status"] = wait(f"eval {entry['eval']} (step {entry['step']})",
                                    lambda e=entry: client.eval.get(e["eval"]).status, args.poll)
    wait(f"base eval {args.base_eval}", lambda: client.eval.get(args.base_eval).status, args.poll)
    base = score_rows(client, eval_id=args.base_eval)
    results["base"] = summarize(base)
    for entry in [e for e in results["checkpoints"] if "eval" in e]:
        entry["score"] = summarize(score_rows(client, eval_id=entry["eval"]), base)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default="vals-hackathon")
    parser.add_argument("--train", required=True, help="the training run (train_...)")
    parser.add_argument("--base-eval", required=True, help="base's eval on the same dataset (eval_...)")
    parser.add_argument("--dataset", default="pokerbots-eval", help="the eval dataset (default: the held-out deals)")
    parser.add_argument("--steps", type=lambda s: {int(x) for x in s.split(",")}, help="e.g. 10,20 (default: all)")
    parser.add_argument("--prefix", default="ckpt", help="deployment name prefix")
    parser.add_argument("--check-hands", type=int, default=6, help="hands in the local tool-call check")
    parser.add_argument("--max-parallel", type=int, default=32, help="concurrent hands per eval on your harness")
    parser.add_argument("--poll", type=int, default=60, help="seconds between status checks")
    parser.add_argument("--dry-run", action="store_true", help="list what would be deployed, deploy nothing")
    args = parser.parse_args()
    results = run(args)
    out = ROOT / "data" / "launches" / f"after_{args.train}.json"
    if not args.dry_run:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=1))
    print("\n" + json.dumps(results, indent=1))
    if not args.dry_run:
        print(f"saved to {out}. Deployments stay up (1 GPU each); delete unused ones with "
              f"`python -m vals_tools.deploy --name <name> --delete`.")


if __name__ == "__main__":
    main()
