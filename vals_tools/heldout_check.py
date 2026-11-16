"""Score a training run's checkpoints on held-out PokerBench spots: does the model now play the solver's move?

    uv run python -m vals_tools.heldout_check --train train_xxx
    uv run python -m vals_tools.heldout_check --summarize data/pokerbench/heldout_base.jsonl data/pokerbench/heldout_x.jsonl

Once the run has finished, for each checkpoint (default: every one it saved): deploy it (1 GPU), ask it the held-out
spots with `harness/pokerbench_replies.py` (no hint, no opponents, so no harness or connector is needed), and delete
the deployment again unless `--keep`. With `--keep-best`, the checkpoint with the highest agreement so far stays
deployed (at most two deployments at once) and the rest are deleted, so the winner is left serving, ready to submit
as the arena entry. Then print solver agreement per checkpoint next to base and the hinted teacher, overall and by
the solver's action, on exactly the same spots (same `--count` and `--seed` as the base file).

`--follow` scores each checkpoint as soon as it is saved, while the run is still training, deleting each deployment
right after (training leaves little GPU room); with `--keep-best` the winner is deployed again once training ends.
A checkpoint that could not be deployed while training ran is tried again once the run ends and its GPUs come free.
Add `--hold-best` when there is room for two deployments during training: the best so far then stays deployed the
whole time (the arena entry), next to the one being scored.

    uv run python -m vals_tools.heldout_check --train train_xxx --follow --keep-best [--hold-best]

Every new best is written to data/launches/heldout_<train>_best.json (step, agreement, deployment name and id), so
whoever submits the arena entry can watch that one file. A restarted watcher reuses finished reply files and
deployments that are still up, instead of asking again.

The held-out spots are PokerBench's test split (data/pokerbench/{preflop,postflop}_test.jsonl), which no training
dataset here draws from. Needs AC2_API_KEY for the deploy probe and the replies.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "pokerbench"
ACTIONS = ("fold", "check", "call", "bet/raise")


def action(kind: str) -> str:
    return "bet/raise" if kind in ("bet", "raise") else kind


def load(path: Path) -> dict[str, dict[str, Any]]:
    return {json.dumps(r["source"], sort_keys=True): r for r in map(json.loads, path.open(encoding="utf-8"))}


def summarize(files: dict[str, Path]) -> dict[str, dict[str, Any]]:
    """Agreement with the solver per file, on the spots every file answered; by the solver's action too."""
    loaded = {label: load(path) for label, path in files.items() if path.exists()}
    keys = set.intersection(*(set(v) for v in loaded.values())) if loaded else set()
    out = {}
    for label, rows in loaded.items():
        rs = [rows[k] for k in sorted(keys)]
        by = {a: [r for r in rs if action(r["label"]["action"]) == a] for a in ACTIONS}
        played = Counter(action(r["decision"]["action"]) for r in rs if r["clean"])
        tokens = sorted(r["reply"]["out_tokens"] or 0 for r in rs)
        share = lambda part, whole: round(part / whole, 3) if whole else None  # noqa: E731
        out[label] = {
            "spots": len(rs),
            "agree": share(sum(r["agrees"] for r in rs), len(rs)),
            **{f"agree_when_{a}": share(sum(r["agrees"] for r in by[a]), len(by[a])) for a in ACTIONS},
            "clean": share(sum(r["clean"] for r in rs), len(rs)),
            **{f"plays_{a}": share(played[a], sum(played.values())) for a in ACTIONS},
            "tokens_median": tokens[len(tokens) // 2] if tokens else None,
        }
    return out


def table(summary: dict[str, dict[str, Any]]) -> str:
    cols = ["agree", *(f"agree_when_{a}" for a in ACTIONS), "clean", "plays_bet/raise", "tokens_median"]
    head = ["", "agree", "| solver folds", "checks", "calls", "bets/raises", "| clean", "it raises", "tokens"]
    rows = [[label] + [f"{s[c]:.0%}" if isinstance(s[c], float) else str(s[c]) for c in cols]
            for label, s in summary.items()]
    widths = [max(len(str(x)) for x in col) for col in zip(head, *rows)]
    return "\n".join("  ".join(str(x).rjust(w) for x, w in zip(row, widths)) for row in [head, *rows])


def agreement(path: Path) -> float:
    """Share of a reply file's spots that agree with the solver (0 for a missing or empty file)."""
    rows = list(load(path).values()) if path.exists() else []
    return sum(r["agrees"] for r in rows) / len(rows) if rows else 0.0


def keep_best(best: tuple[float, str] | None, score: float, name: str) -> tuple[tuple[float, str], str | None]:
    """The new best (agreement, deployment) and which deployment to delete now: the loser of the two."""
    if best is None or score > best[0]:
        return (score, name), (best[1] if best else None)
    return best, name


def wait_for(client, train_id: str, poll: int) -> str:
    last = None
    while True:
        state = str(client.train.get(train_id).status).lower()
        if state != last:
            print(f"training {train_id}: {state}", flush=True)
            last = state
        if any(w in state for w in ("succeeded", "failed", "stopped", "cancel", "error")):
            return state
        time.sleep(poll)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default="vals-hackathon")
    parser.add_argument("--train", help="the training run whose checkpoints to score")
    parser.add_argument("--steps", type=lambda s: {int(x) for x in s.split(",")}, help="AC2 iterations, e.g. 9,19")
    parser.add_argument("--count", type=int, default=150, help="spots per test file (match the base file: 150)")
    parser.add_argument("--seed", type=int, default=7, help="spot sample seed (match the base file: 7)")
    parser.add_argument("--keep", action="store_true", help="leave each checkpoint deployed after scoring")
    parser.add_argument("--keep-best", action="store_true", help="leave only the best-scoring checkpoint deployed")
    parser.add_argument("--follow", action="store_true",
                        help="score each checkpoint as soon as it lands, while training runs (one deployment at a time)")
    parser.add_argument("--hold-best", action="store_true",
                        help="with --follow --keep-best: keep the best deployed during training too (needs 2 GPUs)")
    parser.add_argument("--poll", type=int, default=120)
    parser.add_argument("--summarize", nargs="+", type=Path, help="only print the table for these reply files")
    args = parser.parse_args()
    reference = {"base": DATA / "heldout_base.jsonl", "base + hint (teacher)": DATA / "heldout_teacher_opsd.jsonl"}
    if args.summarize:
        print(table(summarize({p.stem.removeprefix("heldout_"): p for p in args.summarize})))
        return
    if not args.train:
        parser.error("pass --train, or --summarize FILES")

    from ac2.sdk import Client

    client = Client(project=args.project)
    if not args.follow:
        state = wait_for(client, args.train, args.poll)
        if "succeeded" not in state:
            raise SystemExit(f"{args.train} ended {state}: nothing to score")
    files = dict(reference)
    best: tuple[float, str] | None = None
    scores: dict[int, tuple[float, str]] = {}  # iteration -> (agreement, checkpoint id)
    deploy_name = lambda it: f"ho-{args.train.removeprefix('train_')[-6:]}-step{it}"  # noqa: E731
    hold = args.keep_best and (args.hold_best or not args.follow)  # the best stays deployed while scoring goes on
    best_file = ROOT / "data" / "launches" / f"heldout_{args.train}_best.json"

    def patient(call, what: str):
        """An AC2 API call that survives network blips (DNS, timeouts): retried every minute for half an hour."""
        for attempt in range(30):
            try:
                return call()
            except Exception as exc:  # noqa: BLE001 - any API or network error; the watcher must keep going
                print(f"{what}: {type(exc).__name__}: {exc}; retrying in 60 s ({attempt + 1}/30)", flush=True)
                time.sleep(60)
        return call()

    def live(name: str) -> str | None:
        """The id of a deployment with this name that is up (or coming up), if there is one."""
        for d in patient(client.inference.list, "listing deployments"):
            if getattr(d, "name", None) == name and str(d.state) not in ("failed", "stopped", "deleted", "succeeded"):
                return d.id
        return None

    def delete(name: str) -> None:
        subprocess.run([sys.executable, "-m", "vals_tools.deploy", "--project", args.project, "--name", name, "--delete"], cwd=ROOT)

    def deploy(ckpt_id: str, name: str, tries: int = 1) -> bool:
        cmd = [sys.executable, "-m", "vals_tools.deploy", "--project", args.project, "--name", name, "--checkpoint", ckpt_id]
        for attempt in range(tries):
            if subprocess.run(cmd, cwd=ROOT).returncode == 0:
                return True
            delete(name)  # a half-made deployment must not hold a GPU, or be found by name later
            if attempt + 1 < tries:
                print(f"deploy failed; trying again in 5 min ({attempt + 2}/{tries})", flush=True)
                time.sleep(300)
        return False

    def record_best(iteration: int, ckpt_id: str) -> None:
        top, name = best
        entry = {"train": args.train, "step": iteration + 1, "checkpoint": ckpt_id, "agree": round(top, 4),
                 "deployment": f"{args.project}/{name}", "deployment_id": live(name)}
        best_file.parent.mkdir(parents=True, exist_ok=True)
        best_file.write_text(json.dumps(entry, indent=1))
        print(f"NEW BEST: {json.dumps(entry)}", flush=True)

    def score(ckpt, tries: int = 1) -> bool:
        """Score one checkpoint; False if it could not be deployed (the caller tries again once training ends)."""
        nonlocal best
        name = deploy_name(ckpt.iter_num)
        out = DATA / f"heldout_{args.train}_step{ckpt.iter_num}.jsonl"
        files[f"step {ckpt.iter_num + 1}"] = out
        print(f"\n=== step {ckpt.iter_num + 1}: {ckpt.id} as {args.project}/{name}", flush=True)
        finished = out.exists() and len(load(out)) >= len(load(reference["base"]))
        if finished:
            print(f"replies already in {out.name}: reused", flush=True)
        else:
            ready = False
            if existing := live(name):  # left by a watcher that was stopped mid-way
                from vals_tools.deploy import wait_until_serving
                try:
                    wait_until_serving(client, existing, f"{args.project}/{name}")
                    ready = True
                except SystemExit as exc:
                    print(f"{name} left over but broken ({exc}): deploying it again", flush=True)
                    delete(name)
            # holding the best, the next one may wait a while for its GPU (up to 2 h: retried every 5 min)
            if not ready and not deploy(ckpt.id, name, tries=24 if hold else tries):
                print("deploy failed: skipped", flush=True)
                return False
            ask = ["uv", "run", "python", "pokerbench_replies.py", str(DATA / "preflop_test.jsonl"),
                   str(DATA / "postflop_test.jsonl"), "--count", str(args.count), "--seed", str(args.seed),
                   "--model", f"{args.project}/{name}", "--concurrency", "8", "--out", str(out)]
            subprocess.run(ask, cwd=ROOT / "harness")
        scores[ckpt.iter_num] = (agreement(out), ckpt.id)
        if hold:
            best, loser = keep_best(best, scores[ckpt.iter_num][0], name)
            print(f"step {ckpt.iter_num + 1}: {scores[ckpt.iter_num][0]:.1%} agree; best so far {best[1]} at {best[0]:.1%}",
                  flush=True)
            if loser:
                delete(loser)
            if best[1] == name and live(name):  # a reused file's best may be gone: deployed after this pass
                record_best(ckpt.iter_num, ckpt.id)
        elif not args.keep:
            delete(name)
        print("\n" + table(summarize(files)), flush=True)  # the table so far, after every checkpoint
        return True

    def redeploy_best() -> None:
        """After a restart the best may be a reused reply file whose deployment is gone: bring it back."""
        if hold and best and not live(best[1]):
            iteration = next(it for it in scores if deploy_name(it) == best[1])
            if deploy(scores[iteration][1], best[1], tries=4):
                record_best(iteration, scores[iteration][1])

    scored: set[int] = set()
    skipped: set[int] = set()  # no GPU while training ran: tried again once it ends, when its GPUs come free
    while True:
        state = str(patient(lambda: client.train.get(args.train), "reading the run").status).lower()
        finished = any(w in state for w in ("succeeded", "failed", "stopped", "cancel", "error"))
        ckpts = patient(lambda: client.train.checkpoints(args.train), "listing checkpoints")
        for ckpt in sorted(ckpts, key=lambda c: c.iter_num):
            todo = ckpt.iter_num not in scored and (not args.steps or ckpt.iter_num in args.steps)
            if not todo or (ckpt.iter_num in skipped and not finished):
                continue
            if score(ckpt, tries=6 if finished else 1):  # once training ends: retried every 5 min for 30 min
                scored.add(ckpt.iter_num)
                skipped.discard(ckpt.iter_num)
            else:
                skipped.add(ckpt.iter_num)
        redeploy_best()
        if finished or not args.follow:
            break
        time.sleep(args.poll)
    if args.follow:
        print(f"training {args.train}: {state}", flush=True)
        if args.keep_best and scores and not hold:  # GPUs were short during training: deploy the winner again now
            iteration, (top, ckpt_id) = max(scores.items(), key=lambda item: (item[1][0], -item[0]))
            if deploy(ckpt_id, deploy_name(iteration)):
                best = (top, deploy_name(iteration))
                record_best(iteration, ckpt_id)
    summary = summarize(files)
    print("\n" + table(summary))
    if best:
        print(f"\nstill deployed: {args.project}/{best[1]} ({best[0]:.1%} agree), ready to submit as the arena entry; "
              f"delete it with `python -m vals_tools.deploy --name {best[1]} --delete`")
    target = ROOT / "data" / "launches" / f"heldout_{args.train}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=1))
    print(f"saved to {target}")


if __name__ == "__main__":
    main()
