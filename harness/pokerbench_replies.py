"""Collect a deployment's own replies to converted PokerBench spots, judged against the solver's play.

Each spot from `pokerbench_convert.py` is the exact conversation the event table shows the trained seat. This sends
it to a deployment (default: the base model) as the harness would, with the event's tools, thinking on and the
5,000-token cap, and records the reply: its reasoning, text and tool calls. Each reply is judged:

- `clean`: exactly one `submit_action` call, nothing else called, not cut at the cap, and legal under the event's
  strict rules (an unclean reply would be played as check or fold);
- `agrees`: the applied action matches the solver's (bet and raise amounts within 25%, or 2 chips).

Clean replies are what offline OPSD trains on (`opsd/pokerbench_tasks.py --mode offline`: the reply, with the
solver's play as the teacher's hint). Clean replies that agree are rejection-sampled SFT data (`--mode rft`): the
model's own reasoning that reached the solver's play. The agreement rate also shows how far base is from the solver.

    AC2_API_KEY=... uv run python pokerbench_replies.py ../data/pokerbench/preflop_train.jsonl \
        --out ../data/pokerbench/replies.jsonl --count 1500 --concurrency 8

Spots are sampled at random per input file (`--count`, `--seed`); `--samples K` asks for K replies per spot. The
output is appended line by line and a rerun skips replies already there, so an interrupted run resumes. A request
that fails outright is not written (a rerun retries it). Mind the load: the deployment may also be your arena entry.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx

from poker_harness.completions import GATEWAY_URL, relay_completion
from poker_harness.engine import Decision, LegalActions
from poker_harness.engine.table import resolve
from poker_harness.prompts import SUBMIT_ACTION, parse_tool_call
from poker_harness.seat import Completion, invalid_action
from pokerbench_convert import PARAMS

AMOUNT_TOLERANCE = 0.25  # a bet or raise agrees within 25% of the solver's amount (or 2 chips)


def reply_key(record: dict[str, Any], sample: int) -> str:
    return f"{json.dumps(record['source'], sort_keys=True)}#{sample}"


def judge(record: dict[str, Any], reply: Completion) -> dict[str, Any]:
    """Whether the reply is clean under the event's strict rules, what it plays, and whether that is the solver's."""
    legal = LegalActions(**record["legal"])
    submits = [c for c in reply.tool_calls if c["function"]["name"] == SUBMIT_ACTION]
    error = None
    if reply.truncated:
        error = "reply hit the token cap"
    elif len(submits) != 1 or len(reply.tool_calls) != 1:
        error = f"{len(submits)} submit_action calls and {len(reply.tool_calls) - len(submits)} other calls"
    if error:
        return {"clean": False, "error": error, "decision": None, "agrees": False}
    decision, error = parse_tool_call(submits[0]["function"]["arguments"], talk=True)
    error = error or invalid_action(decision, SimpleNamespace(legal=legal))
    if error:
        return {"clean": False, "error": error, "decision": None, "agrees": False}
    mine, _ = resolve(decision, legal)
    label = record["label"]
    solver, _ = resolve(Decision(label["action"], label.get("amount")), legal)
    agrees = mine.type == solver.type
    if agrees and solver.amount is not None:
        agrees = abs(mine.amount - solver.amount) <= max(2, AMOUNT_TOLERANCE * solver.amount)
    played = {"action": mine.type, **({"amount": mine.amount} if mine.amount is not None else {})}
    return {"clean": True, "error": None, "decision": played, "agrees": agrees}


def reply_record(record: dict[str, Any], sample: int, model: str, reply: Completion) -> dict[str, Any]:
    return {
        **record,
        "sample": sample,
        "model": model,
        "reply": {
            "reasoning": reply.reasoning,
            "text": reply.text,
            "tool_calls": reply.tool_calls,
            "out_tokens": reply.out_tokens,
            "reasoning_tokens": reply.reasoning_tokens,
            "truncated": reply.truncated,
        },
        **judge(record, reply),
    }


def sample_records(paths: list[Path], count: int | None, seed: int, streets: set[str] | None) -> list[dict]:
    """Up to `count` spots per file, at random, from the wanted streets."""
    chosen = []
    for path in paths:
        records = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        records = [r for r in records if not streets or r["street"] in streets]
        random.Random(seed).shuffle(records)
        chosen += records[:count] if count else records
    return chosen


async def collect(
    records: list[dict[str, Any]],
    out: Path,
    model: str,
    key: str,
    samples: int = 1,
    concurrency: int = 8,
    base_url: str = GATEWAY_URL,
) -> dict[str, int]:
    done = set()
    if out.exists():
        done = {reply_key(r, r["sample"]) for r in map(json.loads, out.open(encoding="utf-8")) if r}
    todo = [(r, k) for r in records for k in range(samples) if reply_key(r, k) not in done]
    stats = {"skipped": len(records) * samples - len(todo), "written": 0, "clean": 0, "agrees": 0, "failed": 0}
    print(f"{len(todo)} replies to collect ({stats['skipped']} already in {out})", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    gate, started = asyncio.Semaphore(concurrency), time.monotonic()
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
        complete = relay_completion(client, base_url, key, PARAMS | {"model_label": model})
        with out.open("a", encoding="utf-8") as sink:

            async def one(record: dict[str, Any], sample: int) -> None:
                async with gate:
                    try:
                        reply = await complete(record["messages"])
                    except Exception as exc:  # noqa: BLE001 - not written, so a rerun retries it
                        stats["failed"] += 1
                        print(f"  request failed for {reply_key(record, sample)}: {type(exc).__name__}", flush=True)
                        return
                row = reply_record(record, sample, model, reply)
                sink.write(json.dumps(row) + "\n")
                sink.flush()
                stats["written"] += 1
                stats["clean"] += row["clean"]
                stats["agrees"] += row["agrees"]
                if stats["written"] % 50 == 0:
                    minutes = (time.monotonic() - started) / 60
                    print(f"  {stats['written']}/{len(todo)} in {minutes:.1f} min; {stats['clean']} clean, "
                          f"{stats['agrees']} agree with the solver", flush=True)

            await asyncio.gather(*(one(r, k) for r, k in todo))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("jsonl", nargs="+", type=Path, help="spots from pokerbench_convert.py")
    parser.add_argument("--out", type=Path, required=True, help="JSONL of replies (appended; reruns resume)")
    parser.add_argument("--model", default="vals-hackathon/base", help="the deployment to ask, <project>/<name>")
    parser.add_argument("--count", type=int, help="spots per input file, sampled at random (default: all)")
    parser.add_argument("--streets", help="only these streets, e.g. preflop,flop")
    parser.add_argument("--samples", type=int, default=1, help="replies per spot (more gives rejection sampling more)")
    parser.add_argument("--concurrency", type=int, default=8, help="requests at once (the arena may share this deployment)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    key = os.environ.get("AC2_API_KEY")
    if not key:
        raise SystemExit("export AC2_API_KEY first")
    streets = set(args.streets.split(",")) if args.streets else None
    records = sample_records(args.jsonl, args.count, args.seed, streets)
    stats = asyncio.run(collect(records, args.out, args.model, key, args.samples, args.concurrency))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
