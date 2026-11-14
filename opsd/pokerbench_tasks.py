"""Build AC2 training tasks from PokerBench spots converted by `harness/pokerbench_convert.py`.

The converter replays each PokerBench spot through the harness, so every task's prompt is exactly what the event
table shows the trained seat at that decision (system prompt, scratchpad, the hand so far, the seat's earlier calls
and their results), with the solver's play as a legal `submit_action` call. Two ways to train on it:

- `--mode hinted` (recommended): a `HintedTask` per spot whose input is the prompt only, with one hint selection
  that appends the solver's play (`--hint-template`) to the system message, as `build_dataset.py` does. Train it with
  `--method opsd --opsd-rollout-mode one_step`: the student writes its own reply (thinking included, and its own
  `submit_action` call) and is pulled toward the teacher, the same model reading the solver's play. Nothing is
  written for the model, so its format and its thinking stay its own.
- `--mode sft`: a plain `Task` per spot whose input is the prompt followed by the solver's play as the seat's
  `submit_action` call, for `--method sft` (AC2's SFT trains on the conversations in the task inputs). That target
  has no reasoning: training on it may teach a thinking model to answer without thinking, and earlier hero turns in
  postflop spots are bare calls too. Check a checkpoint with `harness/play_local.py` before trusting it.

Every task lists the seat's tools in `env_params.tools` (from `tools.json`; training rejects a transcript with tool
calls unless they're listed), and the script checks that the converter captured those same tools.

    cd harness && uv run python pokerbench_convert.py <PokerBench train CSVs> --out ../runs/pokerbench.jsonl
    uv run python opsd/pokerbench_tasks.py --input runs/pokerbench.jsonl --mode hinted            # dry run
    uv run python opsd/pokerbench_tasks.py --input runs/pokerbench.jsonl --mode hinted --dataset pokerbots-pb-hinted

Without `--dataset` (or with `--dry-run`) it prints counts and the first task and uploads nothing. Convert
PokerBench's train files for training; its test files are its held-out split.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from ac2.runtime import FunctionCall, FunctionCallOutput, HintedTask, HintMutation, HintSelection, Item, Message, Task

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hint import SEPARATOR  # noqa: E402

SUBMIT = "submit_action"
RECORDED_TOOLS = json.loads((Path(__file__).parent / "tools.json").read_text())
HINT_TEMPLATE = (
    "Reference play for this exact spot, from a game-theory solver: {solver_text}. "
    "Amounts are this table's chips. Reason through the spot, then submit the strongest play."
)
MODES = ("hinted", "sft")


def to_items(messages: list[dict[str, Any]]) -> list[Item]:
    """Chat messages (as the harness sends them) as AC2 items, the way the harness maps them for the Responses API."""
    items: list[Item] = []
    for message in messages:
        if message["role"] == "tool":
            items.append(FunctionCallOutput(call_id=message["tool_call_id"], output=message["content"]))
            continue
        if message.get("content"):
            items.append(Message(role=message["role"], content=message["content"]))
        for call in message.get("tool_calls") or []:
            function = call["function"]
            items.append(FunctionCall(call_id=call["id"], name=function["name"], arguments=function["arguments"]))
    return items


def env_params(record: dict[str, Any]) -> dict[str, Any]:
    if record["tools"] != RECORDED_TOOLS:
        raise SystemExit(f"tools in {record['source']} differ from opsd/tools.json: reconvert with this repo's harness")
    keys = ("source", "street", "hero_position", "label", "solver_text")
    return {"tools": RECORDED_TOOLS, "pokerbench": {k: record[k] for k in keys}}


def hinted_task(record: dict[str, Any], template: str = HINT_TEMPLATE) -> HintedTask:
    prompt = to_items(record["messages"])
    return HintedTask(
        input=prompt,
        env_params=env_params(record),
        opsd_hint_selections=[
            HintSelection(
                hint=template.format(solver_text=record["solver_text"]),
                rollout_prefix_len=len(prompt),
                # The system message is item 0: append the hint to it, for the teacher only.
                mutation=HintMutation(item_boundary=1, action="append_to_previous_item", separator=SEPARATOR),
            )
        ],
    )


def sft_task(record: dict[str, Any]) -> Task:
    call = FunctionCall(call_id="call_solver", name=SUBMIT, arguments=json.dumps(record["label"]))
    return Task(input=[*to_items(record["messages"]), call], env_params=env_params(record))


def build(records: list[dict[str, Any]], mode: str, template: str = HINT_TEMPLATE) -> list[Task]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    return [hinted_task(r, template) if mode == "hinted" else sft_task(r) for r in records]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default="vals-hackathon")
    parser.add_argument("--input", type=Path, required=True, help="JSONL from harness/pokerbench_convert.py")
    parser.add_argument("--mode", choices=MODES, default="hinted")
    parser.add_argument("--hint-template", default=HINT_TEMPLATE, help="hinted mode; {solver_text} is the solver's play")
    parser.add_argument("--streets", help="keep only these streets, e.g. preflop,flop")
    parser.add_argument("--limit", type=int, help="at most this many tasks, after shuffling")
    parser.add_argument("--seed", type=int, default=0, help="shuffle seed")
    parser.add_argument("--dataset", help="upload to this dataset; without it nothing is uploaded")
    parser.add_argument("--dry-run", action="store_true", help="print counts and the first task, upload nothing")
    args = parser.parse_args()

    records = [json.loads(line) for line in args.input.open(encoding="utf-8") if line.strip()]
    if args.streets:
        keep = set(args.streets.split(","))
        records = [r for r in records if r["street"] in keep]
    random.Random(args.seed).shuffle(records)
    records = records[: args.limit] if args.limit else records
    tasks = build(records, args.mode, args.hint_template)
    counts = Counter(f"{r['street']}:{r['label']['action']}" for r in records)
    print(json.dumps({"mode": args.mode, "tasks": len(tasks), "by_street_and_action": dict(sorted(counts.items()))}))
    if args.dry_run or not args.dataset or not tasks:
        if tasks:
            print(tasks[0].model_dump_json(indent=1, exclude_none=True))
        print("dry run: nothing uploaded" + ("" if args.dataset else " (pass --dataset NAME to upload)"))
        return

    from ac2.sdk import Client

    client = Client(project=args.project)
    required = ["hinted"] if args.mode == "hinted" else ["traces"]
    client.datasets.create(args.dataset, required_types=required, exist_ok=True)
    for start in range(0, len(tasks), 500):
        client.datasets.add_tasks(args.dataset, tasks=tasks[start : start + 500])
    print(f"uploaded {len(tasks)} {args.mode} tasks to {args.dataset}")


if __name__ == "__main__":
    main()
