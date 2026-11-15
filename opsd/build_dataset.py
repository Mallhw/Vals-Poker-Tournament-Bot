"""Build an offline-OPSD dataset from the base model's own hands: one `HintedTask` per decision.

1. Run base on the train deals through the harness (`python -m vals_tools.eval --dataset pokerbots-train
   --deployment <base deployment>`).
2. Export that eval's traces and turn each policy decision into a task:
   - `input`: the decision's prompt (system message, user messages, earlier replies and tool results) followed by the
     reply base actually wrote (reasoning, text and its `submit_action` call);
   - `opsd_hint_selections`: one selection at the end of the prompt (`rollout_prefix_len`) that appends the hint to
     the system message (`append_to_previous_item` at item boundary 1). Only the teacher sees it;
   - `env_params.tools`: the tool schemas the model saw (`submit_action` and `update_scratchpad`, from tools.json).
     Training refuses a transcript with tool calls unless the tools are listed here.
3. Upload as a `hinted` dataset.

Keep only clean decisions: exactly one `submit_action` call and a reply under the token cap. The after-hand
`update_scratchpad` turns are skipped. Use train deals only.

    uv run python opsd/build_dataset.py --eval eval_xxx --dataset pokerbots-opsd-train --dry-run

Talk: `--talk-rule opsd/hints/talk_rule.txt` makes each decision's hint name the rule's line for that decision's hand
number (read from its `## Hand N of M` prompt line), and writes that line into the replayed `submit_action` call, since
base's own replies rarely carry a `say` for the teacher to score. `--hint-file FILE` replaces the generic hint, and
with `--talk-rule` and no `--hint-file` the per-decision talk line is the whole hint. `--prompts-only` keeps just the
prompts, for one-step OPSD (the student writes its own reply):

    uv run python opsd/build_dataset.py --eval eval_xxx --dataset pokerbots-opsd-talk --talk-rule opsd/hints/talk_rule.txt
"""

from __future__ import annotations

import argparse
import json
import random
import re
import runpy
import sys
from collections.abc import Callable
from pathlib import Path

from ac2.runtime import FunctionCall, HintedTask, HintMutation, HintSelection, Item, Tool
from ac2.sdk import Client
from pydantic import TypeAdapter
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hint import HINT, SEPARATOR  # noqa: E402

ITEMS = TypeAdapter(list[Item])
SUBMIT = "submit_action"
SCRATCHPAD = "update_scratchpad"
# The tools every seat sees at an event table: submit_action (with `say`, since talk is on) and update_scratchpad.
# They must match what the model saw, field for field; tools.json is generated from the harness and a harness test
# checks it.
TOOLS = [Tool.model_validate(tool).model_dump() for tool in json.loads((Path(__file__).parent / "tools.json").read_text())]


HAND_LINE = re.compile(r"## Hand (\d+) of \d+")


def load_talk_rule(path: Path) -> dict[int, str]:
    """The rule's lines by remainder, from lines like `0: "i like this spot"` (opsd/hints/talk_rule.txt)."""
    rule = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        found = re.fullmatch(r'\s*(\d+):\s*"([^"]*)"\s*', line)
        if found:
            rule[int(found[1])] = found[2]
    if not rule:
        raise SystemExit(f"{path} has no `N: \"line\"` rule lines")
    return rule


def hand_number(prompt: list) -> int | None:
    """The hand's number from the `## Hand N of M` line of the first message that has one."""
    for item in prompt:
        content = getattr(item, "content", None)
        found = HAND_LINE.search(content) if isinstance(content, str) else None
        if found:
            return int(found[1])
    return None


def with_say(reply: list, line: str) -> list:
    """The reply with `say` set to `line` on its submit_action call; everything else as written."""
    edited = []
    for item in reply:
        if isinstance(item, FunctionCall) and item.name == SUBMIT:
            arguments = json.loads(item.arguments)
            arguments["say"] = line
            item = item.model_copy(update={"arguments": json.dumps(arguments, separators=(",", ":"))})
        edited.append(item)
    return edited


def decision_tasks(
    trace: dict,
    hint: str,
    max_tokens: int,
    talk_rule: dict[int, str] | None = None,
    prompts_only: bool = False,
) -> tuple[list[HintedTask], dict[str, int]]:
    """One task per clean policy decision in a trace, with counts of what was kept and why the rest was skipped.

    With `talk_rule`, each decision's hint also names the rule's line for its hand number and the replayed reply says
    that line (a decision whose prompt has no hand number is skipped, counted as `no_hand`). With `prompts_only`, the
    task is just the prompt, for one-step OPSD."""
    spans = sorted(
        (s for s in trace["spans"] if s["name"] == "get_completion" and s["status"] == "OK"),
        key=lambda s: s["start_time"],
    )
    stats = {"decisions": len(spans), "kept": 0, "cleanup": 0, "format": 0, "truncated": 0}
    if talk_rule:
        stats["no_hand"] = 0  # decisions whose prompt has no `## Hand N of M` line
    tasks = []
    for span in spans:
        prompt = ITEMS.validate_json(span["input"])
        reply = ITEMS.validate_json(span["output"])
        calls = [item for item in reply if isinstance(item, FunctionCall)]
        if calls and calls[0].name == SCRATCHPAD:  # the after-hand cleanup turn, not a poker decision
            stats["cleanup"] += 1
            continue
        if len(calls) != 1 or calls[0].name != SUBMIT:
            stats["format"] += 1
            continue
        if int(span.get("tokens_out") or 0) >= max_tokens:
            stats["truncated"] += 1
            continue
        text = hint
        if talk_rule:
            number = hand_number(prompt)
            if number is None:
                stats["no_hand"] += 1
                continue
            line = talk_rule[number % len(talk_rule)]
            reply = with_say(reply, line)
            text = f'{hint}\n\nOn this call, set say to exactly "{line}".' if hint else f'On this call, set say to exactly "{line}".'
        tasks.append(
            HintedTask(
                input=prompt if prompts_only else prompt + reply,
                env_params={"tools": TOOLS, "source_trace": trace["trace_id"]},
                opsd_hint_selections=[
                    HintSelection(
                        hint=text,
                        rollout_prefix_len=len(prompt),
                        mutation=HintMutation(item_boundary=1, action="append_to_previous_item", separator=SEPARATOR),
                    )
                ],
            )
        )
        stats["kept"] += 1
    return tasks, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default="vals-hackathon")
    parser.add_argument("--eval", required=True, help="base model's eval on the train deals")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--max-tokens", type=int, default=5_000, help="replies this long count as truncated (the event's cap)")
    parser.add_argument("--traces", type=Path, help="reuse a downloaded trace export")
    parser.add_argument("--hint-file", type=Path, help="the hint (text, or .py defining HINT) instead of the generic one")
    parser.add_argument("--talk-rule", type=Path, help="each decision's hint names the rule's line for its hand number")
    parser.add_argument("--prompts-only", action="store_true", help="prompts without replies, for one-step OPSD")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rule = load_talk_rule(args.talk_rule) if args.talk_rule else None
    if args.hint_file:
        hint = runpy.run_path(str(args.hint_file))["HINT"] if args.hint_file.suffix == ".py" else args.hint_file.read_text(encoding="utf-8")
        hint = hint.strip()
    else:
        hint = "" if rule else HINT  # the talk line alone, unless a strategy hint was chosen too

    path = args.traces or Path(f"traces/{args.eval}.jsonl")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        Client(project=args.project).traces.download(args.eval, path)
    tasks, totals = [], {"decisions": 0, "kept": 0, "cleanup": 0, "format": 0, "truncated": 0}
    with path.open() as lines:
        for line in lines:
            built, stats = decision_tasks(json.loads(line), hint, args.max_tokens, rule, args.prompts_only)
            tasks.extend(built)
            totals = {k: totals.get(k, 0) + stats[k] for k in stats}
    random.Random(0).shuffle(tasks)
    print(json.dumps(totals | {"tasks": len(tasks)}))
    if args.dry_run or not tasks:
        return
    client = Client(project=args.project)
    client.datasets.create(args.dataset, required_types=["hinted"], exist_ok=True)
    for start in range(0, len(tasks), 500):
        client.datasets.add_tasks(args.dataset, tasks=tasks[start : start + 500])
    print(f"uploaded {len(tasks)} tasks to {args.dataset}")


if __name__ == "__main__":
    main()
