"""Make the RL datasets: single 6-max hands at 100 BB on fresh seeds, one AC2 task per hand.

Writes <prefix>-train, -select (for picking checkpoints), -eval (held out), -smoke and -table-smoke. Every task
starts from the harness's "event" preset. Runs on these datasets name no grader, so the harness's sample
grader (harness/src/poker_harness/grading.py) scores each hand by its EV in big blinds.

    uv run python rl/make_dataset.py --dry-run
    uv run python rl/make_dataset.py
    uv run python rl/make_dataset.py --opponent-model vals-hackathon/opponent --opponent-api responses --opponent-effort medium

By default the opponents are `vals-hackathon/ac-vals-model`, the model every team org starts with.

`--enforced-cap N` makes only a training split, <prefix>-cap<N>-train, whose policy seat must finish every decision in
N output tokens (the harness's `enforced_max_completion_tokens`; the prompt still states the event's 5,000). Longer
replies score 0 like any format error, so GRPO learns a margin below the event's cap. Train on it with the plain
select split as the in-run eval, which keeps the event's rules:

    uv run python rl/make_dataset.py --enforced-cap 3000
    uv run python rl/train.py --prefix pokerbots-cap3000 --eval-dataset pokerbots-select --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ac2.sdk import Client

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ac2.runtime import Task  # noqa: E402

from vals_tools.tasks import SEATS, SPLITS, Opponents, datasets, hand_tasks  # noqa: E402


def capped_train(prefix: str, opponents: Opponents, cap: int, seats: int = SEATS) -> dict[str, list[Task]]:
    """The train split's deals with the policy seat held to `cap` output tokens per decision."""
    tasks = hand_tasks(*SPLITS["train"], opponents, seats)
    # Built anew, not copied, so each task's content hash covers the cap.
    capped = [Task(input=t.input, env_params={**(t.env_params or {}), "enforced_max_completion_tokens": cap}) for t in tasks]
    return {f"{prefix}-cap{cap}-train": capped}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default="vals-hackathon")
    parser.add_argument("--prefix", default="pokerbots")
    parser.add_argument(
        "--opponent-model",
        default="vals-hackathon/ac-vals-model",
        help="model the opponent seats call (default: the model every team org starts with)",
    )
    parser.add_argument(
        "--opponent-api", choices=["responses", "chat"], default="chat", help="responses for OpenAI reasoning models"
    )
    parser.add_argument("--opponent-effort", default="none", help="reasoning effort for the opponents, e.g. medium")
    parser.add_argument("--seats", type=int, default=SEATS)
    parser.add_argument(
        "--enforced-cap", type=int, help="only a train split whose policy seat must finish each decision in N tokens"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.enforced_cap is not None and not 0 < args.enforced_cap < 5000:
        parser.error("--enforced-cap must be between 1 and 4999 (the event's cap is 5000)")

    effort = None if args.opponent_effort == "none" else args.opponent_effort
    opponents = Opponents(args.opponent_model, args.opponent_api, effort)
    if args.enforced_cap:
        built = capped_train(args.prefix, opponents, args.enforced_cap, args.seats)
    else:
        built = datasets(args.prefix, opponents, args.seats)
    if args.dry_run:
        for name, tasks in built.items():
            print(name, len(tasks), tasks[0].env_params)
        return
    client = Client(project=args.project)
    for name, tasks in built.items():
        client.datasets.create(name, description=f"{len(tasks)} {args.seats}-max poker tasks.", exist_ok=True)
        for start in range(0, len(tasks), 500):
            client.datasets.add_tasks(name, tasks=tasks[start : start + 500])
        print(f"{name}: {len(tasks)} tasks")


if __name__ == "__main__":
    main()
