"""Make the self-play datasets: the train split's deals against league opponents, and a 200-deal yardstick.

    uv run python rl/make_league_dataset.py --dry-run
    uv run python rl/make_league_dataset.py

- `<prefix>-league-cap<N>-train`: the train split's deals with `opponent_league: true`, so the harness draws each
  opponent seat from its league file (`POKER_LEAGUE_FILE`, see harness/src/poker_harness/league.py) at every table,
  and with `enforced_max_completion_tokens` N (default 3000) for the policy seat, as `rl/make_dataset.py
  --enforced-cap` does. The league changes while the run trains (vals_tools/ladder.py), so one dataset serves every
  run, rollback and continuation.
  League opponents get 180 s a decision (`opponent_timeout_seconds`), so a busy frozen copy is waited for, not folded.
- `<prefix>-select200`: the first 200 deals of the select split, unchanged (the house model in every opponent seat).
  It's the yardstick each new checkpoint plays on its own deployment, paired by deal with base on the same deals.
- `<prefix>-eval1000`: the first 1,000 held-out deals, for the morning's verdict (3,000 take too long overnight).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ac2.runtime import Task
from ac2.sdk import Client

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vals_tools.tasks import SEATS, Opponents, split_tasks  # noqa: E402

HOUSE = "vals-hackathon/ac-vals-model"
YARDSTICK = 200
FINAL = 1000


def league_train(prefix: str, cap: int, opponents: Opponents, seats: int = SEATS) -> dict[str, list[Task]]:
    tasks = split_tasks("train", opponents, seats)
    # Built anew, not copied, so each task's content hash covers the league flag and the cap. `opponent_model` stays
    # as the harness's fallback for any seat a lineup leaves out.
    built = [
        Task(input=t.input, env_params={**(t.env_params or {}), "opponent_league": True,
                                        "enforced_max_completion_tokens": cap, "opponent_timeout_seconds": 180})
        for t in tasks
    ]
    return {f"{prefix}-league-cap{cap}-train": built}


def yardstick(prefix: str, opponents: Opponents, seats: int = SEATS) -> dict[str, list[Task]]:
    """The yardstick and the final verdict's deals, with opponents given 180 s a decision: at 60 s the house model
    timed out on most decisions under load, and an eval whose opponents only check or fold measures nothing."""
    slow = Opponents(opponents.model, opponents.api, opponents.reasoning_effort, timeout_seconds=180)
    return {
        f"{prefix}-select{YARDSTICK}": split_tasks("select", slow, seats, limit=YARDSTICK),
        f"{prefix}-eval{FINAL}": split_tasks("eval", slow, seats, limit=FINAL),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default="vals-hackathon")
    parser.add_argument("--prefix", default="pokerbots")
    parser.add_argument("--cap", type=int, default=3000, help="the policy seat's training cap (default 3000)")
    parser.add_argument("--opponent-model", default=HOUSE, help="the yardstick's opponents and the league fallback")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 0 < args.cap < 5000:
        parser.error("--cap must be between 1 and 4999 (the event's cap is 5000)")
    opponents = Opponents(args.opponent_model, "chat", None)  # as rl/make_dataset.py's defaults, so deals pair
    built = league_train(args.prefix, args.cap, opponents) | yardstick(args.prefix, opponents)
    if args.dry_run:
        for name, tasks in built.items():
            print(name, len(tasks), tasks[0].env_params)
        return
    client = Client(project=args.project)
    for name, tasks in built.items():
        client.datasets.create(name, description=f"{len(tasks)} {SEATS}-max poker tasks.", exist_ok=True)
        for start in range(0, len(tasks), 500):
            client.datasets.add_tasks(name, tasks=tasks[start : start + 500])
        print(f"{name}: {len(tasks)} tasks")


if __name__ == "__main__":
    main()
