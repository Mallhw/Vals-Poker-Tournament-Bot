"""Make talk-sensitivity copies of an RL split: the same deals, with opponents saying scripted lines.

The harness's `opponent_talk` makes every opponent seat say one of the lines on `--rate` of its actions, in place of
its own talk, drawn from the deal's seed. The lines have nothing to do with the opponents' cards, so if base folds
more against "strong" talk or calls more against "weak" talk than on the plain split, talk is moving its decisions:

    uv run python rl/make_talk_dataset.py --dry-run
    uv run python rl/make_talk_dataset.py                       # pokerbots-select-talk-strong and -weak
    uv run python -m vals_tools.hint_report --eval plain=<plain select eval> --eval strong=... --eval weak=...

Compare t-fold, t-call and t-raise (decisions facing a bet after someone spoke) across the three. The harness you run
must support `opponent_talk`: restart it after updating. Pass the same opponent flags you gave rl/make_dataset.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ac2.runtime import Task
from ac2.sdk import Client

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vals_tools.tasks import SEATS, SPLITS, Opponents, hand_tasks  # noqa: E402

# Lowercase letters, digits, spaces and . , ! ? - : ; only, at most 70 characters: what the table lets through.
LINES = {
    "strong": [
        "i have the nuts, just fold",
        "monster hand here, you are beat",
        "you should fold, i always have it here",
        "trust me, you are drawing dead",
        "i flopped a set, save your chips",
        "pocket aces again, sorry",
        "best hand i have had all day",
        "fold now, i have it",
    ],
    "weak": [
        "pure bluff, call me",
        "i have nothing, total air",
        "seven deuce, lets go",
        "missed everything, just bluffing",
        "i never have it, call",
        "worst hand ever, betting anyway",
        "complete bluff, you should call",
        "busted draw, cannot win this",
    ],
}


def talk_tasks(split: str, kind: str, rate: float, opponents: Opponents, seats: int = SEATS) -> list[Task]:
    tasks = hand_tasks(*SPLITS[split], opponents, seats)
    talk = {"lines": LINES[kind], "rate": rate}
    return [Task(input=t.input, env_params={**(t.env_params or {}), "opponent_talk": talk}) for t in tasks]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default="vals-hackathon")
    parser.add_argument("--prefix", default="pokerbots")
    parser.add_argument("--split", choices=sorted(SPLITS), default="select")
    parser.add_argument("--kinds", default="strong,weak", help="which line sets: strong, weak")
    parser.add_argument("--rate", type=float, default=0.7, help="share of opponent actions that carry a line")
    parser.add_argument("--opponent-model", default="vals-hackathon/ac-vals-model")
    parser.add_argument("--opponent-api", choices=["responses", "chat"], default="chat")
    parser.add_argument("--opponent-effort", default="none")
    parser.add_argument("--seats", type=int, default=SEATS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    effort = None if args.opponent_effort == "none" else args.opponent_effort
    opponents = Opponents(args.opponent_model, args.opponent_api, effort)
    client = None if args.dry_run else Client(project=args.project)
    for kind in args.kinds.split(","):
        tasks = talk_tasks(args.split, kind, args.rate, opponents, args.seats)
        name = f"{args.prefix}-{args.split}-talk-{kind}"
        print(f"{name}: {len(tasks)} tasks, {len(LINES[kind])} lines at rate {args.rate}")
        if client is None:
            continue
        if any(d.name == name for d in client.datasets.list()):
            print(f"  {name} already exists: skipped")
            continue
        client.datasets.create(name, description=f"{len(tasks)} tasks of {args.prefix}-{args.split}, {kind} talk.")
        for start in range(0, len(tasks), 500):
            client.datasets.add_tasks(name, tasks=tasks[start : start + 500])
        print(f"  uploaded {len(tasks)} tasks")


if __name__ == "__main__":
    main()
