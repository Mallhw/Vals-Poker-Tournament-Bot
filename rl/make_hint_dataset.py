"""Make a hinted copy of an RL split: the same deals, with a test hint in every task's `env_params.system_hint`.

The harness appends the hint to the policy seat's system prompt exactly as OPSD's teacher sees it, so base playing
this dataset shows what that hint does, before you spend GPUs distilling it. Compare against base on the plain split;
`vals_tools.score` pairs the two evals by deal:

    uv run python rl/make_hint_dataset.py --hint-file opsd/hint.py --name tag --dry-run
    uv run python rl/make_hint_dataset.py --hint-file opsd/hint.py --name tag      # makes pokerbots-select-hint-tag
    uv run python -m vals_tools.eval --dataset pokerbots-select --deployment vals-hackathon/base
    uv run python -m vals_tools.eval --dataset pokerbots-select-hint-tag --deployment vals-hackathon/base
    uv run python -m vals_tools.score --eval <hinted eval> --base <plain eval>

`--hint-file` is a text file, or a .py file that defines HINT (such as opsd/hint.py). Screen hints on `select` (500
hands, the default); keep `eval` for the final verdict on a trained model. Pass the same opponent flags you gave
rl/make_dataset.py, so both datasets play the same opponents. The harness you run must have `system_hint` support,
and its scores the `deal` artifact: restart it after updating.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import runpy
import sys
from pathlib import Path

from ac2.runtime import Task
from ac2.sdk import Client

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vals_tools.tasks import SEATS, SPLITS, TABLE_SPLITS, Opponents, split_tasks, spread_opponents  # noqa: E402


def read_hint(path: Path) -> str:
    """A hint file's text, or HINT from a .py file (such as opsd/hint.py)."""
    hint = runpy.run_path(str(path))["HINT"] if path.suffix == ".py" else path.read_text(encoding="utf-8")
    if not isinstance(hint, str) or not hint.strip():
        raise SystemExit(f"{path} has no hint text")
    return hint.strip()


def hinted_tasks(
    split: str,
    hint: str,
    opponents: Opponents,
    seats: int = SEATS,
    limit: int | None = None,
    total_hands: int = 1,
    models: list[str] | None = None,
) -> list[Task]:
    """The split's tasks (same seeds and seats as rl/make_dataset.py), each with `system_hint` set. `total_hands` above
    1 makes tables, and `models` spreads opponents exactly as rl/make_dataset.py does, so deals stay paired."""
    tasks = spread_opponents(split_tasks(split, opponents, seats, total_hands, limit), models)
    # Built anew, not copied, so each task's content hash covers the hint.
    return [Task(input=t.input, env_params={**(t.env_params or {}), "system_hint": hint}) for t in tasks]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default="vals-hackathon")
    parser.add_argument("--prefix", default="pokerbots")
    parser.add_argument("--hint-file", type=Path, required=True)
    parser.add_argument("--name", required=True, help="short name for the hint: lowercase letters, digits and -")
    parser.add_argument("--split", choices=sorted({*SPLITS, *TABLE_SPLITS}), default="select")
    parser.add_argument("--total-hands", type=int, default=1, help="more than 1: N-hand tables, as rl/make_dataset.py")
    parser.add_argument("--opponent-models", help="comma-separated deployments to spread over the tasks, as make_dataset")
    parser.add_argument("--limit", type=int, help="only the first N tasks (a multiple of 6 covers every seat evenly)")
    parser.add_argument("--opponent-model", default="vals-hackathon/ac-vals-model")
    parser.add_argument("--opponent-api", choices=["responses", "chat"], default="chat")
    parser.add_argument("--opponent-effort", default="none")
    parser.add_argument("--seats", type=int, default=SEATS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,40}", args.name):
        parser.error("--name: lowercase letters, digits and -, at most 41 characters")

    hint = read_hint(args.hint_file)
    effort = None if args.opponent_effort == "none" else args.opponent_effort
    models = [m.strip() for m in args.opponent_models.split(",") if m.strip()] if args.opponent_models else None
    if args.total_hands > 1 and args.split not in TABLE_SPLITS:
        parser.error(f"--split for tables: choose from {sorted(TABLE_SPLITS)}")
    if args.total_hands == 1 and args.split not in SPLITS:
        parser.error(f"--split for single hands: choose from {sorted(SPLITS)}")
    opponents = Opponents(args.opponent_model, args.opponent_api, effort)
    tasks = hinted_tasks(args.split, hint, opponents, args.seats, args.limit, args.total_hands, models)
    tables = f"t{args.total_hands}-" if args.total_hands > 1 else ""
    name = f"{args.prefix}-{tables}{args.split}-hint-{args.name}"
    digest = hashlib.sha256(hint.encode()).hexdigest()[:12]  # the grader reports the same hash as `hint`
    print(f"{name}: {len(tasks)} tasks, hint {digest} ({len(hint)} characters)")
    if args.dry_run:
        print(tasks[0].env_params)
        return
    client = Client(project=args.project)
    if any(d.name == name for d in client.datasets.list()):
        raise SystemExit(f"{name} already exists: pick another --name (a dataset keeps every task added to it)")
    client.datasets.create(name, description=f"{len(tasks)} tasks of {args.prefix}-{args.split}, hint {digest}.")
    for start in range(0, len(tasks), 500):
        client.datasets.add_tasks(name, tasks=tasks[start : start + 500])
    print(f"uploaded {len(tasks)} tasks to {name}")


if __name__ == "__main__":
    main()
