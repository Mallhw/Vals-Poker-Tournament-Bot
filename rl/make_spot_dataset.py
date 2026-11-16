"""Make GRPO datasets of PokerBench solver spots: one decision per task, rewarded for matching the solver.

Each task carries a converted spot (`harness/pokerbench_convert.py`) in `env_params.pokerbench_spot`; the harness asks
the policy that one decision and rewards it continuously for how close it lands to the solver's play
(`harness/src/poker_harness/solver_spot.py`). No opponents play, so no opponent model is needed.

    uv run python rl/make_spot_dataset.py --dry-run
    uv run python rl/make_spot_dataset.py                       # makes pbspot-train and pbspot-select
    uv run python rl/train.py --prefix pbspot --max-response-len 5000 --max-total-len 16000 --dry-run

`<prefix>-train` samples `--count` spots from each training file (PokerBench's train split). `<prefix>-select` is the
held-out set every checkpoint has been scored on (data/pokerbench/heldout_base.jsonl: 300 spots of PokerBench's test
split), so the in-run eval reports the same agreement as `vals_tools.heldout_check`.

`--low-raise-count N` weights the training spots by situation, never by the solver's answer: from each preflop file it
takes N spots with at most one raise before the hero's decision (opens, isolations, facing one open: where the arena
showed us limping and flatting) and `--count` spots with more raises. `--reward action_first` has the harness score
the action before the size (`harness/src/poker_harness/solver_spot.py`), so calling cannot hedge between fold and raise.

    uv run python rl/make_spot_dataset.py --prefix pbspot3 --count 750 --low-raise-count 1500 --reward action_first --seed 29
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from ac2.runtime import Message, Task
from ac2.sdk import Client

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rl"))
from spot_situations import raises_before_hero  # noqa: E402
from vals_tools.tasks import TABLE_RULES  # noqa: E402

DATA = ROOT / "data" / "pokerbench"
SPOT_FIELDS = ("messages", "legal", "label", "solver_text", "source", "pot", "street", "hero_position")
TOOLS = json.loads((ROOT / "opsd" / "tools.json").read_text())


def spot_task(record: dict[str, Any], seed: int, reward: str | None = None, leak_penalty: float = 0,
              talk_penalty: float = 0) -> Task:
    if "tools" in record and record["tools"] != TOOLS:  # the harness rebuilds these tools; they must match the spot's
        raise SystemExit(f"tools in {record['source']} differ from opsd/tools.json: reconvert with this repo's harness")
    spot = {k: record[k] for k in SPOT_FIELDS} | ({"reward": reward} if reward else {}) | (
        {"leak_penalty": leak_penalty} if leak_penalty else {}) | ({"talk_penalty": talk_penalty} if talk_penalty else {})
    return Task(
        input=[Message(role="user", content="Make the decision in one PokerBench spot.")],
        env_params=TABLE_RULES | {"seats": 6, "total_hands": 1, "seed": seed, "pokerbench_spot": spot},
    )


def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def train_tasks(paths: list[Path], count: int, seed: int, low_raise_count: int = 0,
                reward: str | None = None, counts: list[str] | None = None, leak_penalty: float = 0,
                talk_penalty: float = 0) -> list[Task]:
    """`count` spots per file, or per-file `counts`: "N" spots, or "LOW+HIGH" (LOW with 0-1 raises before the hero,
    HIGH with more). `low_raise_count` does the LOW+HIGH split for every preflop file, with HIGH = `count`."""
    records = []
    for i, path in enumerate(paths):
        rows = read(path)
        random.Random(seed).shuffle(rows)
        spec = counts[i] if counts else (f"{low_raise_count}+{count}" if low_raise_count and rows
                                         and rows[0].get("street") == "preflop" else str(count))
        if "+" in spec:
            low, high = map(int, spec.split("+"))
            records += [r for r in rows if raises_before_hero(r) <= 1][:low]
            records += [r for r in rows if raises_before_hero(r) > 1][:high]
        else:
            records += rows[: int(spec)]
    random.Random(seed).shuffle(records)
    return [spot_task(r, i, reward, leak_penalty, talk_penalty) for i, r in enumerate(records)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default="vals-hackathon")
    parser.add_argument("--prefix", default="pbspot")
    parser.add_argument("--train", nargs="+", type=Path,
                        default=[DATA / "preflop_train.jsonl", DATA / "postflop_train_25k.jsonl"])
    parser.add_argument("--count", type=int, default=1500, help="training spots per file (default 1500)")
    parser.add_argument("--select", type=Path, default=DATA / "heldout_base.jsonl", help="held-out spots")
    parser.add_argument("--counts", type=lambda s: s.split(","),
                        help="per --train file, in order: N spots, or LOW+HIGH (0-1 raises before the hero + more)")
    parser.add_argument("--low-raise-count", type=int, default=0,
                        help="per preflop file: this many spots with 0-1 raises before the hero, plus --count with more")
    parser.add_argument("--reward", choices=("distance", "action_first", "lean_aggressive", "lean_small"),
                        help="the harness's spot reward for the training tasks (default: the harness's, distance)")
    parser.add_argument("--leak-penalty", type=float, default=0,
                        help="points a training reply loses when its say names its hand (harness HAND_TALK)")
    parser.add_argument("--talk-penalty", type=float, default=0,
                        help="points a training reply loses for any non-empty say")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    built = {
        f"{args.prefix}-train": train_tasks(args.train, args.count, args.seed, args.low_raise_count, args.reward,
                                            args.counts, args.leak_penalty, args.talk_penalty),
        f"{args.prefix}-select": [spot_task(r, 1_000_000 + i) for i, r in enumerate(read(args.select))],
    }
    for name, tasks in built.items():
        streets = {}
        for t in tasks:
            street = t.env_params["pokerbench_spot"]["street"]
            streets[street] = streets.get(street, 0) + 1
        print(f"{name}: {len(tasks)} spots {streets}")
    if args.dry_run:
        first = built[f"{args.prefix}-train"][0].env_params
        print({k: v for k, v in first.items() if k != "pokerbench_spot"}, "| spot:", first["pokerbench_spot"]["solver_text"])
        return
    client = Client(project=args.project)
    existing = {d.name for d in client.datasets.list()}
    for name, tasks in built.items():
        if name in existing:
            print(f"{name} already exists: not adding to it")
            continue
        client.datasets.create(name, description=f"{len(tasks)} PokerBench solver spots (one decision per task).")
        for start in range(0, len(tasks), 250):
            client.datasets.add_tasks(name, tasks=tasks[start : start + 250])
        print(f"uploaded {len(tasks)} tasks to {name}")


if __name__ == "__main__":
    main()
