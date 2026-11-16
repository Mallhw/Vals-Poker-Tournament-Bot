"""Add opponents' table talk to PokerBench solver spots, so one GRPO round also teaches the model to ignore talk.

A spot's label is the solver's play, which doesn't depend on what opponents say. So a talk-injected copy keeps the same
label, and a model that lets talk move its decision loses reward. Talk is written exactly as the harness renders it
at a talk-on table (`prompts._action_text`): `  flop: seat4 (seat 4) bet to 12 and says: "i have the nuts"`.

Line pools (all already in the table's alphabet, at most 70 characters):
  - strong / weak: the scripted sets from rl/make_talk_dataset.py (claims of strength / of bluffing);
  - arena: real talk from other teams' entries in data/arena/matches (hand claims, bluffs, chatter, fake "admin" notes);
  - mixed: all of them.
Only opponents' action lines get talk (never the hero's), each with probability `--rate`; an augmented spot always has
at least one talking opponent in the history it sees last, so the decision comes right after talk.

    uv run python rl/spot_talk.py --in data/pokerbench/preflop_train.jsonl --out data/pokerbench/preflop_train_talk.jsonl
    uv run python rl/spot_talk.py --in data/pokerbench/postflop_test.jsonl --out data/pokerbench/postflop_test_talk-strong.jsonl \
        --pool strong --share 1

`--share` is the fraction of spots that get talk (the rest are copied unchanged); each output record gets a `talk`
field naming its pool, or null. Record order and count are kept, so `pokerbench_replies.py --count N --seed S` picks
the same spots from a talk copy as from the original, which pairs a talk test spot by spot.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rl"))
from make_talk_dataset import LINES  # noqa: E402

TEAM = "org_9197691edc5e"  # our own entries' talk is left out of the arena pool
ACTION_LINE = re.compile(r"^(\s+(?:preflop|flop|turn|river): )(seat\d+) \(seat (\d+)\) (fold|check|call|bet|raise)(.*)$")
ALLOWED = re.compile(r"^[a-z0-9 .,!?:;-]{1,70}$")


def arena_lines(matches: str = str(ROOT / "data" / "arena" / "matches" / "m_*.json")) -> list[str]:
    """Distinct talk lines other teams' entries said in the downloaded arena matches."""
    seen = set()
    for path in glob.glob(matches):
        match = json.loads(Path(path).read_text(encoding="utf-8"))
        kind = {s["seat"]: s.get("entry") or {} for s in match["seats"]}
        for hand in match["hands"]:
            for a in hand["actions"]:
                e = kind.get(a["seat"], {})
                say = (a.get("say") or "").strip()
                if say and e.get("kind") == "team" and e.get("org_id") != TEAM and ALLOWED.match(say):
                    seen.add(say)
    return sorted(seen)


def pools() -> dict[str, list[str]]:
    arena = arena_lines()
    return {"strong": LINES["strong"], "weak": LINES["weak"], "arena": arena,
            "mixed": LINES["strong"] + LINES["weak"] + arena}


def add_talk(record: dict, rng: random.Random, lines: list[str], rate: float) -> dict:
    """A copy of the spot with opponents' action lines given talk; the hero's own lines are never touched."""
    hero = int(record["hero_seat"])
    messages = [dict(m) for m in record["messages"]]
    users = [i for i, m in enumerate(messages) if m.get("role") == "user" and isinstance(m.get("content"), str)]
    last_with_actions = None
    for i in users:
        out, opp_lines, spoke = [], [], False
        for line in messages[i]["content"].split("\n"):
            found = ACTION_LINE.match(line)
            if found and int(found.group(3)) != hero and "and says:" not in line:
                opp_lines.append(len(out))
                if rng.random() < rate:
                    line = f'{line} and says: "{rng.choice(lines)}"'
                    spoke = True
            out.append(line)
        if opp_lines:
            last_with_actions = (i, out, opp_lines, spoke)
        messages[i] = {**messages[i], "content": "\n".join(out)}
    if last_with_actions and not last_with_actions[3]:  # make sure the decision comes right after talk
        i, out, opp_lines, _ = last_with_actions
        k = opp_lines[-1]
        out[k] = f'{out[k]} and says: "{rng.choice(lines)}"'
        messages[i] = {**messages[i], "content": "\n".join(out)}
    return {**record, "messages": messages}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="src", type=Path, required=True, help="spots from pokerbench_convert.py")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pool", choices=["mixed", "strong", "weak", "arena"], default="mixed")
    parser.add_argument("--rate", type=float, default=0.5, help="chance each opponent action line carries talk")
    parser.add_argument("--share", type=float, default=0.5, help="fraction of spots that get talk")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    lines = pools()[args.pool]
    rng = random.Random(args.seed)
    n = talked = 0
    with args.src.open(encoding="utf-8") as src, args.out.open("w", encoding="utf-8") as out:
        for raw in src:
            if not raw.strip():
                continue
            record = json.loads(raw)
            n += 1
            if rng.random() < args.share and any(
                ACTION_LINE.match(l) and int(ACTION_LINE.match(l).group(3)) != int(record["hero_seat"])
                for m in record["messages"] if m.get("role") == "user" and isinstance(m.get("content"), str)
                for l in m["content"].split("\n")
            ):
                record = add_talk(record, rng, lines, args.rate) | {"talk": args.pool}
                talked += 1
            else:
                record = {**record, "talk": None}
            out.write(json.dumps(record) + "\n")
    print(f"{args.out}: {n} spots, {talked} with {args.pool} talk ({len(lines)} lines in the pool)")


if __name__ == "__main__":
    main()
