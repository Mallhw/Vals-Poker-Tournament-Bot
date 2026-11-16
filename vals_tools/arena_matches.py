"""Download a team's arena matches from pokerbots.ac: one JSON file per match, as the replay page loads it.

Each file has every seat's entry (team, deployment and checkpoint ids), every hole card, every action with its
amount, pot, `say`, tokens, latency and failure flags (format error, timeout, token cap, fallback), the board,
showdown and chip deltas. Thinking is not published. Files go to `data/arena/matches/<match id>.json`; a match
already there is skipped.

    uv run python -m vals_tools.arena_matches                        # our latest matches (the team page lists 10)
    uv run python -m vals_tools.arena_matches --pages 4              # also scan the arena-wide match list, 4 pages
    uv run python -m vals_tools.arena_matches --follow               # keep polling every 5 minutes (Ctrl-C stops)

The team page only shows a team's 10 latest matches (about 25 minutes of play), so `--follow` polls it often
enough to miss nothing. `--pages` backfills from the arena-wide list (25 matches a page, newest first), keeping
only matches with the team seated. Requests are spaced out; the site is public and needs no login.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

SITE = "https://www.pokerbots.ac"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "arena" / "matches"
TEAM = "org_9197691edc5e"  # pineapple
MATCH_ID = re.compile(r"m_\d{8}t\d{6}_[0-9a-f]{6}")
BLOB = re.compile(r"https://[a-z0-9]+\.public\.blob\.vercel-storage\.com/matches/[^\"\\\s]+?\.json")
PAUSE_S = 0.5
HEADERS = {"User-Agent": "Mozilla/5.0 (arena match downloader)"}


def get(url: str) -> str:
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def match_ids(page_url: str) -> list[str]:
    """Match ids on a list page, in page order (newest first), without repeats."""
    return list(dict.fromkeys(MATCH_ID.findall(get(page_url))))


def fetch_match(match_id: str) -> dict | None:
    """The match's replay JSON, or None if its page links none (an unfinished match)."""
    found = BLOB.search(get(f"{SITE}/matches/{match_id}"))
    time.sleep(PAUSE_S)
    if not found:
        return None
    match = json.loads(get(found.group(0)))
    time.sleep(PAUSE_S)
    return match


def seated(match: dict, team: str) -> bool:
    return any((seat.get("entry") or {}).get("org_id") == team for seat in match.get("seats", []))


def collect(ids: list[str], team: str, out: Path) -> tuple[int, int]:
    """Download the matches not on disk yet; keep those with `team` seated. Returns (saved, skipped as not ours)."""
    saved = other = 0
    for match_id in ids:
        path = out / f"{match_id}.json"
        if path.exists() or (out / ".others" / match_id).exists():
            continue
        try:
            match = fetch_match(match_id)
        except Exception as error:  # a network blip: try again on the next poll
            print(f"{match_id}: {error}", flush=True)
            continue
        if match is None or match.get("status") not in (None, "complete", "completed", "finished"):
            continue
        if not seated(match, team):
            (out / ".others").mkdir(exist_ok=True)
            (out / ".others" / match_id).touch()
            other += 1
            continue
        path.write_text(json.dumps(match), encoding="utf-8")
        saved += 1
    return saved, other


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--team", default=TEAM, help="the team's org id (default: pineapple)")
    parser.add_argument("--pages", type=int, default=0, help="also scan this many pages of the arena-wide list")
    parser.add_argument("--follow", action="store_true", help="poll the team page every --every seconds")
    parser.add_argument("--every", type=int, default=300)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ids = []
    for page in range(1, args.pages + 1):
        ids += match_ids(f"{SITE}/matches" + (f"?page={page}" if page > 1 else ""))
        time.sleep(PAUSE_S)
    while True:
        try:
            ids = match_ids(f"{SITE}/teams/{args.team}") + ids
        except Exception as error:
            print(f"team page: {error}", flush=True)
        saved, other = collect(list(dict.fromkeys(ids)), args.team, args.out)
        total = len(list(args.out.glob("m_*.json")))
        print(f"{time.strftime('%H:%M')} saved {saved} new matches ({other} without us); {total} on disk", flush=True)
        if not args.follow:
            return
        ids = []
        time.sleep(args.every)


if __name__ == "__main__":
    main()
