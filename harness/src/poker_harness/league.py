"""Opponent leagues for self-play training: per-seat opponents drawn from a league file.

A task with `env_params.opponent_league: true` has its opponent seats drawn from the league file at
`POKER_LEAGUE_FILE` (JSON, re-read whenever it changes, so a script can add or retire opponents while a training run
is going):

    {"version": 3,
     "noise": {"size_jitter": 0.25, "action_epsilon": 0.03},
     "entries": [
       {"id": "base", "model": "vals-hackathon/base", "weight": 35},
       {"id": "lag-copy", "model": "vals-hackathon/ac-vals-model", "persona": "loose_aggressive", "weight": 4},
       {"id": "tag-bot", "bot": "tag", "weight": 4}]}

`bot` names a style bot: one of our presets (`engine/style_bot.py`) or a bot adapted from a public repo
(`engine/adapted_bots.py`).

Each opponent seat draws one entry by weight (entries at weight 0 are retired). The draw is seeded by the deal, the
policy seats and the league version, and the first draw of a task is pinned for `pin_seconds` in a pin file shared by
every harness shard, so all replays of a task (one GRPO group) face the same lineup even when the league changes
while the group is being played.

`env_params.opponent_seats` gives the same per-seat specs directly ({"1": {"bot": "nit"}, "2": {"model": ...}}), with
`env_params.opponent_noise` for their noise; tests and local games use it.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .personas import PERSONAS

logger = logging.getLogger("poker_harness.league")

SPEC_KEYS = {"id", "model", "persona", "bot", "api", "reasoning_effort", "weight"}
NOISE_KEYS = {"size_jitter", "action_epsilon"}


class LeagueError(ValueError):
    pass


def check_spec(spec: Any, where: str) -> dict[str, Any]:
    """One opponent seat's spec: a model (optionally with a persona) or a style bot."""
    from .engine.adapted_bots import bot_styles  # late: the engine imports nothing from here

    styles = bot_styles()
    if not isinstance(spec, dict):
        raise LeagueError(f"{where}: must be an object")
    unknown = set(spec) - SPEC_KEYS
    if unknown:
        raise LeagueError(f"{where}: unknown keys {sorted(unknown)}")
    if ("model" in spec) == ("bot" in spec):
        raise LeagueError(f"{where}: needs exactly one of model or bot")
    if "model" in spec and not (isinstance(spec["model"], str) and spec["model"]):
        raise LeagueError(f"{where}: model must be a non-empty string")
    if "bot" in spec and spec["bot"] not in styles:
        raise LeagueError(f"{where}: unknown bot style {spec['bot']!r}; known: {sorted(styles)}")
    if "persona" in spec and ("model" not in spec or spec["persona"] not in PERSONAS):
        raise LeagueError(f"{where}: persona needs a model and one of {sorted(PERSONAS)}")
    if spec.get("api", "chat") not in ("chat", "responses"):
        raise LeagueError(f"{where}: api must be chat or responses")
    return spec


def check_noise(noise: Any, where: str = "noise") -> dict[str, float]:
    if noise is None:
        return {}
    if not isinstance(noise, dict) or set(noise) - NOISE_KEYS:
        raise LeagueError(f"{where}: must be an object with {sorted(NOISE_KEYS)}")
    out = {}
    for key, value in noise.items():
        if isinstance(value, bool) or not isinstance(value, int | float) or not 0 <= value < 1:
            raise LeagueError(f"{where}.{key} must be a number in [0, 1)")
        out[key] = float(value)
    return out


def check_league(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict) or not isinstance(data.get("entries"), list):
        raise LeagueError("league: needs an entries list")
    if isinstance(data.get("version"), bool) or not isinstance(data.get("version"), int):
        raise LeagueError("league: needs an integer version")
    ids = set()
    for i, entry in enumerate(data["entries"]):
        check_spec(entry, f"entries[{i}]")
        weight = entry.get("weight")
        if isinstance(weight, bool) or not isinstance(weight, int | float) or weight < 0:
            raise LeagueError(f"entries[{i}]: weight must be a number >= 0")
        if not isinstance(entry.get("id"), str) or entry["id"] in ids:
            raise LeagueError(f"entries[{i}]: needs a unique string id")
        ids.add(entry["id"])
    if not any(e["weight"] > 0 for e in data["entries"]):
        raise LeagueError("league: every entry has weight 0")
    check_noise(data.get("noise"))
    return data


def explicit_lineup(params: dict[str, Any]) -> dict[str, Any] | None:
    """`env_params.opponent_seats` as a lineup, or None when the task doesn't set it."""
    seats = params.get("opponent_seats")
    if seats is None:
        return None
    if not isinstance(seats, dict) or not seats:
        raise LeagueError("opponent_seats must be an object of seat number to spec")
    lineup = {}
    for seat, spec in seats.items():
        if not str(seat).isdigit():
            raise LeagueError(f"opponent_seats: {seat!r} is not a seat number")
        lineup[int(seat)] = check_spec(spec, f"opponent_seats[{seat}]")
    return {"version": None, "seats": lineup, "noise": check_noise(params.get("opponent_noise"), "opponent_noise")}


def draw(entries: list[dict[str, Any]], seed: int, policy: list[int], seats: int, version: int) -> dict[int, dict]:
    """Each non-policy seat's entry, drawn by weight, the same for the same deal, policy seats and version."""
    live = [e for e in entries if e["weight"] > 0]
    rng = random.Random(f"league:{seed}:{','.join(map(str, sorted(policy)))}:{version}")
    weights = [e["weight"] for e in live]
    return {seat: dict(rng.choices(live, weights)[0]) for seat in range(seats) if seat not in policy}


class League:
    """The league file, re-read when its mtime changes (a bad edit keeps the last good league), plus the pins."""

    def __init__(self, path: Path, pins: Path | None = None, pin_seconds: float = 1200.0) -> None:
        self.path = path
        self.pins = pins or path.parent / f"{path.stem}_pins"
        self.pin_seconds = pin_seconds
        self._mtime: float | None = None
        self._league: dict[str, Any] | None = None

    def current(self) -> dict[str, Any]:
        try:
            mtime = self.path.stat().st_mtime
        except OSError as exc:
            if self._league is None:
                raise LeagueError(f"no league file at {self.path}") from exc
            return self._league
        if mtime != self._mtime:
            try:
                self._league = check_league(json.loads(self.path.read_text(encoding="utf-8")))
            except (OSError, ValueError) as exc:  # LeagueError is a ValueError
                if self._league is None:
                    raise LeagueError(f"bad league file {self.path}: {exc}") from exc
                logger.warning("league file %s unreadable, keeping version %s: %s", self.path, self._league["version"], exc)
            self._mtime = mtime
        return self._league

    def lineup(self, seed: int, policy: list[int], seats: int) -> dict[str, Any]:
        """{"version", "seats": {seat: spec}, "noise"}: pinned per task, so a task's replays share one lineup."""
        key = f"{seed}-{'_'.join(map(str, sorted(policy)))}-{seats}"
        pin = self.pins / f"{key}.json"
        try:
            if time.time() - pin.stat().st_mtime < self.pin_seconds:
                saved = json.loads(pin.read_text(encoding="utf-8"))
                return {**saved, "seats": {int(s): spec for s, spec in saved["seats"].items()}}
        except (OSError, ValueError, KeyError):
            pass  # no pin yet, expired, or half-written by another shard: draw (the draw is deterministic anyway)
        league = self.current()
        lineup = {
            "version": league["version"],
            "seats": draw(league["entries"], seed, policy, seats, league["version"]),
            "noise": check_noise(league.get("noise")),
        }
        self.pins.mkdir(parents=True, exist_ok=True)
        tmp = self.pins / f".{key}.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(lineup), encoding="utf-8")
        os.replace(tmp, pin)
        return lineup


def lineup_player(
    seat: int,
    spec: dict[str, Any],
    *,
    rules: Any,
    params: dict[str, Any],
    key: str,
    make_complete: Callable[[str, dict[str, Any]], Any],
    timeout_seconds: float | None = None,
    talk_lines: tuple[str, ...] = (),
    talk_rate: float = 0.0,
) -> Any:
    """The player for one league seat: a style bot, or a model seat (with an optional persona) whose calls
    `make_complete(api, completion_params)` builds."""
    from .completions import cleanup_params, opponent_params
    from .engine.adapted_bots import make_bot
    from .seat import ModelSeat

    if "bot" in spec:
        return make_bot(spec["bot"], f"seat{seat}", key, rules.big_blind)
    seat_params = params | {"opponent_model": spec["model"]}
    if "reasoning_effort" in spec:
        seat_params["opponent_reasoning_effort"] = spec["reasoning_effort"]
    api = spec.get("api", params.get("opponent_api", "chat"))
    theirs = opponent_params(seat_params)
    persona = spec.get("persona")
    return ModelSeat(
        f"seat{seat}",
        make_complete(api, theirs),
        rules,
        model=f"opponent/{spec['model']}" + (f"+{persona}" if persona else ""),
        timeout_seconds=timeout_seconds,
        cleanup_complete=make_complete(api, cleanup_params(theirs)),
        system_hint=PERSONAS[persona] if persona else "",
        talk_lines=talk_lines,
        talk_rate=talk_rate,
        talk_seed=int(params["seed"]),
    )


def with_noise(players: list[Any], lineup_seats: dict[int, dict], noise: dict[str, float], key: str) -> list[Any]:
    """League seats wrapped in `NoisyPlayer` (when the noise is on); the policy seats are never perturbed."""
    from .engine.noise import NoisyPlayer

    if not any(noise.values()):
        return players
    return [NoisyPlayer(p, f"{key}:{i}", **noise) if i in lineup_seats else p for i, p in enumerate(players)]


def lineup_log(lineup: dict[str, Any], lineup_seats: dict[int, dict], players: list[Any]) -> dict[str, Any]:
    """What `log["lineup"]` records: the league version, the noise and each seat's spec (a bot's drawn numbers too)."""
    return {
        "version": lineup["version"],
        "noise": lineup.get("noise") or {},
        "seats": {
            f"seat{i}": {k: v for k, v in spec.items() if k != "weight"}
            | ({"params": players[i].describe()} if "bot" in spec else {})
            for i, spec in sorted(lineup_seats.items())
        },
    }


def league_from_env() -> League | None:
    path = os.environ.get("POKER_LEAGUE_FILE")
    if not path:
        return None
    pins = os.environ.get("POKER_LEAGUE_PINS")
    return League(Path(path), Path(pins) if pins else None, float(os.environ.get("POKER_LEAGUE_PIN_SECONDS", 1200)))
