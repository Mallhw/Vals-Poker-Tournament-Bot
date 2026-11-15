"""Sample grader: how the harness scores each rollout for AC2.

How it is wired
---------------
A run that names no grader is graded by the harness itself, through AC2's Grader API (routes in `server.py`):

    POST /submit_grading      start grading one rollout; AC2 may repeat it, so it is idempotent
    POST /get_grading_status  queued | running | completed | failed
    POST /collect_score       the reply `grade_log` builds below: {"score", "reasoning", "artifacts"}

`server.py` calls `check_grader_params` with the task's `grader_params`, then `grade_log` with the finished game log,
or `failure_score` when the rollout never produced one. Nothing in this module talks to the network.

What it scores
--------------
The trained seat's result in big blinds. With `env_params.ev_labels`, an all-in before the river counts its expected
share of the pot instead of the cards that came (`ev.py`); otherwise it is the chips actually won.

    x            = clip(EV, -100 BB, +100 BB) / ev_unit_bb
    hand_reward  = sign(x) * ln(1 + |x|)               (shape "signed_log", the default)
    hand_reward *= loss_weight   if it is negative     (losses hurt more than equal wins)
    score        = failure_gap + hand_reward - hand_reward(-100 BB)

With the defaults (ev_unit_bb 1, loss_weight 1.25, failure_gap 2):

    EV (BB)   -100   -20    -5    -1     0    +1    +5   +20  +100
    score     2.00  3.96  5.53  6.90  7.77  8.46  9.56 10.81 12.38

Why this shape, not the old `6 + clip(EV, -5, +5)`:
- The old clip made every pot over 5 BB look the same, so a correct 100 BB stack-off and a reckless one both scored
  like a 5 BB pot. The log curve still separates 5, 20 and 100 BB while keeping small pots (most hands) well spread.
- In the event a busted seat is out for the rest of the 30-hand game, so losing chips costs more than winning the
  same amount gains. `loss_weight` > 1 makes the reward risk-averse in that direction.
- GRPO only sees differences within a group, so a constant offset changes nothing except the distance between a
  failure and a real hand. The old gap was 1 point (0 vs. the worst hand's 1), smaller than a typical small-pot
  swing; `failure_gap` sets it explicitly (2 by default).

Model-caused failures score exactly 0: a format error (no usable `submit_action`, or a reply cut off at the token
cap), or a rollout AC2 reports as failed or truncated. Rewards are never negative, a failure is always worse than any
real hand, and failed samples still train instead of being dropped.

Every knob can be set per task in the dataset's `grader_params` (see `DEFAULT_PARAMS`). To get the old reward back
exactly: {"shape": "clip", "loss_weight": 1, "failure_gap": 1}.

Extension points (search for "EXTENSION POINT")
-----------------------------------------------
1. `hand_reward`: the raw reward for a played hand. Replace it with your own.
2. `GRADER_PARAMS`: per-task knobs your reward reads from the dataset's `grader_params`.
3. `artifacts`: anything you add to the reply is stored with each score row on AC2, for analysis.
Keep failures at 0 and real hands above it, or a model can learn that failing pays.

This grader scores chips only. On single-hand tasks the scratchpad starts blank and its after-hand update is never
read, so nothing here rewards or exercises scratchpad use; how to train that is up to you.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

FAILURE_SCORE = 0.0  # every model-caused failure
EV_CLIP_BB = 100.0  # one hand can't swing the reward by more than a full stack
CLIP_LIMIT = 5.0  # shape "clip" only: the old +-5 band
SHAPES = ("signed_log", "clip")
MAX_SCORE_BYTES = 1 << 20  # AC2 rejects collect_score replies over 1 MiB

# EXTENSION POINT 2: the `grader_params` keys your reward accepts, with their defaults. Unknown keys fail the
# grading, so a typo in a dataset can't silently change the reward.
DEFAULT_PARAMS: dict[str, Any] = {
    "shape": "signed_log",  # "signed_log" (default) or "clip" (the old +-5 BB band)
    "ev_unit_bb": 1.0,  # BB of EV per unit before shaping
    "loss_weight": 1.25,  # multiplier on negative rewards (risk aversion: busting ends the game)
    "failure_gap": 2.0,  # how far the worst real hand scores above a failure (0)
}
GRADER_PARAMS: frozenset[str] = frozenset(DEFAULT_PARAMS)


class GradingError(ValueError):
    """The log can't be scored. AC2 drops the sample (GRADER_ERROR) instead of training on a made-up reward."""


@dataclass
class SeatResult:
    """What the trained seat did over one game log."""

    ev_bb: float  # EV in big blinds (all-in luck removed when the log has EV labels)
    realized_bb: float  # chips actually won, in big blinds
    method: str  # how ev_bb was computed: "realized", "allin", "showdown", ...
    decisions: int
    format_errors: int
    folded: bool
    hands: int


def check_grader_params(grader_params: dict[str, Any] | None) -> dict[str, Any]:
    """The task's `grader_params`, after refusing any key not in `GRADER_PARAMS`."""
    given = dict(grader_params or {})
    unknown = set(given) - GRADER_PARAMS
    if unknown:
        raise GradingError(f"unknown grader_params: {sorted(unknown)}")
    params = {**DEFAULT_PARAMS, **given}
    if params["shape"] not in SHAPES:
        raise GradingError(f"grader_params.shape must be one of {SHAPES}, got {params['shape']!r}")
    for key in ("ev_unit_bb", "loss_weight", "failure_gap"):
        value = params[key]
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
            raise GradingError(f"grader_params.{key} must be a positive number, got {value!r}")
        params[key] = float(value)
    return params


def seat_result(log: dict[str, Any], policy_seats: list[str]) -> SeatResult:
    """Sum the trained seat's results over every hand in the log."""
    policy = set(policy_seats)
    if not log.get("hands") or not policy:
        raise GradingError("the rollout has no hands or no policy seats")
    info = log["session_info"]
    big_blind = info["big_blind"]
    start = dict(zip(info["player_names"], info["starting_stacks"], strict=True))
    seats = [(hand, seat) for hand in log["hands"] for seat in policy if seat in hand["starting_state"]]
    actions = [a for hand in log["hands"] for a in hand["actions"] if a["player"] in policy]
    labels = [hand.get("ev", {}).get(seat) for hand, seat in seats]
    if labels and all(label is not None for label in labels):
        ev_bb = sum(label["ev_bb"] for label in labels)
        realized_bb = sum(label["realized_bb"] for label in labels)
        method = labels[0]["method"]
    else:  # no EV labels on the task: score the chips actually won
        realized_bb = ev_bb = sum(log["final_stacks"][seat] - start[seat] for seat in policy) / big_blind
        method = "realized"
    return SeatResult(
        ev_bb=ev_bb,
        realized_bb=realized_bb,
        method=method,
        decisions=len(actions),
        format_errors=sum(bool(a["model_response"].get("format_error")) for a in actions),
        folded=any(a["action"]["type"] == "fold" for a in actions),
        hands=len(log["hands"]),
    )


def shaped_ev(ev_bb: float, params: dict[str, Any]) -> float:
    """EV in BB -> reward before the offset: clipped to +-100 BB, compressed, with losses weighted up."""
    x = max(-EV_CLIP_BB, min(EV_CLIP_BB, ev_bb)) / params["ev_unit_bb"]
    if params["shape"] == "clip":
        value = max(-CLIP_LIMIT, min(CLIP_LIMIT, x))
    else:  # signed_log
        value = math.copysign(math.log1p(abs(x)), x)
    return value * params["loss_weight"] if value < 0 else value


def hand_reward(result: SeatResult, params: dict[str, Any]) -> float:
    """EXTENSION POINT 1: the raw reward for a game the seat played without failing, before the offset.

    The default is `shaped_ev` of the seat's EV. `result` has everything the log says about the trained seat; read
    more from the log in `seat_result` if you need it. `params` is the task's checked `grader_params`. If you replace
    it, also update `reward_floor` so the worst real hand still lands `failure_gap` above a failure.
    """
    return shaped_ev(result.ev_bb, params)


def reward_floor(params: dict[str, Any]) -> float:
    """The lowest value `hand_reward` can return (a full-stack loss)."""
    return shaped_ev(-EV_CLIP_BB, params)


def failure_score(reason: str) -> dict[str, Any]:
    """The reply for a rollout that failed before producing a log (AC2 reported it failed or truncated)."""
    return {"score": FAILURE_SCORE, "reasoning": reason, "artifacts": {"failure": True}}


def grade_log(
    log: dict[str, Any], policy_seats: list[str], params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Score one finished game for its policy seats, as a `collect_score` reply."""
    params = check_grader_params(params)
    result = seat_result(log, policy_seats)
    if result.format_errors:  # model-caused failure: exactly 0, below every real hand
        score = FAILURE_SCORE
    else:
        floor = reward_floor(params)
        score = FAILURE_SCORE + params["failure_gap"] + max(0.0, hand_reward(result, params) - floor)
    return {
        "score": score,
        "reasoning": f"Policy seat {'/'.join(sorted(policy_seats))}: EV {result.ev_bb:+.1f} BB "
        f"(realized {result.realized_bb:+.1f}), {result.method}, {result.format_errors} format errors.",
        # EXTENSION POINT 3: add any per-rollout numbers you want to analyze later.
        "artifacts": {
            "ev_bb": result.ev_bb,
            "realized_bb": result.realized_bb,
            "ev_method": result.method,
            "policy_decisions": result.decisions,
            "policy_format_errors": result.format_errors,
            "folded": result.folded,
            "hands": result.hands,
        },
    }


def bounded(score: dict[str, Any]) -> dict[str, Any]:
    """Drop artifacts, then truncate reasoning, until the reply fits AC2's 1 MiB limit."""
    if len(json.dumps(score)) <= MAX_SCORE_BYTES:
        return score
    trimmed = {"score": score["score"], "reasoning": score.get("reasoning", ""), "artifacts": {"trimmed": True}}
    overflow = len(json.dumps(trimmed)) - MAX_SCORE_BYTES
    if overflow > 0:
        trimmed["reasoning"] = trimmed["reasoning"][: max(0, len(trimmed["reasoning"]) - overflow - 16)]
    return trimmed
