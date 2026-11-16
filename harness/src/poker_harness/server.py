"""Harness API for the AC2 Harness Connector: one table per rollout.

`ac2 harness connect --upstream http://127.0.0.1:<port>` forwards three POSTs here: `submit` starts a table,
`get_status` reports it, and `collect_outputs` returns the game log. When a run names no grader, AC2 grades through
the Grader API here instead (`submit_grading`, `get_grading_status`, `collect_score`), scored by the sample
grader in `grading.py`. Each policy decision is one chat completion against the rollout's relay URL, so AC2 records every call in the rollout's trace.

`env_params.table_talk` (default false) turns on table talk: every seat's `submit_action` gains an optional `say`,
sanitized and shown to the other seats.

`env_params.system_hint` (training and testing only; the arena never sets it) appends a hint to the policy seats'
system prompt, as OPSD's teacher sees it, so a prompt can be tested before it is distilled. Opponents never see it.

`env_params.enforced_max_completion_tokens` (training only) gives the policy seats a tighter output cap than the
prompt states: their replies are requested with it and any longer reply is played as cut off (a format error, scored
0), so training teaches a margin below the event's cap while the prompt stays the event's.

`env_params.opponent_talk` (testing only), `{"lines": [...], "rate": 0.7}`, makes every opponent seat say one of
`lines` on `rate` of its actions in place of its own talk (silent otherwise), drawn per seat and spot from the deal's
seed: the talk carries nothing about their cards, so any change in the policy's play measures how much talk moves it.

`env_params.opponent_league: true` (self-play training only) draws each opponent seat from the league file at
`POKER_LEAGUE_FILE` (`league.py`): frozen checkpoints, persona-prompted models and style bots, with small seeded noise
on their decisions. `env_params.opponent_seats` gives such per-seat opponents directly. The lineup is logged as
`log["lineup"]`.

`env_params.pokerbench_spot` (training on solver spots) makes the rollout a single PokerBench decision instead of a
table: the policy gets the spot's exact conversation once and is graded against the solver's play (`solver_spot.py`).

A server keeps its tables in memory, so scale out by sharding (`shard_router.py` sends every call for a rollout to
the same server), not with `uvicorn --workers`.

    uv run uvicorn poker_harness.server:app --host 127.0.0.1 --port 8765
"""

from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing
import os
import re
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .completions import (
    DEFAULT_DECISION_TIMEOUT_SECONDS,
    GATEWAY_URL,
    cleanup_params,
    opponent_params,
    relay_completion,
    responses_completion,
    rules_output,
    table_rules,
    with_preset,
)
from .ev import label_ev
from .game import play_session
from .grading import GradingError, bounded, check_grader_params, failure_score, grade_log
from .labels import label_folds
from .league import (
    League,
    LeagueError,
    explicit_lineup,
    league_from_env,
    lineup_log,
    lineup_player,
    with_noise,
)
from .prompts import TableRules, talk_fields
from .seat import ModelSeat
from .solver_spot import SPOT_KEY, check_spot, grade_spot, play_spot

logger = logging.getLogger("poker_harness.server")

MAX_OUTPUT_BYTES = 900_000  # the connector rejects upstream responses over 1 MiB


class HarnessTask(BaseModel):
    input: list[Any]
    env_params: dict[str, Any] | None = None


class SubmitRequest(BaseModel):
    ac_api_url: str = Field(min_length=1)
    ac_api_key: str = Field(min_length=1, repr=False)
    task: HarnessTask
    rollout_id: str | None = None


class RolloutRequest(BaseModel):
    rollout_id: str = Field(min_length=1)


class GradingTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: list[Any]
    env_params: dict[str, Any] | None = None
    grader_params: dict[str, Any] | None = None


class SubmitGradingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rollout_id: str = Field(min_length=1, max_length=255)
    rollout_status: Literal["completed", "failed", "truncated"]
    task: GradingTask


class GradingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rollout_id: str = Field(min_length=1, max_length=255)


@dataclass
class Grading:
    status: str = "queued"
    error: str | None = None
    score: dict[str, Any] | None = None
    task: asyncio.Task[None] | None = None


@dataclass
class Rollout:
    id: str
    rules: TableRules
    policy_seats: list[str] = field(default_factory=list)
    status: str = "queued"
    error: str | None = None
    log: dict[str, Any] | None = None
    task: asyncio.Task[None] | None = None


FREE_TEXT = ("reasoning", "read", "raw_output")  # model text kept in research logs while they fit
ESSENTIAL_RESPONSE = (
    "model_name",
    "provider",
    "format_error",
    "fallback_used",
    "afk",
    "afk_reason",
    "error",
    "truncated",
    "submitted_by_tool",
    "say",
    "in_tokens",
    "out_tokens",
    "reasoning_tokens",
)


def _size(log: dict[str, Any]) -> int:
    return len(json.dumps(log, separators=(",", ":")))


def _compact_private(log: dict[str, Any]) -> None:
    """In place: each action's private model text goes, and each seat's private section shrinks to its final
    scratchpad plus counts of its cleanup outcomes. The full history stays in memory (harness-side grading) and in the
    runs directory; a 30-hand scratchpad game's history alone is several MB, past what the connector carries."""
    for hand in log["hands"]:
        for action in hand["actions"]:
            action["model_response"].pop("private", None)
    for seat in (log.get("private") or {}).values():
        cleanups = [h["cleanup"] for h in seat.pop("hands", {}).values() if h.get("cleanup")]
        results = [c.get("result") for c in cleanups]
        seat["cleanups"] = {
            "turns": len(cleanups),
            **{result: results.count(result) for result in ("updated", "cut", "unchanged")},
            "retried": sum(len(c.get("attempts", [])) > 1 for c in cleanups),
        }
        seat["scratchpad_chars"] = len(seat.get("scratchpad", ""))


def _bounded(log: dict[str, Any]) -> dict[str, Any]:
    """The game log as AC2 receives it from `collect_outputs`, under the connector's 1 MiB limit: private history
    compacted (`_compact_private`), then, only while still too big, model free text dropped, the final scratchpads
    dropped (their lengths stay), model responses cut to their essentials, and finally every action cut to the
    move and its chips. Research logs that fit come back unchanged; grading reads the full log, not this."""
    if "private" not in log and _size(log) <= MAX_OUTPUT_BYTES:
        return log
    trimmed = json.loads(json.dumps(log))
    _compact_private(trimmed)
    actions = [a for hand in trimmed["hands"] for a in hand["actions"]]

    def drop_free_text() -> None:
        for action in actions:
            for key in FREE_TEXT:
                action["model_response"].pop(key, None)

    def drop_scratchpads() -> None:
        for seat in (trimmed.get("private") or {}).values():
            seat.pop("scratchpad", None)

    def essential_responses() -> None:
        for action in actions:
            response = action["model_response"]
            action["model_response"] = {k: response[k] for k in ESSENTIAL_RESPONSE if k in response}
            if isinstance(response.get("error"), str):
                action["model_response"]["error"] = response["error"][:200]

    def bare_actions() -> None:
        keep = ("timestep", "player", "seat", "street", "action", "put_in", "chip_change", "stack_after")
        for action in actions:
            flags = {k: action["model_response"][k] for k in ("format_error", "afk") if k in action["model_response"]}
            bare = {k: action[k] for k in keep if k in action} | {"model_response": flags}
            action.clear()
            action.update(bare)

    for step in (drop_free_text, drop_scratchpads, essential_responses, bare_actions):
        if _size(trimmed) <= MAX_OUTPUT_BYTES:
            break
        step()
    return trimmed


def create_app(
    relay_transport: httpx.AsyncBaseTransport | None = None,
    runs_dir: Path | None = None,
    opponent_transport: httpx.AsyncBaseTransport | None = None,
    league: League | None = None,
) -> FastAPI:
    """`relay_transport` and `opponent_transport` replace the network for tests; `runs_dir` keeps every game log.

    With `env_params.policy_seats` and `env_params.opponent_model`, only the policy seats call the rollout's relay
    (the model AC2 evaluates or trains); every other seat calls `opponent_model` at `POKER_OPPONENT_BASE_URL` (default: the AC2 gateway) with
    `POKER_OPPONENT_API_KEY` (default: `AC2_API_KEY`). Without them, every seat calls the relay.
    """
    token = os.environ.get("POKER_HARNESS_TOKEN")
    opponent_key = os.environ.get("POKER_OPPONENT_API_KEY") or os.environ.get("AC2_API_KEY")
    opponent_url = os.environ.get("POKER_OPPONENT_BASE_URL", GATEWAY_URL)
    runs = runs_dir or Path(os.environ.get("POKER_HARNESS_RUNS", "runs"))
    league = league or league_from_env()
    rollouts: dict[str, Rollout] = {}
    gradings: dict[str, Grading] = {}
    pools: list[ProcessPoolExecutor] = []

    def label_pool() -> ProcessPoolExecutor:
        """All-in EV labels are CPU-bound, so they run off the event loop."""
        if not pools:
            workers = max(1, min(3, (os.cpu_count() or 2) - 1))
            pools.append(ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")))
        return pools[0]

    app = FastAPI(title="poker harness")
    timeout = httpx.Timeout(DEFAULT_DECISION_TIMEOUT_SECONDS, connect=30.0)

    def authorize(authorization: str | None) -> None:
        if token and authorization != f"Bearer {token}":
            raise HTTPException(status_code=401, detail="invalid harness token")

    def save_log(rollout: Rollout, request: SubmitRequest) -> None:
        # The relay URL names the AC2 job the hand belongs to; keep it so concurrent runs can be told apart.
        job = re.search(r"/(job_[0-9a-f]+)/", request.ac_api_url)
        rollout.log["relay_job_id"] = job[1] if job else None
        runs.mkdir(parents=True, exist_ok=True)
        (runs / f"{rollout.id}.json").write_text(json.dumps(rollout.log, indent=1))

    async def run_spot(rollout: Rollout, request: SubmitRequest, params: dict[str, Any]) -> None:
        """One PokerBench decision for the policy, judged against the solver (`solver_spot.py`)."""
        rollout.status = "running"
        try:
            enforced = int(params.get("enforced_max_completion_tokens") or 0) or None
            policy_params = params | {"max_completion_tokens": enforced} if enforced else params
            async with httpx.AsyncClient(timeout=timeout, transport=relay_transport) as relay_client:
                complete = relay_completion(relay_client, request.ac_api_url, request.ac_api_key, policy_params)
                rollout.policy_seats = ["hero"]
                cap = enforced or rollout.rules.max_completion_tokens
                rollout.log = await play_spot(rollout.id, int(params["seed"]), params[SPOT_KEY], complete, cap)
            save_log(rollout, request)
            rollout.status = "completed"
        except Exception as exc:  # reported to AC2 as a failed rollout, which scores 0
            logger.exception("rollout %s failed", rollout.id)
            rollout.status, rollout.error = "failed", f"{type(exc).__name__}: {str(exc)[:500]}"

    async def run_table(rollout: Rollout, request: SubmitRequest) -> None:
        params = with_preset(request.task.env_params or {})  # `preset: "event"` applies the event rules
        if rollout.rules.table_talk:  # every seat's tool gains `say`
            params = params | {"submit_fields": talk_fields(params.get("submit_fields"))}
        if SPOT_KEY in params:  # a single solver spot, not a table
            await run_spot(rollout, request, params)
            return
        rollout.status = "running"
        try:
            seats = int(params.get("seats", 10))
            policy = {int(i) for i in params.get("policy_seats", range(seats))}
            hint = params.get("system_hint") or ""  # checked to be a string in `submit`
            enforced = int(params.get("enforced_max_completion_tokens") or 0) or None  # checked in `submit`
            talk = params.get("opponent_talk") or {}  # checked in `submit`
            talk_lines, talk_rate = tuple(talk.get("lines") or ()), float(talk.get("rate", 0.0))
            seed = int(params["seed"])
            if params.get("opponent_league"):  # self-play training: per-seat opponents from the league file
                if league is None:
                    raise RuntimeError("opponent_league needs POKER_LEAGUE_FILE in the harness environment")
                lineup = league.lineup(seed, sorted(policy), seats)
            else:
                lineup = explicit_lineup(params)  # env_params.opponent_seats, or None
            lineup_seats = {i: spec for i, spec in (lineup or {}).get("seats", {}).items() if i not in policy}
            needs_key = any("model" in spec for spec in lineup_seats.values()) or (
                params.get("opponent_model") and len(policy) + len(lineup_seats) < seats
            )
            if needs_key and not opponent_key:
                raise RuntimeError("opponent seats need POKER_OPPONENT_API_KEY or AC2_API_KEY in the harness environment")
            decision_timeout = float(params.get("decision_timeout_seconds", DEFAULT_DECISION_TIMEOUT_SECONDS))
            opponent_timeout = float(params.get("opponent_timeout_seconds", decision_timeout))
            async with (
                httpx.AsyncClient(timeout=timeout, transport=relay_transport) as relay_client,
                httpx.AsyncClient(timeout=timeout, transport=opponent_transport) as opponent_client,
            ):
                # A training-only tighter cap: asked of the server for the policy seats and enforced by their seats,
                # while the prompt still states the table's cap.
                policy_params = params | {"max_completion_tokens": enforced} if enforced else params
                policy_complete = relay_completion(relay_client, request.ac_api_url, request.ac_api_key, policy_params)
                opponent_api = responses_completion if params.get("opponent_api") == "responses" else relay_completion
                opponent_complete = (
                    opponent_api(opponent_client, opponent_url, opponent_key or "", opponent_params(params))
                    if params.get("opponent_model")
                    else policy_complete
                )
                # The after-hand cleanup turn has its own output cap (`cleanup_max_completion_tokens`).
                policy_cleanup = relay_completion(
                    relay_client, request.ac_api_url, request.ac_api_key, cleanup_params(params)
                )
                opponent_cleanup = (
                    opponent_api(
                        opponent_client, opponent_url, opponent_key or "", cleanup_params(opponent_params(params))
                    )
                    if params.get("opponent_model")
                    else policy_cleanup
                )
                def make_complete(api: str, completion_params: dict[str, Any]) -> Any:
                    call = responses_completion if api == "responses" else relay_completion
                    return call(opponent_client, opponent_url, opponent_key or "", completion_params)

                table_key = f"{seed}:{sorted(policy)}"  # seeds each league seat's bot and noise, the same every replay
                players: list[Any] = [
                    lineup_player(
                        i,
                        lineup_seats[i],
                        rules=rollout.rules,
                        params=params,
                        key=f"{table_key}:{i}",
                        make_complete=make_complete,
                        timeout_seconds=opponent_timeout,
                        talk_lines=talk_lines,
                        talk_rate=talk_rate,
                    )
                    if i in lineup_seats
                    else ModelSeat(
                        f"seat{i}",
                        policy_complete if i in policy else opponent_complete,
                        rollout.rules,
                        model="ac2/policy" if i in policy else f"opponent/{params.get('opponent_model', 'policy')}",
                        timeout_seconds=decision_timeout if i in policy else opponent_timeout,
                        cleanup_complete=policy_cleanup if i in policy else opponent_cleanup,
                        system_hint=hint if i in policy else "",
                        enforced_cap=enforced if i in policy else None,
                        talk_lines=() if i in policy else talk_lines,
                        talk_rate=talk_rate,
                        talk_seed=seed,
                    )
                    for i in range(seats)
                ]
                if lineup is not None:  # league opponents only; the policy seat is never perturbed
                    players = with_noise(players, lineup_seats, lineup.get("noise") or {}, table_key)
                rollout.policy_seats = [f"seat{i}" for i in sorted(policy)]
                rollout.log = await play_session(rollout.id, seed, players, rollout.rules)
                if hint:  # so a hinted game can be told apart from a plain one (the grader reports its hash)
                    rollout.log["system_hint"] = hint
                if lineup is not None:
                    rollout.log["lineup"] = lineup_log(lineup, lineup_seats, players)
                label_folds(rollout.log, set(rollout.policy_seats))
                if params.get("ev_labels"):  # all-in EV for the policy seats, scored by grading.py
                    rollout.log = await asyncio.get_running_loop().run_in_executor(
                        label_pool(),
                        label_ev,
                        rollout.log,
                        set(rollout.policy_seats),
                        int(params.get("ev_variants", 8)),
                        int(params.get("ev_runouts", 32)),
                        bool(params.get("ev_redraw", False)),
                    )
            # The relay URL names the AC2 job the hand belongs to; keep it so concurrent runs can be told apart.
            job = re.search(r"/(job_[0-9a-f]+)/", request.ac_api_url)
            rollout.log["relay_job_id"] = job[1] if job else None
            runs.mkdir(parents=True, exist_ok=True)
            (runs / f"{rollout.id}.json").write_text(json.dumps(rollout.log, indent=1))
            rollout.status = "completed"
        except Exception as exc:  # reported to AC2 as a failed rollout
            logger.exception("rollout %s failed", rollout.id)
            rollout.status, rollout.error = "failed", f"{type(exc).__name__}: {str(exc)[:500]}"
            try:  # a record of the failure beside the game logs, which self-play's watchdog counts
                (runs / "failed").mkdir(parents=True, exist_ok=True)
                league_table = bool((request.task.env_params or {}).get("opponent_league"))
                record = {"rollout_id": rollout.id, "error": rollout.error, "opponent_league": league_table}
                (runs / "failed" / f"{rollout.id}.json").write_text(json.dumps(record))
            except OSError:
                pass

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "rollouts": {s: sum(r.status == s for r in rollouts.values()) for s in ("running",)}}

    @app.post("/submit")
    async def submit(
        request: SubmitRequest,
        authorization: str | None = Header(default=None),
        idempotency_key: str | None = Header(default=None),
    ) -> dict[str, str]:
        authorize(authorization)
        rollout_id = request.rollout_id or idempotency_key or f"table_{uuid.uuid4().hex}"
        if rollout_id in rollouts:
            return {"rollout_id": rollout_id}
        params = with_preset(request.task.env_params or {})  # `preset: "event"` applies the event rules
        if "seed" not in params:
            raise HTTPException(status_code=422, detail="env_params.seed is required")
        if not isinstance(params.get("system_hint", ""), str):
            raise HTTPException(status_code=422, detail="env_params.system_hint must be a string")
        talk = params.get("opponent_talk")
        if talk is not None and not (
            isinstance(talk, dict)
            and isinstance(talk.get("lines"), list)
            and talk["lines"]
            and all(isinstance(line, str) and line for line in talk["lines"])
            and isinstance(talk.get("rate"), int | float)
            and not isinstance(talk.get("rate"), bool)
            and 0 <= talk["rate"] <= 1
        ):
            raise HTTPException(
                status_code=422, detail='env_params.opponent_talk must be {"lines": [non-empty strings], "rate": 0-1}'
            )
        enforced = params.get("enforced_max_completion_tokens")
        if enforced is not None and (not isinstance(enforced, int) or isinstance(enforced, bool) or enforced < 1):
            raise HTTPException(status_code=422, detail="env_params.enforced_max_completion_tokens must be a positive integer")
        if SPOT_KEY in params and (problem := check_spot(params[SPOT_KEY])):
            raise HTTPException(status_code=422, detail=f"env_params.{SPOT_KEY} {problem}")
        if not isinstance(params.get("opponent_league", False), bool):
            raise HTTPException(status_code=422, detail="env_params.opponent_league must be true or false")
        try:
            explicit_lineup(params)  # env_params.opponent_seats and opponent_noise, when set
        except LeagueError as exc:
            raise HTTPException(status_code=422, detail=f"env_params: {exc}") from exc
        rollout = Rollout(id=rollout_id, rules=table_rules(params))
        rollouts[rollout_id] = rollout
        rollout.task = asyncio.create_task(run_table(rollout, request))
        logger.info("rollout %s: %d hands, relay %s", rollout_id, rollout.rules.total_hands, request.ac_api_url)
        return {"rollout_id": rollout_id}

    def lookup(rollout_id: str) -> Rollout:
        rollout = rollouts.get(rollout_id)
        if rollout is None:
            raise HTTPException(status_code=404, detail="unknown rollout")
        return rollout

    @app.post("/get_status")
    async def get_status(request: RolloutRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        authorize(authorization)
        rollout = lookup(request.rollout_id)
        status: dict[str, Any] = {"status": rollout.status}
        if rollout.error:
            status["error_message"] = rollout.error[:2000]
        if rollout.log:
            status["annotations"] = {"hands_played": len(rollout.log["hands"])}
        return status

    @app.post("/collect_outputs")
    async def collect_outputs(
        request: RolloutRequest, authorization: str | None = Header(default=None)
    ) -> dict[str, Any]:
        authorize(authorization)
        rollout = lookup(request.rollout_id)
        log = _bounded(rollout.log) if rollout.log else None
        return {
            "log": log,
            "rules": rules_output(rollout.rules),
            "policy_seats": rollout.policy_seats,
            "error": rollout.error,
        }

    async def run_grading(grading: Grading, request: SubmitGradingRequest) -> None:
        grading.status = "running"
        try:
            params = check_grader_params(request.task.grader_params)  # the sample grader: grading.py
            rollout = rollouts.get(request.rollout_id)
            if request.rollout_status != "completed":
                grading.score = failure_score(f"AC2 reported the rollout {request.rollout_status}.")
            elif rollout is None:  # e.g. the harness restarted since the table was played: nothing to score
                raise GradingError("unknown rollout; the harness has no log for it")
            else:
                if rollout.task is not None:
                    await asyncio.shield(rollout.task)
                if rollout.status != "completed" or not rollout.log:
                    grading.score = failure_score(f"The table ended {rollout.status}: {rollout.error}")
                elif rollout.log.get("kind") == "solver_spot":
                    grading.score = bounded(grade_spot(rollout.log))
                else:
                    grading.score = bounded(grade_log(rollout.log, rollout.policy_seats, params))
            grading.status = "completed"
        except Exception as exc:  # AC2 drops the sample (GRADER_ERROR)
            logger.exception("grading %s failed", request.rollout_id)
            grading.status, grading.error = "failed", f"{type(exc).__name__}: {str(exc)[:500]}"

    @app.post("/submit_grading")
    async def submit_grading(
        request: SubmitGradingRequest,
        authorization: str | None = Header(default=None),
        idempotency_key: str | None = Header(default=None),
    ) -> dict[str, str]:
        """One grading per rollout: a replay (same rollout_id) returns the grading already started."""
        authorize(authorization)
        if idempotency_key is not None and idempotency_key != f"{request.rollout_id}:grading":
            raise HTTPException(status_code=422, detail="Idempotency-Key must be '<rollout_id>:grading'")
        if request.rollout_id not in gradings:
            grading = Grading()
            gradings[request.rollout_id] = grading
            grading.task = asyncio.create_task(run_grading(grading, request))
        return {"rollout_id": request.rollout_id}

    def lookup_grading(rollout_id: str) -> Grading:
        grading = gradings.get(rollout_id)
        if grading is None:
            raise HTTPException(status_code=404, detail="unknown grading")
        return grading

    @app.post("/get_grading_status")
    async def get_grading_status(
        request: GradingRequest, authorization: str | None = Header(default=None)
    ) -> dict[str, Any]:
        authorize(authorization)
        grading = lookup_grading(request.rollout_id)
        status: dict[str, Any] = {"status": grading.status}
        if grading.error:
            status["error_message"] = grading.error[:2000]
        if grading.score is not None:
            status["annotations"] = {"score": grading.score["score"]}
        return status

    @app.post("/collect_score")
    async def collect_score(
        request: GradingRequest, authorization: str | None = Header(default=None)
    ) -> dict[str, Any]:
        authorize(authorization)
        grading = lookup_grading(request.rollout_id)
        if grading.status != "completed" or grading.score is None:
            raise HTTPException(status_code=409, detail=f"grading is {grading.status}")
        return grading.score

    return app


app = create_app()
