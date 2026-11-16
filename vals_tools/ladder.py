"""The overnight self-play ladder: train against a league of frozen copies, refresh it, recover, and report.

    uv run python -m vals_tools.ladder --until 08:00                      # the night (ops/overnight.sh runs this)
    uv run python -m vals_tools.ladder --until 08:00 --start-checkpoint ckpt_xxx --lr 1e-6
    uv run python -m vals_tools.ladder --status                           # the saved state's latest events

It runs unattended until the deadline and never waits for a person. Each poll (every 2 minutes) it:

1. keeps one GRPO run going on `pokerbots-league-cap3000-train` (`rl/train.py --no-eval`: 6 GPUs). A run that
   finishes early is continued from its last checkpoint (a warm start); one that fails is restarted;
2. fills its 3 slots (10 GPUs: 6 training, base, 3 frozen copies) with copies of the starting model until the run's
   first checkpoint, then deploys each newest checkpoint as `sp-<run6>-step<N>` (1 GPU), checks it still calls
   `submit_action` in a short local game against bots, adds it to the league file the harness draws opponents from,
   and plays it on the yardstick (200 select deals, paired by deal with base on the same deals);
3. keeps the newest copy, the second newest and the best on the yardstick, and frees a slot by first taking a copy
   out of the league, waiting 25 minutes for tables that drew it to finish, then deleting the deployment;
4. watches the training hands (the harness's league tables) over the last 30 minutes against base's own behavior,
   and acts when a rule trips twice in a row. Opponents timing out: more weight to bots, then a restart at half the
   concurrency. The trained seat misbehaving or two checkpoints in a row failing the yardstick: a rollback, which stops
   the run, retires the checkpoints since the last one that passed, and warm-starts from that one at half the
   learning rate. After 2 rollbacks it stops training early;
5. at the deadline: stops training, then plays the best 3 checkpoints on the yardstick, the newest one and base on the
   final dataset (held-out deals) and writes the report (data/launches/ladder_report.txt).

It never deletes anything but its own `sp-` deployments, never touches base, never enters the arena, and never
deploys past the GPU quota. Everything is saved to the state file after every poll, so a restart resumes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "harness" / "runs"
STATE = ROOT / "data" / "launches" / "ladder.json"
LEAGUE_FILE = ROOT / "data" / "league" / "league.json"
LOGS = ROOT / "data" / "launches" / "ladder"

QUOTA = 10
TRAIN_GPUS = 6  # rl/train.py --no-eval: 4 training + 1 rollout replica (2 GPUs)
MAX_OWN = 3  # our own frozen copies deployed at once
POLL_S = 120
DRAIN_S = 25 * 60  # > the harness's 20-minute lineup pin
WINDOW_S = 30 * 60
WATCH_EVERY_S = 15 * 60
MIN_DECISIONS = 150  # a window with fewer policy decisions judges nothing about the trained seat
MAX_ROLLBACKS = 2
MAX_RESTARTS = 3
CHECK_HANDS = 6
CHECK_MAX_FAILED = 0.10
OPPONENT_FAILED_MAX = 0.15  # opponents timing out or failing: their tables teach nothing
EVAL_OPPONENT_FAILED_MAX = 0.10  # an eval whose opponents fail more than this is not judged
BOT_SHARE, BOT_SHARE_MAX = 30.0, 70.0
MIN_CONCURRENCY = 8
ROUND_STEPS = 20  # steps per run once checkpoints have to wait for their run to end before they deploy

TRAIN_PREFIX = "pokerbots-league-cap3000"
YARDSTICK = "pokerbots-select200"
FINAL_DATASET = "pokerbots-eval1000"
BASE = "vals-hackathon/base"
HOUSE = "vals-hackathon/ac-vals-model"
BASE_CATALOG = "Qwen/Qwen3.6-35B-A3B"
DEFAULT_BASELINE = {"failed": 0.02, "vpip": 0.22, "allin": 0.02}
NOISE = {"size_jitter": 0.25, "action_epsilon": 0.03}
DONE = ("succeeded", "failed", "stopped", "cancelled", "canceled", "error")
NAME = re.compile(r"sp-([0-9a-f]{6}|start\d)-step(\d+)")

PERSONAS = ("tight_aggressive", "loose_aggressive", "tight_passive", "loose_passive", "trappy", "talker")
BOT_STYLES = ("nit", "tag", "lag", "station", "loose_passive", "maniac",  # ours
              "hybrid_tag", "hybrid_lag", "tag_punisher", "wide_defender", "yosief", "range_gto")  # adapted


def finished(status: Any) -> bool:
    return any(word in str(status or "").lower() for word in DONE)


# ---------------------------------------------------------------- pure decisions (tested in tests/test_ladder.py)


def league_entries(live: list[dict], base: str, persona_model: str, house: str | None = None,
                   house_weight: float = 0.0, bot_share: float = BOT_SHARE) -> list[dict]:
    """The league's entries, weights summing to 100. `live` is our frozen copies in the league, newest first, each
    {"id", "deployment", "kind": "ckpt" | "start"}: the newest checkpoint gets the most weight."""
    others: list[dict] = []
    if live:
        frozen = 45.0
        if live[0]["kind"] == "ckpt" and len(live) > 1:
            others.append({"id": live[0]["id"], "model": live[0]["deployment"], "weight": 25.0})
            others += [{"id": c["id"], "model": c["deployment"], "weight": 20.0 / (len(live) - 1)} for c in live[1:]]
        else:
            others += [{"id": c["id"], "model": c["deployment"], "weight": frozen / len(live)} for c in live]
        others.append({"id": "base", "model": base, "weight": 15.0})
    else:
        others.append({"id": "base", "model": base, "weight": 60.0})
    others += [{"id": f"persona-{p}", "model": persona_model, "persona": p, "weight": 10.0 / len(PERSONAS)}
               for p in PERSONAS]
    if house and house_weight > 0:
        others.append({"id": "house", "model": house, "weight": house_weight})
    scale = (100.0 - bot_share) / sum(e["weight"] for e in others)
    entries = [e | {"weight": round(e["weight"] * scale, 3)} for e in others]
    entries += [{"id": f"bot-{b}", "bot": b, "weight": round(bot_share / len(BOT_STYLES), 3)} for b in BOT_STYLES]
    return entries


def keep_set(copies: list[dict]) -> set[str]:
    """Which copies (each {"id", "n", "bad", "kind", "yardstick"}) to keep: the newest, the second newest and the best
    on the yardstick, none of them bad; copies of the starting model only until checkpoints replace them."""
    good = sorted((c for c in copies if not c.get("bad")), key=lambda c: c["n"], reverse=True)
    keep = {c["id"] for c in good[:2]}
    scored = [c for c in good if (c.get("yardstick") or {}).get("passed")]
    if scored:
        keep.add(max(scored, key=lambda c: c["yardstick"]["diff"])["id"])
    return keep


def choose_evictee(deployed: list[dict], incoming: dict | None) -> dict | None:
    """The deployed copy to retire to make room for `incoming`: a bad one first, then a copy of the starting model,
    then the oldest one not kept."""
    pool = deployed + ([incoming] if incoming else [])
    keep = keep_set(pool)
    candidates = [c for c in deployed if c["id"] not in keep or c.get("bad") or c.get("kind") == "start"]
    if incoming and incoming.get("kind") == "start":
        candidates = [c for c in candidates if c.get("bad")]  # a start copy never displaces anything but a bad copy
    candidates.sort(key=lambda c: (not c.get("bad"), c.get("kind") != "start", c["n"]))
    return candidates[0] if candidates else None


def free_gpus(deployments: int, training: bool, quota: int = QUOTA) -> int:
    return quota - deployments - (TRAIN_GPUS if training else 0)


def may_delete(name: str, our_runs: set[str]) -> bool:
    """Only our own copies: `sp-<run6>-step<N>` of one of our training runs, or `sp-start<k>-step0`."""
    found = NAME.fullmatch(name)
    if not found:
        return False
    return found[1].startswith("start") or found[1] in {r.removeprefix("train_")[-6:] for r in our_runs}


def yardstick_passed(vs_base: dict | None) -> bool:
    """Failed means more than 1 BB/hand worse than base with the 95% interval wholly below 0."""
    if not vs_base or vs_base.get("diff") is None:
        return True  # nothing to judge on
    ci = vs_base.get("ci95")
    ci = 0.0 if ci is None or ci != ci else ci  # NaN with one pair
    return not (vs_base["diff"] < -1.0 and vs_base["diff"] + ci < 0)


def tripped(stats: dict, baseline: dict) -> list[str]:
    """The rules this window breaks (see the module docstring)."""
    out = []
    if (stats.get("opponent_decisions") or 0) >= 50 and (stats.get("opponent_failed") or 0) > OPPONENT_FAILED_MAX:
        out.append("opponent_failures")
    if stats.get("tables", 0) >= 20 and (stats.get("rollout_failed_share") or 0) > 0.10:
        out.append("rollout_failures")
    if stats.get("decisions", 0) < MIN_DECISIONS:
        return out
    if (stats.get("failed") or 0) > max(0.05, 2 * baseline["failed"]):
        out.append("failed")
    vpip = stats.get("vpip")
    if vpip is not None and (vpip < 0.5 * baseline["vpip"] or vpip > 2 * baseline["vpip"]):
        out.append("vpip")
    if (stats.get("allin") or 0) > max(0.03, 3 * baseline["allin"]):
        out.append("allin")
    return out


def allin_rate(logs: list[dict]) -> float | None:
    """Share of the policy's hands in which it put its whole stack in."""
    hands = [[a for a in h["actions"] if a["model_response"].get("model_name") == "ac2/policy"]
             for log in logs for h in log["hands"]]
    hands = [h for h in hands if h]
    if not hands:
        return None
    shoved = sum(any(a.get("stack_after") == 0 and a["action"]["type"] in ("call", "bet", "raise") for a in h)
                 for h in hands)
    return round(shoved / len(hands), 3)


def opponent_failures(logs: list[dict]) -> tuple[int, float | None]:
    """(decisions, failed share) of the opponent model seats: replies that failed or timed out (bots excluded)."""
    picks = [a["model_response"] for log in logs for h in log["hands"] for a in h["actions"]
             if str(a["model_response"].get("model_name", "")).startswith("opponent/")]
    if not picks:
        return 0, None
    bad = sum(bool(r.get("format_error") or r.get("afk")) for r in picks)
    return len(picks), round(bad / len(picks), 3)


def window_stats(logs: list[dict], failures: int) -> dict:
    from vals_tools.hint_report import policy_stats

    stats = policy_stats(logs)
    stats["allin"] = allin_rate(logs)
    stats["opponent_decisions"], stats["opponent_failed"] = opponent_failures(logs)
    stats["tables"] = len(logs) + failures
    stats["rollout_failures"] = failures
    stats["rollout_failed_share"] = round(failures / stats["tables"], 3) if stats["tables"] else 0.0
    return stats


def deadline(text: str, now: datetime) -> datetime:
    """HH:MM (the next such local time) or an ISO date-time."""
    if re.fullmatch(r"\d{1,2}:\d{2}", text):
        hour, minute = map(int, text.split(":"))
        at = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        return at if at > now else at + timedelta(days=1)
    at = datetime.fromisoformat(text)
    return at if at.tzinfo else at.astimezone()


def final_candidates(checkpoints: dict[str, dict]) -> list[str]:
    """The best 3 on the yardstick plus the newest that passed its check, never bad, never a start copy."""
    good = [c for c in checkpoints.values() if not c.get("bad") and c.get("check_passed") and c.get("kind") == "ckpt"]
    scored = sorted((c for c in good if (c.get("yardstick") or {}).get("passed")),
                    key=lambda c: c["yardstick"]["diff"], reverse=True)
    picks = [c["id"] for c in scored[:3]]
    newest = max(good, key=lambda c: c["n"], default=None)
    if newest and newest["id"] not in picks:
        picks.append(newest["id"])
    return picks


# ---------------------------------------------------------------- the ladder


class Ladder:
    def __init__(self, args: argparse.Namespace) -> None:
        from ac2.sdk import Client

        self.args = args
        self.client = Client(project=args.project)
        self.procs: dict[str, subprocess.Popen] = {}
        self._rows: dict[str, dict] = {}
        self.state = self.load()

    # -- state --

    def load(self) -> dict:
        if self.args.state.exists():
            state = json.loads(self.args.state.read_text())
            print(f"resuming from {self.args.state}", flush=True)
        else:
            state = {"started": datetime.now().astimezone().isoformat(), "runs": [], "checkpoints": {},
                     "league_version": 0, "rollbacks": 0, "restarts": 0, "harness_restarts": 0, "phase": "training",
                     "watch": {"trips": {}, "last": 0, "history": []}, "events": [], "next_n": 0,
                     "bot_share": BOT_SHARE, "base_evals": {}}
        # A resumed ladder keeps its deadline (after 08:00, "--until 08:00" would mean tomorrow) unless told otherwise.
        if self.args.until and ("until" not in state or self.args.new_deadline):
            state["until"] = deadline(self.args.until, datetime.now().astimezone()).isoformat()
        cfg = state.setdefault("config", {})
        for key in ("start_checkpoint", "lr", "steps", "concurrency", "persona_model", "base", "house", "house_weight",
                    "manage_harness", "yardstick", "final_dataset", "start_copies"):
            value = getattr(self.args, key)
            if value is not None or key not in cfg:
                cfg[key] = value
        for key in ("base_yardstick_eval", "base_final_eval"):  # given ids win over ones the ladder launched
            if getattr(self.args, key):
                state["base_evals"][key] = getattr(self.args, key)
        if not state["checkpoints"] and cfg["start_copies"]:
            for k in range(1, cfg["start_copies"] + 1):  # fill idle slots until the first checkpoint
                cid = f"start{k}"
                state["checkpoints"][cid] = {"id": cid, "kind": "start", "train": None, "step": 0, "n": -k,
                                             "name": f"sp-start{k}-step0", "state": "pending",
                                             "source": cfg["start_checkpoint"]}
        return state

    def save(self) -> None:
        self.args.state.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.args.state.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1, default=str))
        os.replace(tmp, self.args.state)

    def log(self, message: str) -> None:
        line = f"{datetime.now().astimezone():%Y-%m-%d %H:%M:%S} {message}"
        print(line, flush=True)
        self.state["events"] = (self.state["events"] + [line])[-500:]

    # -- AC2 helpers --

    def deployment_names(self) -> dict[str, Any]:
        return {getattr(d, "name", None): d for d in self.client.inference.list()}

    def run(self) -> dict | None:
        return self.state["runs"][-1] if self.state["runs"] else None

    def training(self) -> bool:
        run = self.run()
        return bool(run) and not finished(run.get("status"))

    def our_runs(self) -> set[str]:
        return {r["train"] for r in self.state["runs"] if r.get("train")}

    def model(self, name: str) -> str:
        return f"{self.args.project}/{name}"

    def launch_eval(self, dataset: str, deployment: str, parallel: int) -> str:
        from vals_tools.eval import eval_config

        launched = asyncio.run(self.client.eval.run(eval_config(dataset, deployment=deployment, max_parallel=parallel)))
        return getattr(launched, "eval_id", str(launched))

    def rows(self, eval_id: str) -> dict:
        from vals_tools.score import score_rows

        if eval_id not in self._rows:
            self._rows[eval_id] = score_rows(self.client, eval_id=eval_id)
        return self._rows[eval_id]

    def eval_opponents_ok(self, eval_id: str) -> tuple[bool, float | None]:
        """Whether an eval's opponents mostly answered (the harness's logs of that eval's AC2 job)."""
        from vals_tools.hint_report import logs_by_job

        job = self.client.eval.get(eval_id).job_id
        _, share = opponent_failures(logs_by_job(RUNS, {job}).get(job, []))
        return share is None or share <= EVAL_OPPONENT_FAILED_MAX, share

    # -- the loop --

    def loop(self) -> None:
        self.log(f"ladder up: deadline {self.state['until']}, phase {self.state['phase']}")
        while self.state["phase"] != "done":
            try:
                self.tick()
            except Exception:  # never exit on an error: log it and try again next poll
                self.log("error in poll:\n" + traceback.format_exc()[-2000:])
            self.save()
            if self.state["phase"] != "done":
                time.sleep(self.args.poll)
        self.save()

    def tick(self) -> None:
        if self.state["phase"] == "training":
            if datetime.now().astimezone() >= datetime.fromisoformat(self.state["until"]):
                self.log("deadline reached: stopping training")
                self.begin_final()
            else:
                self.train_step()
        if self.state["phase"] == "final":
            self.final_step()

    # -- the league --

    def ensure_league(self) -> None:
        live = sorted((c for c in self.state["checkpoints"].values() if c["state"] == "live" and not c.get("bad")),
                      key=lambda c: c["n"], reverse=True)
        cfg = self.state["config"]
        entries = league_entries(
            [{"id": c["name"], "deployment": self.model(c["name"]), "kind": c["kind"]} for c in live],
            cfg["base"], cfg["persona_model"], cfg["house"], cfg["house_weight"], self.state["bot_share"],
        )
        try:
            current = json.loads(LEAGUE_FILE.read_text())
        except (OSError, ValueError):
            current = None
        if current and current.get("entries") == entries:
            return
        self.state["league_version"] += 1
        league = {"version": self.state["league_version"], "noise": NOISE, "entries": entries}
        LEAGUE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = LEAGUE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(league, indent=1))
        os.replace(tmp, LEAGUE_FILE)
        self.log(f"league v{league['version']}: copies {[c['name'] for c in live]}, bots {self.state['bot_share']}%")

    # -- training runs --

    def launch_run(self, start: str | None, lr: float | None, why: str, concurrency: int | None = None) -> None:
        """Launch a run now, or keep it as wanted (`state["want_run"]`) and retry every poll until it launches."""
        cfg = self.state["config"]
        concurrency = concurrency or (self.run() or {}).get("concurrency") or cfg.get("concurrency")
        self.state["want_run"] = {"start": start, "lr": lr, "why": why, "concurrency": concurrency}
        steps = min(cfg["steps"], ROUND_STEPS) if self.state.get("rounds") else cfg["steps"]
        cmd = ["uv", "run", "python", "rl/train.py", "--prefix", TRAIN_PREFIX, "--no-eval",
               "--steps", str(steps), "--checkpoint-every", "10", "--keep-checkpoints", "30",
               "--name", f"selfplay-{datetime.now():%m%d-%H%M}-{len(self.state['runs']) + 1}"]
        if start:
            cmd += ["--checkpoint", start]
        if lr:
            cmd += ["--lr", f"{lr:.3g}"]
        if concurrency:
            cmd += ["--concurrency", str(concurrency)]
        out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
        found = re.findall(r'"train_id": "(train_[0-9a-f]+)"', out.stdout)
        if out.returncode or not found:
            self.log(f"launching a run failed ({why}): {(out.stderr or out.stdout)[-600:]}")
            return
        run = {"train": found[-1], "start": start, "lr": lr, "why": why, "status": "queued",
               "started_at": time.time(), "concurrency": concurrency}
        self.state["runs"].append(run)
        self.state.pop("want_run", None)
        self.state["watch"]["trips"] = {}
        self.log(f"launched {run['train']} ({why}) from {start or 'the base model'}, lr {lr or 'AC2 default'}, "
                 f"concurrency {concurrency or 'AC2 default'}")

    def learning_rate(self, train: str) -> float | None:
        """The run's learning rate from its step metrics, when AC2 reports one (its default isn't in the SDK)."""
        try:
            metrics = self.client.train.get_step_metrics(train_job_id=train)
        except Exception:
            return None
        for key, rows in metrics.items():
            if re.search(r"(^|[/_.])(lr|learning_rate)$", key) and rows:
                value = rows[-1].get("value", rows[-1].get(key))
                if isinstance(value, int | float) and value > 0:
                    return float(value)
        return None

    def newest_good(self, train: str | None) -> dict | None:
        return max((c for c in self.state["checkpoints"].values()
                    if c["kind"] == "ckpt" and c["train"] == train and not c.get("bad")),
                   key=lambda c: c["n"], default=None)

    def train_step(self) -> None:
        self.ensure_league()
        if self.state.get("want_run"):  # a launch that failed earlier: try again
            self.launch_run(**self.state["want_run"])
        run = self.run()
        if run is None:
            if not self.state.get("want_run"):
                cfg = self.state["config"]
                self.launch_run(cfg["start_checkpoint"], cfg["lr"], "first run")
            self.pipeline()
            return
        if not finished(run.get("status")):
            status = str(getattr(self.client.train.get(run["train"]), "status", "") or "").lower()
            if status != run.get("status"):
                self.log(f"{run['train']}: {status}")
                run["status"] = status
        self.collect_checkpoints(run)
        if finished(run["status"]):
            self.after_run(run)
        self.pipeline()
        if self.training():
            self.watch(self.run())

    def after_run(self, run: dict) -> None:
        """A run ended: one we stopped is already replaced; one that finished is continued; one that failed restarts."""
        if run.get("handled"):
            return
        run["handled"] = True
        if run.get("stopping"):
            return
        newest = self.newest_good(run["train"])
        start = newest["id"] if newest else run.get("start")
        if "succeeded" in run["status"]:
            # In rounds, the run's last checkpoint joins the league before the next run starts (up to 45 minutes).
            waiting = newest and newest["state"] in ("pending", "deploying", "checking")
            if self.state.get("rounds") and waiting and time.time() - run.setdefault("ended_at", time.time()) < 45 * 60:
                run["handled"] = False
                return
            self.launch_run(start, run.get("lr"), f"continuing {run['train']}")
            return
        self.state["restarts"] += 1
        if self.state["restarts"] > MAX_RESTARTS:
            self.log(f"{run['train']} ended {run['status']}; {MAX_RESTARTS} restarts used: stopping training")
            self.begin_final()
            return
        self.launch_run(start, run.get("lr"), f"restart after {run['train']} ended {run['status']}")

    def stop_run(self, why: str) -> dict | None:
        run = self.run()
        if run and not finished(run.get("status")):
            run["stopping"], run["handled"] = True, True
            try:
                self.client.train.stop(run["train"], stop_evals=True)
                self.log(f"stopped {run['train']}: {why}")
            except Exception as exc:
                self.log(f"stopping {run['train']} failed: {exc}")
        return run

    def collect_checkpoints(self, run: dict) -> None:
        from vals_tools.after_training import deployment_name

        try:
            found = sorted(self.client.train.checkpoints(run["train"]), key=lambda c: c.iter_num)
        except Exception as exc:
            self.log(f"listing checkpoints of {run['train']} failed: {exc}")
            return
        for ckpt in found:
            if ckpt.id in self.state["checkpoints"]:
                continue
            for older in self.state["checkpoints"].values():  # only the newest waiting checkpoint gets deployed
                if older["state"] == "pending":
                    older["state"] = "skipped"
            self.state["checkpoints"][ckpt.id] = {
                "id": ckpt.id, "kind": "ckpt", "train": run["train"], "step": ckpt.iter_num, "n": self.state["next_n"],
                "name": deployment_name("sp", run["train"], ckpt.iter_num), "state": "pending",
            }
            self.state["next_n"] += 1
            self.log(f"new checkpoint {ckpt.id} ({run['train']} step {ckpt.iter_num})")

    # -- deploy, check, league, yardstick --

    def spawn(self, key: str, cmd: list[str], cwd: Path) -> None:
        LOGS.mkdir(parents=True, exist_ok=True)
        out = open(LOGS / f"{key}.log", "w")  # noqa: SIM115 - the child process owns it
        self.procs[key] = subprocess.Popen(cmd, cwd=cwd, stdout=out, stderr=subprocess.STDOUT, text=True)

    def step_done(self, key: str) -> tuple[bool, int | None, str]:
        """(done, exit code, output) of a spawned step; one lost to a ladder restart reads as done with code None."""
        path = LOGS / f"{key}.log"
        proc = self.procs.get(key)
        if proc is not None and proc.poll() is None:
            return False, None, ""
        text = path.read_text(errors="replace") if path.exists() else ""
        if proc is None:
            return True, None, text
        self.procs.pop(key)
        return True, proc.returncode, text

    def check(self, c: dict) -> None:
        c["state"] = "checking"
        self.spawn(f"check-{c['name']}", ["uv", "run", "python", "play_local.py", "--model", self.model(c["name"]),
                                         "--hands", str(CHECK_HANDS), "--opponents", "bots", "--out", "runs/checks"],
                   ROOT / "harness")

    def go_live(self, c: dict) -> None:
        c["state"] = "live"
        self.ensure_league()
        if c["kind"] == "ckpt":
            try:
                c["yardstick"] = {"eval": self.launch_eval(self.state["config"]["yardstick"], self.model(c["name"]), 16),
                                  "status": "running"}
                self.log(f"yardstick {c['yardstick']['eval']} for {c['name']}")
            except Exception as exc:
                c["yardstick"] = {"status": "not run", "error": str(exc)[:300]}
                self.log(f"yardstick for {c['name']} not launched: {exc}")

    def pipeline(self) -> None:
        from vals_tools.after_training import parse_check

        cps = self.state["checkpoints"]
        names = self.deployment_names()
        for c in cps.values():  # retired copies: delete once drained
            if c["state"] == "retired" and time.time() - c["retired_at"] > DRAIN_S:
                self.delete(c, names)
        self.base_yardstick()
        self.poll_yardsticks()
        for c in cps.values():  # one deploy or check at a time
            if c["state"] == "deploying":
                done, code, text = self.step_done(f"deploy-{c['name']}")
                if not done:
                    return
                if code == 0 or (code is None and c["name"] in names):
                    if c["kind"] == "start" and not c.get("source"):  # a copy of base itself: nothing to check
                        c["check_passed"] = True
                        self.log(f"{c['name']} (a copy of base) is up: joins the league")
                        self.go_live(c)
                    else:
                        self.check(c)
                    return
                self.log(f"deploying {c['name']} failed: {text[-400:]}")
                if c["name"] in names:
                    self.delete(c, names)
                run = self.run()
                if c["kind"] == "ckpt" and run and c["train"] == run["train"] and self.training():
                    # Maybe a checkpoint can't be served while its run is going: after two such failures, switch to
                    # rounds (each run ends, its last checkpoint is deployed, then the next run starts from it).
                    c["deploy_failures"] = c.get("deploy_failures", 0) + 1
                    c["state"] = "pending"
                    if c["deploy_failures"] >= 2 and not self.state.get("rounds"):
                        self.state["rounds"] = True
                        self.log("checkpoints don't deploy mid-run: switching to rounds of "
                                 f"{ROUND_STEPS} steps, deploying between runs")
                    if self.state.get("rounds"):
                        c["state"] = "pending"  # deployed when the run ends (see after_run)
                        c["wait_for_run_end"] = True
                else:
                    c["state"] = "failed"
                return
            if c["state"] == "checking":
                done, code, text = self.step_done(f"check-{c['name']}")
                if not done:
                    return
                found = parse_check(text)
                if code is None and not found:  # lost to a ladder restart: run it again
                    self.check(c)
                    return
                c["check"] = {"decisions": found[0], "failed": found[1]} if found else {"error": text[-300:]}
                if found and found[0] and found[1] / found[0] <= CHECK_MAX_FAILED:
                    c["check_passed"] = True
                    self.log(f"{c['name']} passed its check ({found[1]}/{found[0]} failed): joins the league")
                    self.go_live(c)
                else:
                    self.log(f"{c['name']} failed its check {c['check']}: not used")
                    c["state"] = "rejected"
                    self.delete(c, names)
                return
        run = self.run()
        mid_run = self.state.get("rounds") and self.training() and run
        pending = [c for c in cps.values() if c["state"] == "pending" and not c.get("bad")
                   and not (mid_run and c["kind"] == "ckpt" and c["train"] == run["train"])]  # rounds: after the run
        if not pending:
            return
        ckpts = [c for c in pending if c["kind"] == "ckpt"]
        incoming = max(ckpts, key=lambda c: c["n"]) if ckpts else max(pending, key=lambda c: c["n"])
        mine = [c for c in cps.values() if c["state"] in ("live",)]
        draining = [c for c in cps.values() if c["state"] == "retired"]
        free = free_gpus(len(names), self.training())
        if free >= 1 and len(mine) + len(draining) < MAX_OWN:
            self.deploy(incoming, names)
            return
        if draining or (free < 1 and len(mine) + len(draining) < MAX_OWN):
            return  # a slot is draining, or the GPUs are taken by something else: wait
        evictee = choose_evictee(mine, incoming)
        if evictee is None:
            if incoming["kind"] == "ckpt":
                self.log(f"no slot for {incoming['name']}: keeping the current copies")
            incoming["state"] = "skipped"
            return
        evictee["state"], evictee["retired_at"] = "retired", time.time()
        self.log(f"retiring {evictee['name']} to make room for {incoming['name']}")
        self.ensure_league()

    def deploy(self, c: dict, names: dict) -> None:
        if c["name"] in names:  # already up (e.g. a ladder restart mid-deploy)
            c["state"] = "deploying"
            return
        c["state"] = "deploying"
        source = c["id"] if c["kind"] == "ckpt" else c.get("source")
        how = ["--checkpoint", source] if source else ["--model", BASE_CATALOG]
        self.log(f"deploying {c['name']} from {source or BASE_CATALOG}")
        self.spawn(f"deploy-{c['name']}", [sys.executable, "-m", "vals_tools.deploy", "--project", self.args.project,
                                           "--name", c["name"], *how], ROOT)

    def delete(self, c: dict, names: dict) -> None:
        if not may_delete(c["name"], self.our_runs()):
            self.log(f"refusing to delete {c['name']}: not one of ours")
            return
        deployment = names.get(c["name"])
        if deployment is not None:
            try:
                self.client.inference.delete(deployment.id)
                self.log(f"deleted {c['name']} ({deployment.id})")
            except Exception as exc:
                self.log(f"deleting {c['name']} failed: {exc}")
                return
        if c["state"] not in ("rejected", "failed"):
            c["state"] = "deleted"

    def base_yardstick(self) -> None:
        """Base on the yardstick's deals, the comparison every checkpoint's yardstick pairs with."""
        evals = self.state["base_evals"]
        if "base_yardstick_eval" not in evals:
            try:
                evals["base_yardstick_eval"] = self.launch_eval(self.state["config"]["yardstick"], self.state["config"]["base"], 16)
                self.log(f"base yardstick {evals['base_yardstick_eval']}")
            except Exception as exc:
                self.log(f"base yardstick not launched: {exc}")

    def base_ready(self, key: str) -> bool:
        eval_id = self.state["base_evals"].get(key)
        return bool(eval_id) and finished(getattr(self.client.eval.get(eval_id), "status", ""))

    def poll_yardsticks(self) -> None:
        from vals_tools.score import summarize

        if not self.base_ready("base_yardstick_eval"):
            return
        base = self.rows(self.state["base_evals"]["base_yardstick_eval"])
        changed = False
        for c in self.state["checkpoints"].values():
            y = c.get("yardstick") or {}
            if y.get("status") != "running":
                continue
            status = str(getattr(self.client.eval.get(y["eval"]), "status", "") or "").lower()
            if not finished(status):
                continue
            ok, share = self.eval_opponents_ok(y["eval"])
            summary = summarize(self.rows(y["eval"]), base)
            y.update(status=status, summary=summary, diff=(summary.get("vs_base") or {}).get("diff"),
                     opponent_failed=share, passed=yardstick_passed(summary.get("vs_base")) if ok else None)
            verdict = {True: "pass", False: "FAIL", None: f"not judged (opponents failed {share})"}[y["passed"]]
            self.log(f"yardstick for {c['name']}: {summary.get('vs_base')} -> {verdict}")
            changed = True
        if changed:
            judged = sorted((c for c in self.state["checkpoints"].values()
                             if (c.get("yardstick") or {}).get("passed") is not None and not c.get("bad")),
                            key=lambda c: c["n"])
            run = self.run()
            if (self.state["phase"] == "training" and len(judged) >= 2 and judged[-1]["yardstick"]["passed"] is False
                    and judged[-2]["yardstick"]["passed"] is False and run and judged[-1]["train"] == run["train"]):
                self.rollback("two checkpoints in a row failed the yardstick")

    # -- watchdog --

    def window_logs(self, since: float) -> tuple[list[dict], int]:
        logs, failures = [], 0
        for path in RUNS.glob("*.json"):
            try:
                if path.stat().st_mtime < since:
                    continue
                log = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if "lineup" in log and log["lineup"].get("version") is not None:  # league tables are training tables
                logs.append(log)
        for path in (RUNS / "failed").glob("*.json"):
            try:
                if path.stat().st_mtime >= since and json.loads(path.read_text()).get("opponent_league"):
                    failures += 1
            except (OSError, ValueError):
                continue
        return logs, failures

    def baseline(self) -> dict:
        """Base's own behavior on the yardstick (once that eval is done), else defaults."""
        if "baseline" in self.state:
            return self.state["baseline"]
        if not self.base_ready("base_yardstick_eval"):
            return DEFAULT_BASELINE
        base = dict(DEFAULT_BASELINE, source="defaults")
        try:
            from vals_tools.hint_report import logs_by_job, policy_stats

            job = self.client.eval.get(self.state["base_evals"]["base_yardstick_eval"]).job_id
            logs = logs_by_job(RUNS, {job}).get(job, [])
            if logs:
                stats = policy_stats(logs)
                base = {"failed": stats["failed"] or 0.0, "vpip": stats["vpip"] or DEFAULT_BASELINE["vpip"],
                        "allin": allin_rate(logs) or DEFAULT_BASELINE["allin"], "source": f"base yardstick, {len(logs)} hands"}
        except Exception as exc:
            self.log(f"baseline unavailable ({exc}); using defaults")
        self.state["baseline"] = base
        self.log(f"baseline: {base}")
        return base

    def watch(self, run: dict) -> None:
        watch = self.state["watch"]
        if time.time() - watch["last"] < WATCH_EVERY_S:
            return
        watch["last"] = time.time()
        logs, failures = self.window_logs(max(time.time() - WINDOW_S, run["started_at"]))
        stats = window_stats(logs, failures)
        rules = tripped(stats, self.baseline())
        watch["history"] = (watch["history"] + [{"at": time.time(), "train": run["train"], "stats": stats,
                                                 "tripped": rules}])[-200:]
        self.log(f"watch {run['train']}: {stats['tables']} tables, policy failed {stats['failed']}, vpip {stats['vpip']}, "
                 f"allin {stats['allin']}, opponents failed {stats['opponent_failed']}, rollouts failed "
                 f"{stats['rollout_failures']} -> {rules or 'ok'}")
        trips = {rule: watch["trips"].get(rule, 0) + 1 for rule in rules}
        watch["trips"] = trips
        if trips.get("rollout_failures", 0) >= 2:
            if self.state["config"]["manage_harness"] and not watch.get("harness_restarted"):
                watch["harness_restarted"], watch["trips"] = True, {}
                self.restart_harness()
            else:
                self.rollback("training tables keep failing")
            return
        if trips.get("opponent_failures", 0) >= 2:
            watch["trips"] = {}
            if self.state["bot_share"] < BOT_SHARE_MAX:
                self.state["bot_share"] = min(BOT_SHARE_MAX, self.state["bot_share"] * 1.5)
                self.log(f"opponents keep failing: bots now {self.state['bot_share']}% of the league")
                self.ensure_league()
            else:
                concurrency = max(MIN_CONCURRENCY, (run.get("concurrency") or 64) // 2)
                self.stop_run("opponents keep failing even with mostly bots")
                newest = self.newest_good(run["train"])
                self.launch_run(newest["id"] if newest else run.get("start"), run.get("lr"),
                                f"restart at concurrency {concurrency}: opponents overloaded", concurrency)
            return
        behavior = [rule for rule, count in trips.items() if count >= 2 and rule not in ("opponent_failures", "rollout_failures")]
        if behavior:
            self.rollback(f"the trained seat's behavior is out of range twice in a row: {behavior}")

    def restart_harness(self) -> None:
        """ops/overnight.sh restarts the harness when it exits; killing it drops the tables in flight."""
        self.state["harness_restarts"] += 1
        self.log("restarting the harness (ops/overnight.sh brings it back)")
        subprocess.run(["pkill", "-f", "shard_router:app"], check=False)
        subprocess.run(["pkill", "-f", "poker_harness.server:app"], check=False)

    def rollback(self, why: str) -> None:
        self.state["rollbacks"] += 1
        self.log(f"ROLLBACK {self.state['rollbacks']}: {why}")
        run = self.stop_run(f"rollback: {why}")
        cps = self.state["checkpoints"].values()
        passed = [c for c in cps if c["kind"] == "ckpt" and (c.get("yardstick") or {}).get("passed") and not c.get("bad")]
        good = max(passed, key=lambda c: c["n"], default=None)
        for c in cps:
            if run and c["train"] == run["train"] and (good is None or c["n"] > good["n"]):
                c["bad"] = True
                if c["state"] == "live":
                    c["state"], c["retired_at"] = "retired", time.time()
                elif c["state"] == "pending":
                    c["state"] = "skipped"
        self.ensure_league()
        if self.state["rollbacks"] > MAX_ROLLBACKS:
            self.log(f"{MAX_ROLLBACKS} rollbacks used: stopping training")
            self.begin_final()
            return
        lr = (run or {}).get("lr") or (self.learning_rate(run["train"]) if run else None)
        start = good["id"] if good else self.state["config"]["start_checkpoint"]
        self.launch_run(start, lr / 2 if lr else None, f"rollback: {why}")

    # -- the final verdict --

    def begin_final(self) -> None:
        self.stop_run("training is over")
        self.state.pop("want_run", None)
        self.state["phase"] = "final"
        self.state["final"] = {"candidates": final_candidates(self.state["checkpoints"]), "evals": {}}
        self.log(f"final candidates: {self.state['final']['candidates']}")

    def final_step(self) -> None:
        from vals_tools.score import summarize

        final, cfg, evals = self.state["final"], self.state["config"], self.state["base_evals"]
        run = self.run()
        if run and not finished(run.get("status")):
            run["status"] = str(getattr(self.client.train.get(run["train"]), "status", "") or "").lower()
            if not finished(run["status"]):
                return  # its GPUs are still held
        self.poll_yardsticks()
        if "base_final_eval" not in evals:
            evals["base_final_eval"] = self.launch_eval(cfg["final_dataset"], cfg["base"], 16)
            self.log(f"base final eval {evals['base_final_eval']}")
        cps = self.state["checkpoints"]
        names = self.deployment_names()
        for cid in final["candidates"]:
            c = cps[cid]
            entry = final["evals"].setdefault(cid, {"name": c["name"]})
            if "eval" in entry or entry.get("error"):
                continue
            if c["name"] not in names:
                key = f"deploy-{c['name']}"
                if key in self.procs:
                    done, code, text = self.step_done(key)
                    if not done:
                        return
                    if code:
                        entry["error"] = f"redeploy failed: {text[-300:]}"
                        continue
                    names = self.deployment_names()
                if c["name"] not in names:
                    spare = [o for o in cps.values() if o["name"] in names and o["id"] not in final["candidates"]]
                    if free_gpus(len(names), False) < 1:
                        if spare:
                            self.delete(spare[0], names)
                        return
                    self.deploy(c, names)
                    return
            entry["eval"] = self.launch_eval(cfg["final_dataset"], self.model(c["name"]), 16)
            self.log(f"final eval {entry['eval']} for {c['name']}")
        if not self.base_ready("base_final_eval"):
            return
        base = self.rows(evals["base_final_eval"])
        for entry in final["evals"].values():
            if "eval" not in entry or "score" in entry:
                continue
            info = self.client.eval.get(entry["eval"])
            if not finished(getattr(info, "status", "")):
                return
            entry["status"] = str(info.status)
            entry["score"] = summarize(self.rows(entry["eval"]), base)
            ok, share = self.eval_opponents_ok(entry["eval"])
            entry["opponent_failed"], entry["valid"] = share, ok
            try:
                from vals_tools.hint_report import logs_by_job, policy_stats

                entry["behavior"] = policy_stats(logs_by_job(RUNS, {info.job_id}).get(info.job_id, []))
            except Exception as exc:
                entry["behavior"] = {"error": str(exc)[:200]}
        if len(final["evals"]) == len(final["candidates"]) and all("score" in e or e.get("error") for e in final["evals"].values()):
            self.state["phase"] = "done"
            self.write_report()

    def write_report(self) -> None:
        s = self.state
        lines = [f"Self-play ladder report ({datetime.now().astimezone():%Y-%m-%d %H:%M})", ""]
        lines += [f"runs: {r['train']} ({r['why']}; {r['status']})" for r in s["runs"]]
        lines.append(f"rollbacks {s['rollbacks']}, restarts {s['restarts']}, harness restarts {s['harness_restarts']}, "
                     f"league versions {s['league_version']}, bots {s['bot_share']}% of the league at the end")
        lines += ["", f"{'copy':<20}  {'state':<12} {'check':<7} yardstick vs base (paired by deal)"]
        for c in sorted(s["checkpoints"].values(), key=lambda c: c["n"]):
            y = c.get("yardstick") or {}
            vs = (y.get("summary") or {}).get("vs_base") or {}
            yard = (f"{vs['diff']:+.2f} +-{vs['ci95']:.2f} (n={vs['paired_hands']})" if vs.get("diff") is not None
                    else y.get("status", "-"))
            if y.get("passed") is None and "summary" in y:
                yard += f", not judged: opponents failed {y.get('opponent_failed')}"
            check = c.get("check") or {}
            state = c["state"] + (" bad" if c.get("bad") else "")
            lines.append(f"{c['name']:<20}  {state:<12} {str(check.get('failed', '-')) + '/' + str(check.get('decisions', '-')):<7} {yard}")
        lines += ["", f"final verdict on {s['config']['final_dataset']} vs base ({s['base_evals'].get('base_final_eval')}), "
                      "paired by deal:"]
        for entry in (s.get("final") or {}).get("evals", {}).values():
            vs = (entry.get("score") or {}).get("vs_base") or {}
            b = entry.get("behavior") or {}
            if vs:
                flag = "" if entry.get("valid", True) else f"  (NOT VALID: opponents failed {entry.get('opponent_failed')})"
                lines.append(f"  {entry['name']}: {vs['diff']:+.3f} BB/hand +-{vs['ci95']:.3f} (n={vs['paired_hands']}); "
                             f"failed {b.get('failed')}, VPIP {b.get('vpip')}, PFR {b.get('pfr')}{flag}")
            else:
                lines.append(f"  {entry['name']}: {entry.get('error') or entry.get('status')}")
        lines += ["", "Deployments are left up. Entering one in the arena is your call."]
        text = "\n".join(lines)
        (self.args.state.parent / "ladder_report.txt").write_text(text)
        self.log("report written:\n" + text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default="vals-hackathon")
    parser.add_argument("--until", help="when to stop training: HH:MM (local) or an ISO date-time")
    parser.add_argument("--start-checkpoint", help="warm-start the first run from this checkpoint (default: base)")
    parser.add_argument("--lr", type=float, help="learning rate of the first run (default: AC2's)")
    parser.add_argument("--steps", type=int, help="steps per run; runs continue until the deadline (default 60)")
    parser.add_argument("--concurrency", type=int, help="concurrent training hands on the harness (default 64)")
    parser.add_argument("--base", help=f"base's deployment (default {BASE})")
    parser.add_argument("--persona-model", help="the model persona seats call (default: base)")
    parser.add_argument("--house", help=f"the house model (default {HOUSE}); shared and easily overloaded")
    parser.add_argument("--house-weight", type=float, help="the house model's league weight (default 0)")
    parser.add_argument("--start-copies", type=int, help="copies of the starting model in idle slots (default 3)")
    parser.add_argument("--yardstick", help=f"dataset each checkpoint plays (default {YARDSTICK})")
    parser.add_argument("--final-dataset", help=f"dataset of the final verdict (default {FINAL_DATASET})")
    parser.add_argument("--base-yardstick-eval", help="base's eval on the yardstick (default: the ladder runs one)")
    parser.add_argument("--base-final-eval", help="base's eval on the final dataset (default: the ladder runs one)")
    parser.add_argument("--manage-harness", action="store_true", default=None,
                        help="the harness runs in ops/overnight.sh's restart loop, so the ladder may restart it")
    parser.add_argument("--new-deadline", action="store_true", help="on resume, replace the saved deadline with --until")
    parser.add_argument("--state", type=Path, default=STATE)
    parser.add_argument("--poll", type=int, default=POLL_S)
    parser.add_argument("--status", action="store_true", help="print the saved state's latest events and exit")
    args = parser.parse_args()
    if args.status:
        print("\n".join(json.loads(args.state.read_text())["events"][-40:]))
        return
    if not args.until and not args.state.exists():
        parser.error("--until is required for a new ladder")
    if not args.state.exists():
        defaults = {"steps": 60, "concurrency": 64, "base": BASE, "persona_model": args.base or BASE, "house": HOUSE,
                    "house_weight": 0.0, "manage_harness": False, "yardstick": YARDSTICK,
                    "final_dataset": FINAL_DATASET, "start_copies": MAX_OWN}
        for key, value in defaults.items():
            if getattr(args, key) is None:
                setattr(args, key, value)
    Ladder(args).loop()


if __name__ == "__main__":
    main()
