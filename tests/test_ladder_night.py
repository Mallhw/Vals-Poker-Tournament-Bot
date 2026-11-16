"""A whole night of vals_tools.ladder against a fake AC2: start copies, a checkpoint joining the league (making room
for it), its yardstick, a continuation run, the deadline, the final evals and the report. No network, no GPUs."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from vals_tools import ladder as ld


class FakeAC2:
    def __init__(self) -> None:
        self.runs: dict[str, str] = {}
        self.ckpts: dict[str, list] = {}
        self.deployed: dict[str, str] = {"base": "infer_base"}
        self.evals: dict[str, dict] = {}
        self.launched_runs: list[list[str]] = []
        outer = self

        class Train:
            def get(self, train_id):
                return SimpleNamespace(status=outer.runs[train_id])

            def checkpoints(self, train_id):
                return outer.ckpts.get(train_id, [])

            def stop(self, train_id, stop_evals=False):
                outer.runs[train_id] = "stopped"

            def get_step_metrics(self, train_job_id):
                return {"train/lr": [{"value": 2e-6}]}

        class Inference:
            def list(self):
                return [SimpleNamespace(name=n, id=i) for n, i in outer.deployed.items()]

            def delete(self, deployment_id):
                outer.deployed = {n: i for n, i in outer.deployed.items() if i != deployment_id}

        class Eval:
            async def run(self, config):
                eval_id = f"eval_{len(outer.evals):04d}"
                outer.evals[eval_id] = {"dataset": config.dataset, "seat": config.tags["seat_model"]}
                return SimpleNamespace(eval_id=eval_id)

            def get(self, eval_id):
                return SimpleNamespace(status="succeeded", job_id=f"job_{eval_id}")

        self.train, self.inference, self.eval = Train(), Inference(), Eval()


def a_night(tmp_path: Path, monkeypatch) -> tuple[FakeAC2, "ld.Ladder"]:
    ac2 = FakeAC2()
    monkeypatch.setattr("ac2.sdk.Client", lambda project: ac2)
    for name, value in {"LEAGUE_FILE": tmp_path / "league.json", "LOGS": tmp_path / "logs", "RUNS": tmp_path / "runs",
                        "DRAIN_S": 0, "WATCH_EVERY_S": 10**9}.items():
        monkeypatch.setattr(ld, name, value)

    def fake_run(cmd, **kwargs):  # rl/train.py
        train_id = f"train_{len(ac2.runs) + 1:012x}"
        ac2.runs[train_id] = "running"
        ac2.launched_runs.append(cmd)
        return SimpleNamespace(returncode=0, stdout=json.dumps({"train_id": train_id}), stderr="")

    def fake_spawn(self, key, cmd, cwd):  # vals_tools.deploy and play_local finish at once
        ld.LOGS.mkdir(parents=True, exist_ok=True)
        name = key.split("-", 1)[1]
        if key.startswith("deploy-"):
            ac2.deployed[name] = f"infer_{name}"
            text = "deployed"
        else:
            text = f"6 hands, 20 decisions by x/{name}, 1 failed"
        (ld.LOGS / f"{key}.log").write_text(text)
        self.procs[key] = SimpleNamespace(poll=lambda: 0, returncode=0)

    rows = {f"t{i}": {"ev_bb": 0.1 * i, "realized_bb": 0.0, "policy_format_errors": 0, "deal": f"{i}:seat0"} for i in range(20)}
    monkeypatch.setattr(ld.subprocess, "run", fake_run)
    monkeypatch.setattr(ld.Ladder, "spawn", fake_spawn)
    monkeypatch.setattr("vals_tools.score.score_rows", lambda client, eval_id=None, **k: rows)
    args = SimpleNamespace(project="p", until="23:59", start_checkpoint=None, lr=None, steps=60, concurrency=64,
                           base="p/base", persona_model="p/base", house=ld.HOUSE, house_weight=0.0, start_copies=3,
                           yardstick="pokerbots-select200", final_dataset="pokerbots-eval1000", base_yardstick_eval=None,
                           base_final_eval=None, manage_harness=False, new_deadline=False, state=tmp_path / "state.json",
                           poll=0)
    night = ld.Ladder(args)
    night.state["until"] = (datetime.now().astimezone() + timedelta(hours=8)).isoformat()
    return ac2, night


def test_a_night(tmp_path: Path, monkeypatch) -> None:
    ac2, night = a_night(tmp_path, monkeypatch)
    for _ in range(8):  # the first run, three copies of base filling the slots, base's yardstick
        night.tick()
    cps = night.state["checkpoints"]
    assert [c["state"] for c in cps.values() if c["kind"] == "start"] == ["live"] * 3
    assert len(ac2.runs) == 1 and len(ac2.deployed) == 4 and any(e["seat"] == "p/base" for e in ac2.evals.values())
    league = json.loads(ld.LEAGUE_FILE.read_text())
    assert {e["model"] for e in league["entries"] if "model" in e} >= {"p/sp-start1-step0", "p/base"}

    first = next(iter(ac2.runs))
    ac2.ckpts[first] = [SimpleNamespace(id="ckpt_a", iter_num=10)]
    for _ in range(8):  # retire a start copy, delete it, deploy, check, go live, yardstick
        night.tick()
    a = cps["ckpt_a"]
    assert a["state"] == "live" and a["check_passed"] and a["yardstick"]["passed"] is True
    assert sum(c["state"] == "deleted" for c in cps.values()) == 1 and len(ac2.deployed) == 4  # never past the quota
    assert "p/sp-000001-step10" in {e.get("model") for e in json.loads(ld.LEAGUE_FILE.read_text())["entries"]}

    ac2.runs[first] = "succeeded"
    night.tick()
    assert len(ac2.runs) == 2 and "ckpt_a" in ac2.launched_runs[-1]  # continued from its newest checkpoint

    night.state["until"] = (datetime.now().astimezone() - timedelta(minutes=1)).isoformat()
    for _ in range(8):
        night.tick()
        if night.state["phase"] == "done":
            break
    assert night.state["phase"] == "done"
    assert ac2.runs[list(ac2.runs)[-1]] == "stopped"
    assert set(night.state["final"]["candidates"]) == {"ckpt_a"}
    report = (tmp_path / "ladder_report.txt").read_text()
    assert "sp-000001-step10" in report and "final verdict" in report
    assert "base" in ac2.deployed  # base is never touched


def test_a_seat_that_stops_playing_hands_is_rolled_back(tmp_path: Path, monkeypatch) -> None:
    ac2, night = a_night(tmp_path, monkeypatch)
    for _ in range(8):
        night.tick()
    first = next(iter(ac2.runs))
    ac2.ckpts[first] = [SimpleNamespace(id="ckpt_a", iter_num=10)]
    for _ in range(8):
        night.tick()
    ac2.ckpts[first].append(SimpleNamespace(id="ckpt_b", iter_num=20))
    night.tick()
    # the trained seat now folds every hand at the league tables
    ld.RUNS.mkdir(parents=True, exist_ok=True)
    fold = {"player": "seat0", "street": "preflop", "action": {"type": "fold"}, "call_amount": 2, "stack_after": 200,
            "model_response": {"model_name": "ac2/policy"}}
    for i in range(200):
        (ld.RUNS / f"t{i}.json").write_text(json.dumps({"lineup": {"version": 3}, "hands": [{"actions": [fold]}]}))
    monkeypatch.setattr(ld, "WATCH_EVERY_S", 0)
    night.state["baseline"] = {"failed": 0.02, "vpip": 0.22, "allin": 0.02}
    night.tick()
    night.tick()  # a second window in a row
    assert night.state["rollbacks"] == 1 and ac2.runs[first] == "stopped"
    assert night.state["checkpoints"]["ckpt_b"].get("bad") and not night.state["checkpoints"]["ckpt_a"].get("bad")
    relaunch = ac2.launched_runs[-1]
    assert "ckpt_a" in relaunch and relaunch[relaunch.index("--lr") + 1] == "1e-06"  # half of the run's 2e-6
