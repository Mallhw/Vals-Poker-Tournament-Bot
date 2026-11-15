"""Shared settings and helpers for the pokerbot lab (run with the repo's or harness's uv env, or plain python3)."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

LAB = Path(__file__).resolve().parent
REPO = LAB.parent
HARNESS = REPO / "harness"
RUNS = HARNESS / "runs"  # every game the Otzi harness plays (evals and training) lands here
DATA = LAB / "data"
EXPERIMENTS = LAB / "experiments.jsonl"

PROJECT = "vals-hackathon-matt"
ORG = "org_9197691edc5e"
ARENA_ENTRY = "infer_5fe0799751ef"  # live arena entry since ~21:20 (vals-hackathon/ho-af0ee9-step20): never touch; check lab/arena.py status
BASE = f"{PROJECT}/base"


def api_key() -> str:
    return json.load(open(Path.home() / ".ac2" / "config"))["api_key"]


def env() -> dict[str, str]:
    """Environment for ac2 / uv commands: PATH, the AC2 package index login, and the key the harness tools need."""
    e = dict(os.environ)
    key = api_key()
    e["PATH"] = f"{Path.home()}/.local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
    e["UV_INDEX_AC2_USERNAME"] = "__token__"
    e["UV_INDEX_AC2_PASSWORD"] = key
    e["AC2_API_KEY"] = key
    for k in ("AC2_SERVICE_CLIENT_ID", "AC2_SERVICE_CLIENT_SECRET"):
        e.pop(k, None)
    return e


def sh(cmd: list[str] | str, cwd: Path | str | None = None, timeout: int = 1800, check: bool = False) -> str:
    """Run a command, return stdout+stderr as text."""
    p = subprocess.run(cmd, cwd=cwd, env=env(), shell=isinstance(cmd, str), capture_output=True, text=True,
                       timeout=timeout)
    out = (p.stdout or "") + (p.stderr or "")
    if check and p.returncode:
        raise RuntimeError(f"command failed ({p.returncode}): {cmd}\n{out[-2000:]}")
    return out


def ac2(*args: str, project: str | None = None, timeout: int = 300) -> str:
    """Run the ac2 CLI from a neutral folder (so a tracked folder's project doesn't get in the way)."""
    base = ["ac2"] + (["-p", project] if project else [])
    return sh(base + list(args), cwd="/tmp", timeout=timeout)


def field(text: str, name: str) -> str | None:
    """Read 'name: value' from ac2 get output."""
    for line in text.splitlines():
        if line.strip().startswith(f"{name}:"):
            return line.split(":", 1)[1].strip()
    return None


def record(**entry) -> dict:
    """Append one event to the experiment log."""
    entry = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **entry}
    with open(EXPERIMENTS, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def experiments() -> list[dict]:
    if not EXPERIMENTS.exists():
        return []
    return [json.loads(line) for line in EXPERIMENTS.read_text().splitlines() if line.strip()]
