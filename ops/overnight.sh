#!/bin/bash
# The overnight self-play run in one command, in WSL: the harness and the connector in restart loops, then the ladder
# (vals_tools/ladder.py), which also restarts if it dies and resumes from its saved state. Nothing here waits for input.
#
#   bash ops/overnight.sh --until 08:00                        # plus any vals_tools.ladder flags
#   bash ops/overnight.sh --status                             # the ladder's latest events
#   bash ops/overnight.sh --stop                               # stop the ladder, the harness and the connector
#
# Before you start it: stop any harness or connector you already run (this one runs its own, with the league file),
# free the GPUs (only base should be deployed), and set Windows never to sleep while plugged in.
# Logs: data/launches/overnight/{harness,connector,ladder}.log. The ladder's report: data/launches/ladder_report.txt.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGS="$ROOT/data/launches/overnight"
mkdir -p "$LOGS" "$ROOT/data/league"
export PATH="$HOME/.local/bin:$PATH" UV_LINK_MODE=copy UV_CONCURRENT_INSTALLS=1
export POKER_LEAGUE_FILE="$ROOT/data/league/league.json"

case "${1:-}" in
  --stop)
    pkill -f "overnight-loop" ; pkill -f "vals_tools.ladder" ; pkill -f "shard_router:app"
    pkill -f "poker_harness.server:app" ; pkill -f "ac2 harness connect"
    echo "stopped (training runs and deployments on AC2 are left as they are)"; exit 0 ;;
  --status)
    cd "$ROOT" && exec uv run python -m vals_tools.ladder --status ;;
esac

if curl -s --max-time 3 http://127.0.0.1:8765/health >/dev/null || pgrep -f "ac2 harness connect" >/dev/null; then
  echo "A harness or connector is already running. Stop it first: this script runs its own, with the league file."
  exit 1
fi

# The ladder's deploy probe and local checks call AC2 with your login's key (read here, never printed).
AC2_API_KEY="$(python3 -c 'import json, os; print(json.load(open(os.path.expanduser("~/.ac2/config")))["api_key"])')"
export AC2_API_KEY

nohup setsid bash -c 'while true; do pokerbot-harness; echo "harness exited $?; restarting in 10 s"; sleep 10; done' \
  overnight-loop-harness >> "$LOGS/harness.log" 2>&1 &
for _ in $(seq 60); do curl -s --max-time 3 http://127.0.0.1:8765/health >/dev/null && break; sleep 5; done
curl -s --max-time 3 http://127.0.0.1:8765/health >/dev/null || { echo "the harness didn't come up: see $LOGS/harness.log"; exit 1; }
echo "harness up (league file $POKER_LEAGUE_FILE)"

nohup setsid bash -c 'while true; do pokerbot-connector; echo "connector exited $?; restarting in 15 s"; sleep 15; done' \
  overnight-loop-connector >> "$LOGS/connector.log" 2>&1 &
echo "connector started"

cd "$ROOT"
nohup setsid bash -c 'while true; do uv run python -m vals_tools.ladder "$@" --manage-harness && break; echo "ladder exited; restarting in 30 s"; sleep 30; done' \
  overnight-loop-ladder "$@" >> "$LOGS/ladder.log" 2>&1 &
echo "ladder started: tail -f $LOGS/ladder.log  (stop everything: bash ops/overnight.sh --stop)"
