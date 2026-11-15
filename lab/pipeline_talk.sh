#!/bin/zsh
# Wait for the v5 + v7 notes evals on harness2 to finish, restart harness2 (loads the policy_talk code), then upload
# and launch the chaos / bluff-talk table-talk experiments on the same 36 games (policy = v4s30-eval copy).
set -u
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
KEY="$(/usr/bin/python3 -I -c 'import json,os;print(json.load(open(os.path.expanduser("~/.ac2/config")))["api_key"])')"
export UV_INDEX_AC2_USERNAME=__token__ UV_INDEX_AC2_PASSWORD="$KEY"
log(){ echo "$(date +%H:%M) $*"; }
st(){ (cd /tmp && ac2 -p vals-hackathon-matt2 eval get $1 2>/dev/null | awk '/status:/{print $2}'); }
# (v5/v7 already finished)
log "v5=$a v7=$b done"
# restart only when harness2 is idle (no rollouts) on two checks in a row; never force it
idle=0; while [ $idle -lt 2 ]; do r=$(curl -s -m 5 http://127.0.0.1:8785/health | grep -oE '"running": [0-9]+' | grep -oE '[0-9]+'); if [ "${r:-1}" = "0" ]; then idle=$((idle+1)); else idle=0; fi; sleep 60; done
launchctl kickstart -k gui/$(id -u)/com.mallhw.vals-harness2
for i in $(seq 1 60); do curl -s -m 2 http://127.0.0.1:8785/health | grep -q '"ok", "ok", "ok", "ok"' && break; sleep 2; done
log "harness2 restarted: $(curl -s -m 5 http://127.0.0.1:8785/health)"
cd $HOME/Worker/projects/vals-poker-matt2
uv run python lab/make_games.py --project vals-hackathon-matt2 --opponent-model vals-hackathon-matt/base --games 36 --hands 12 --variants chaos bluff-talk 2>&1 | tail -2
for v in chaos bluff-talk; do log "games12-$v eval: $(uv run python -m vals_tools.eval --project vals-hackathon-matt2 --dataset games12-$v --deployment vals-hackathon-matt/v4s30-eval --max-parallel 8 2>&1 | tail -1)"; done
uv run python lab/make_games.py --project vals-hackathon-matt2 --opponent-model vals-hackathon-matt/base --games 36 --hands 12 --variants v8 2>&1 | tail -1
log "games12-v8 eval: $(uv run python -m vals_tools.eval --project vals-hackathon-matt2 --dataset games12-v8 --deployment vals-hackathon-matt/v4s30-eval --max-parallel 8 2>&1 | tail -1)"
