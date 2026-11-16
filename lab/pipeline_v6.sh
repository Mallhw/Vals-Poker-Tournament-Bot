#!/bin/zsh
# After OPSD v6 (train_3866513541d5) finishes: deploy its last checkpoint as vals-hackathon-matt/v6-s20, check it,
# then eval on pokerbots-sh-select (vs butter = rl-s10) and on games12-control (multi-hand, no hint: did it learn notes?).
set -u
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/bin:/bin"
KEY="$(/usr/bin/python3 -I -c 'import json,os;print(json.load(open(os.path.expanduser("~/.ac2/config")))["api_key"])')"
export UV_INDEX_AC2_USERNAME=__token__ UV_INDEX_AC2_PASSWORD="$KEY"
R=$HOME/Worker/projects/vals-poker-matt; R2=$HOME/Worker/projects/vals-poker-matt2
log(){ echo "$(date +%H:%M) $*"; }

until s=$(cd /tmp && ac2 -p vals-hackathon-matt train get train_3866513541d5 2>/dev/null | awk '/^status:/{print $2}'); case "$s" in succeeded|failed|stopped|cancelled|error) true;; *) false;; esac; do sleep 90; done
log "OPSD v6 finished: $s"
[ "$s" = "succeeded" ] || exit 1
CK=$(cd /tmp && ac2 -p vals-hackathon-matt train checkpoints train_3866513541d5 2>/dev/null | awk '/ckpt_/{print $1; exit}')
log "last checkpoint: $CK"

cd $R && uv run python -m vals_tools.deploy --project vals-hackathon-matt --name v6-s20 --checkpoint $CK > /tmp/dep_v6.log 2>&1 &
P=$!; for i in $(seq 1 80); do grep -q deploying /tmp/dep_v6.log && break; sleep 15; done; sleep 3; kill $P 2>/dev/null
log "$(grep -h deploying /tmp/dep_v6.log)"
for i in $(seq 1 80); do (cd /tmp && ac2 inference list 2>/dev/null | grep -E "v6-s20 +ready" -q) && break; sleep 30; done
for j in $(seq 1 15); do c=$(curl -s -m 60 -o /dev/null -w "%{http_code}" https://inference.appliedcompute.com/v1/chat/completions -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' -d '{"model":"vals-hackathon-matt/v6-s20","messages":[{"role":"user","content":"Reply with OK."}],"max_completion_tokens":50,"chat_template_kwargs":{"enable_thinking":false}}'); [ "$c" = "200" ] && break; sleep 20; done
log "v6-s20 probe $c"

cd $R/harness && AC2_API_KEY="$KEY" uv run -q python play_local.py --model vals-hackathon-matt/v6-s20 --opponents bots --hands 6 --out /tmp/playlocal_v6 2>&1 | grep -vE "^hand [0-9]+ done" | tail -2
cd $R && log "sh-select eval: $(uv run python -m vals_tools.eval --project vals-hackathon-matt --dataset pokerbots-sh-select --deployment vals-hackathon-matt/v6-s20 --max-parallel 16 2>&1 | tail -1)"
cd $R2 && log "games12-control eval (v6, no hint): $(uv run python -m vals_tools.eval --project vals-hackathon-matt2 --dataset games12-control --deployment vals-hackathon-matt/v6-s20 --max-parallel 8 2>&1 | tail -1)"
