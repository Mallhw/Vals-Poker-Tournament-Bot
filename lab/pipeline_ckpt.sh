#!/bin/zsh
# Generic: after training run $1 finishes, deploy its last checkpoint as vals-hackathon-matt/$2, check it, eval sh-select + games12-control.
# Usage: lab/pipeline_ckpt.sh <train_id> <deployment-name>   (log to ~/Worker/logs/pipeline_<name>.log)
set -u
TRAIN=$1; NAME=$2; CKARG=${3:-}   # optional 3rd arg: deploy this ckpt now (skip waiting for the run)
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/bin:/bin"
KEY="$(/usr/bin/python3 -I -c 'import json,os;print(json.load(open(os.path.expanduser("~/.ac2/config")))["api_key"])')"
export UV_INDEX_AC2_USERNAME=__token__ UV_INDEX_AC2_PASSWORD="$KEY"
R=$HOME/Worker/projects/vals-poker-matt; R2=$HOME/Worker/projects/vals-poker-matt2
log(){ echo "$(date +%H:%M) $*"; }

if [ -n "$CKARG" ]; then CK=$CKARG; else
until s=$(cd /tmp && ac2 -p vals-hackathon-matt train get $TRAIN 2>/dev/null | awk '/^status:/{print $2}'); case "$s" in succeeded|failed|stopped|cancelled|error) true;; *) false;; esac; do sleep 90; done
log "$TRAIN finished: $s"
[ "$s" = "succeeded" ] || exit 1
CK=$(cd /tmp && ac2 -p vals-hackathon-matt train checkpoints $TRAIN 2>/dev/null | awk '/ckpt_/{print $1; exit}')
fi
log "last checkpoint: $CK"

cd $R && uv run python -m vals_tools.deploy --project vals-hackathon-matt --name $NAME --checkpoint $CK > /tmp/dep_$NAME.log 2>&1 &
P=$!; for i in $(seq 1 80); do grep -q deploying /tmp/dep_$NAME.log && break; sleep 15; done; sleep 3; kill $P 2>/dev/null
log "$(grep -h deploying /tmp/dep_$NAME.log)"
for i in $(seq 1 80); do (cd /tmp && ac2 inference list 2>/dev/null | grep -E "$NAME +ready" -q) && break; sleep 30; done
for j in $(seq 1 15); do c=$(curl -s -m 60 -o /dev/null -w "%{http_code}" https://inference.appliedcompute.com/v1/chat/completions -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' -d "{\"model\":\"vals-hackathon-matt/$NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with OK.\"}],\"max_completion_tokens\":50,\"chat_template_kwargs\":{\"enable_thinking\":false}}"); [ "$c" = "200" ] && break; sleep 20; done
log "$NAME probe $c"

cd $R/harness && AC2_API_KEY="$KEY" uv run -q python play_local.py --model vals-hackathon-matt/$NAME --opponents bots --hands 6 --out /tmp/playlocal_$NAME 2>&1 | grep -vE "^hand [0-9]+ done" | tail -2
cd $R && log "sh-select eval: $(uv run python -m vals_tools.eval --project vals-hackathon-matt --dataset pokerbots-sh-select --deployment vals-hackathon-matt/$NAME --max-parallel 16 2>&1 | tail -1)"
cd $R2 && log "games12-control eval ($NAME, no hint): $(uv run python -m vals_tools.eval --project vals-hackathon-matt2 --dataset games12-control --deployment vals-hackathon-matt/$NAME --max-parallel 8 2>&1 | tail -1)"
