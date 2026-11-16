#!/bin/zsh
# After OPSD v10 (train_6558f04b6d10) finishes and frees GPUs, launch RL v2 from v6 (ckpt_76bbfc0511e7): realistic
# tables (pokerbots-rl2: base LLM opps + shover + maniac, bust_penalty 3). Goal: keep v6's low busts, train out its
# format errors (update_scratchpad mid-hand, no-JSON replies), since a broken move scores 0. Then auto-evaluate.
set -u
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/bin:/bin"
KEY="$(/usr/bin/python3 -I -c 'import json,os;print(json.load(open(os.path.expanduser("~/.ac2/config")))["api_key"])')"
export UV_INDEX_AC2_USERNAME=__token__ UV_INDEX_AC2_PASSWORD="$KEY"
log(){ echo "$(date +%H:%M) $*"; }
until s=$(cd /tmp && ac2 -p vals-hackathon-matt train get train_6558f04b6d10 2>/dev/null | awk '/^status:/{print $2}'); case "$s" in succeeded|failed|stopped|cancelled|error) true;; *) false;; esac; do sleep 90; done
log "v10 training $s; launching RL v2 from v6"
cd $HOME/Worker/projects/vals-poker-matt
OUT=$(uv run python rl/train.py --project vals-hackathon-matt --prefix pokerbots-rl2 --from-checkpoint ckpt_76bbfc0511e7 --steps 30 --no-eval --concurrency 16 --checkpoint-every 10 2>&1 | tail -1)
log "RL v2: $OUT"
TID=$(echo "$OUT" | grep -oE 'train_[0-9a-f]+' | head -1)
[ -n "$TID" ] || exit 1
lab/pipeline_ckpt.sh $TID rl2-s30
