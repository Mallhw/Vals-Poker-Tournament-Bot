#!/bin/zsh
# RL v3: GRPO from v10-s9 (ckpt_0a19c764c8d4, the live arena model) on pokerbots-rl2 (base LLM opps + shover + maniac,
# bust_penalty 3) (preemptible: SDK rejects the flag on launch; if stopped, `ac2 train resume --non-preemptible`). Goal: add chips (v10 too tight, VPIP 8%) without losing survival. Then auto-eval.
set -u
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/bin:/bin"
KEY="$(/usr/bin/python3 -I -c 'import json,os;print(json.load(open(os.path.expanduser("~/.ac2/config")))["api_key"])')"
export UV_INDEX_AC2_USERNAME=__token__ UV_INDEX_AC2_PASSWORD="$KEY"
log(){ echo "$(date +%H:%M) $*"; }
cd $HOME/Worker/projects/vals-poker-matt
OUT=$(uv run python rl/train.py --project vals-hackathon-matt --prefix pokerbots-rl2 --from-checkpoint ckpt_0a19c764c8d4 --steps 30 --no-eval --concurrency 16 --checkpoint-every 2 2>&1 | tail -1)
log "RL v3 from v10-s9: $OUT"
TID=$(echo "$OUT" | grep -oE 'train_[0-9a-f]+' | head -1)
[ -n "$TID" ] || exit 1
lab/pipeline_ckpt.sh $TID rl3-s30
