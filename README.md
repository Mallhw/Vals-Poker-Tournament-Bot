# Vals Poker Bot

Train Qwen3.6-35B-A3B to play 6-player poker, then enter it in the arena.

## The game
- 6 players, 30 hands, 200 chips each. Most chips at the end wins.
- Each turn the model makes **one** `submit_action` call. A bad move = check/fold.
- Max 5,000 tokens per move.

## What's here
| Folder | What it does |
|---|---|
| `harness/` | Poker server that plays the games and scores them. Runs on your machine. |
| `rl/` | Makes the datasets and starts training. |
| `vals_tools/` | Deploy a model, evaluate it, check your setup. |
| `src/pokerbots/` | Empty on purpose (AC2 needs it). |

## Setup (once)
```bash
curl -fsSL https://api.appliedcompute.com/install.sh | sh
ac2 login
uv sync && (cd harness && uv sync)
ac2 project init
uv run python rl/make_dataset.py
```

## Run the harness (keep both open)
**Terminal 1:**
```bash
cd harness
export AC2_API_KEY=<your key>
export POKER_HARNESS_TOKEN=$(openssl rand -hex 16); echo $POKER_HARNESS_TOKEN
SHARDS=4 uv run ./run_sharded.sh
```
**Terminal 2:** (get the ID/secret once with `ac2 harness credentials create`)
```bash
export AC2_SERVICE_CLIENT_ID=svc_...
export AC2_SERVICE_CLIENT_SECRET=...
ac2 harness connect --project-id <prj_...> --upstream http://127.0.0.1:8765 --upstream-token <token>
```
Check it: `uv run python -m vals_tools.doctor --datasets --harness http://127.0.0.1:8765`

## Train
```bash
uv run python rl/train.py --steps 5     # quick test first
uv run python rl/train.py               # real run
ac2 train checkpoints <train_id>        # list saved checkpoints
```

## Test a checkpoint
```bash
uv run python -m vals_tools.deploy --name entry --checkpoint <ckpt_...>
cd harness && uv run python play_local.py --model vals-hackathon/entry --hands 6
```
Want **0 failed** decisions.

## Submit
AC2 console → **Vals Arena** → pick `entry` → submit.

**Tip:** submit the base model first so you're on the leaderboard:
`uv run python -m vals_tools.deploy --name entry --model Qwen/Qwen3.6-35B-A3B`

## Limits
10 GPUs. Training = 8, each deployment = 1. Free a GPU:
`uv run python -m vals_tools.deploy --name <name> --delete`
