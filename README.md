# Vals Poker Bot

Train Qwen3.6-35B-A3B to play 6-player poker, then enter it in the arena.

Team **pineapple**'s code from the Applied Compute × Vals Pokerbots hackathon (Nov 2026). Peak **#2** on the
ranked ladder (rating 1191); finished **8th of 11 teams**. Start with **[SUMMARY.md](SUMMARY.md)** for results
and lessons, [`reports/`](reports/) for model comparisons, and [`lab/LAB.md`](lab/LAB.md) for the full
experiment notebook. Built on [Applied-Compute/pokerbots-starter](https://github.com/Applied-Compute/pokerbots-starter).
Trained weights are not included (they lived on AC2).

## The game
- 6 players, 30 hands, 200 chips each. Most chips at the end wins.
- Each turn the model makes **one** `submit_action` call. A bad move = check/fold.
- Max 5,000 tokens per move.

## What's here
| Folder | What it does |
|---|---|
| `harness/` | Poker server that plays the games and scores them. Runs on your machine. |
| `rl/` | Makes the datasets and starts training. |
| `opsd/` | Teaches the model a written strategy hint (`hint.py`) without needing it in the prompt. |
| `vals_tools/` | Deploy a model, evaluate it, check your setup. |
| `src/pokerbots/` | Empty on purpose (AC2 needs it). |
| `lab/` | Our experiment framework: arena tracking, entry comparison, leak analysis, opponent scouting, paired prompt tests, pipelines. |
| `reports/` | Write-ups comparing our arena entries. |

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

## Teach it the strategy hint (OPSD)
`opsd/hint.py` holds strategy notes aimed at the bot's arena leaks (busting, calling too much, loose early opens).
Try it locally first (any OpenAI-compatible model, no AC2 needed):
```bash
uv run --project harness python opsd/local_test.py --seeds 1 2 3 --hands 10 --no-thinking
```
Then train it in (needs a `base` deployment):
```bash
uv run python -m vals_tools.eval --dataset pokerbots-train --deployment vals-hackathon/base
uv run python opsd/build_dataset.py --eval <eval_id> --dataset pokerbots-opsd-train
uv run python opsd/train.py --dataset pokerbots-opsd-train
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
