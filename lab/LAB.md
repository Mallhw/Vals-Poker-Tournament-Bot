# Pokerbot lab — state and plan

Goal: the best possible arena entry for team pineapple (Vals pokerbots, 6-max NLHE, 30-hand elimination games).
Owner: Matt. Run autonomously overnight 2026-11-14/15 by Claude (hourly check-ins).

## Rules
- Use project `vals-hackathon-matt` (prj_face168dc6d7) for my runs/evals; my deployments: `vals-hackathon-matt/base`.
- May delete teammates' deployments only when clearly idle (no eval/run using them). Never stop teammates' runs.
  Never touch the live arena entry — check `python3 -I lab/arena.py` status "deployment" before deleting anything
  (as of 21:00 it is infer_c5af0a968fd3 = vals-hackathon/ho-af0ee9-step10; the old "👀" entry infer_2ff17466cceb is
  superseded but keep it unless GPUs are needed, since teammates may switch back).
- Dashboard: http://matthewli.org/pokerbot-lab/ (repo Mallhw/pokerbot-lab). Refresh: `python3 -I lab/arena.py &&
  python3 -I lab/publish.py`. Arena submission is console-only (user/teammate); keep the best checked
  candidate deployed and say which in this file.
- Harness on Otzi: launchd `com.mallhw.vals-harness` (127.0.0.1:8775), connector `com.mallhw.vals-connector`.
  Restart harness only when no eval/run is using it (kills in-flight games):
  `launchctl kickstart -k gui/$(id -u)/com.mallhw.vals-harness`.

## Tools (run from ~/Worker/projects/vals-poker-matt)
- `python3 -I lab/arena.py` — pull arena matches, per-entry metrics, rank/rating history.
- `uv run --project harness python lab/leaks.py --eval <eval_id>` — leak report from local game logs.
- `uv run python -m vals_tools.eval --project vals-hackathon-matt --dataset <ds> --deployment <proj/name>`
- `uv run python -m vals_tools.score --project vals-hackathon-matt --eval <new> --base <base>` (paired)
- Datasets (all in vals-hackathon-matt): `pokerbots-mb-*` (opponents = my base, thinking off),
  `pokerbots-sh-*` (same + scripted shover seat 2, maniac seat 5; needs the restarted harness),
  `pokerbots-vb-*` (opponent vals-hackathon/base — DELETED, don't use), `pokerbots-*` (house model — down).

## Findings so far
- Arena (65 games): base busts 22-44% of games; 16/20 busts with weak hands (Chen<8), mostly calling 100 BB
  shoves from house shovers. Single-hand evals vs base opponents almost never bust (0.35/100 hands) — they
  don't contain shove spots, hence the sh datasets.
- House opponent `ac-vals-model` down since ~18:55.
- Research (research_notes/.../strategy_for_hint.md): call off 100 BB only with AA/KK vs tight/unknown shover;
  vs a confirmed any-two shover 99+/AJs+/AQo+; multiway KK+. Survival = +15-20 equity points risk premium.
- Local hint A/B (llama 27B, 6 seeds): hint -> 0 busts vs 2, +108 vs +19 BB, but ~2% format errors (hint v2 fixes
  wording; re-test running in ~/Worker/research/local_hint_test_v2).

## Plan / queue
1. [done] base eval on pokerbots-mb-select: eval_29f9d8cd0fce — mean score 9.461, EV +0.39 (+-1.73).
2. [running] harness restarted 21:03 with shover support (verified). Evals on pokerbots-sh-select:
   base = eval_8b800c4577cd, arena entry (vals-hackathon/ho-af0ee9-step10) = eval_72e98313dcb1.
   When done: vals_tools.score --eval eval_72e98313dcb1 --base eval_8b800c4577cd, and leaks.py on both.
3. [queued] OPSD v3: dataset pokerbots-opsd-v3 (1114 decisions from eval_29f9d8cd0fce), train_f4dce31ab4bf
   (4 GPUs, 20 steps, ckpts 10 & 20). When done: deploy ckpts as vals-hackathon-matt/opsd3-s10, -s20 (they can only
   be deployed in vals-hackathon-matt), play_local check (--opponents bots), sh-select eval, score vs base.
   Local v3 hint test running: ~/Worker/research/local_hint_test_v3.
4. GRPO on pokerbots-sh-train (bust penalty grader), concurrency 16, after OPSD or when 6 GPUs free.
5. Deploy best checkpoints (in vals-hackathon-matt), leaks.py + paired score vs base on sh-select; best = candidate.

## Candidate to submit
- none of mine yet (arena entry = teammate's ho-af0ee9-step10).
