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

- NEVER eval (or otherwise load) the live arena deployment or any teammate deployment directly: my eval of
  ho-af0ee9-step10 (21:04, 32 concurrent games on its 1 GPU) overloaded it and caused 17-31 forced folds per arena
  game in 4 arena games (05:00-05:20 UTC). Deploy a separate copy (matt-* name, in the checkpoint's own project) and
  eval that, or keep eval concurrency low.

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
2. [running] harness restarted 21:03 with shover support (verified). base on pokerbots-sh-select = eval_8b800c4577cd.
   eval_72e98313dcb1 (solver-spot step 10) INVALID: 420/538 decisions were HTTP 502 (deployment overloaded by this
   eval, then deleted). Arena entry is now teammate's step 20 = infer_5fe0799751ef (vals-hackathon/ho-af0ee9-step20).
3. [queued] OPSD v3mix: dataset pokerbots-opsd-v3mix (1861 decisions: 747 from base on sh-select + 1114 from
   mb-select, hint v3), train_f485ce511a1a (4 GPUs, 30 steps, ckpts 10/20/30). Replaced train_f4dce31ab4bf.
   When done: deploy ckpts as vals-hackathon-matt/opsd3-s20, -s30, play_local check (--opponents bots), eval on
   pokerbots-sh-select (concurrency ok: my own deployments), score vs base eval_8b800c4577cd, leaks.py.
   KEY BENCHMARK = pokerbots-sh-select: base mean score 8.845, busts 13.6/100, weak call-offs 15.4/100.
   WATCH: local v3 test was very passive (VPIP 0% in 3/6 games). If OPSD ckpts show VPIP < 15% on sh-select or
   mean score below base, write hint v4 = v3 with wider late-position opens/steals and value-bet emphasis.
   Hint v4 (v3 + wide late-position steals, bigger value bets) written; local A/B v3 vs v4 at arena-like tables
   (shover, maniac, nit, station, random) running: ~/Worker/research/local_hint_v3_v4. If v4 is clearly better and
   OPSD v3mix hasn't started, rebuild the OPSD dataset with v4 (build_dataset reads HINT; set HINT = HINTS["v4"]).
4. GRPO on pokerbots-sh-train (bust penalty grader), concurrency 16, after OPSD or when 6 GPUs free.
5. Deploy best checkpoints (in vals-hackathon-matt), leaks.py + paired score vs base on sh-select; best = candidate.

## Candidate to submit
Submission name for every model: **Alea iacta est** (user's request).
- none of mine yet (arena entry = teammate's solver-spot step 20, infer_5fe0799751ef).
