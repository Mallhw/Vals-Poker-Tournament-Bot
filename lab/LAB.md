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

- 22:31 (user: "replace anything idle"): deleted idle old "👀" entry infer_2ff17466cceb and my idle
  vals-hackathon-matt/base. BEFORE any mb/sh eval, redeploy base (the datasets' opponent):
  `uv run python -m vals_tools.deploy --project vals-hackathon-matt --name base --model Qwen/Qwen3.6-35B-A3B`
  (its readiness probe is buggy — check `ac2 -p vals-hackathon-matt inference list` for "ready" instead).

- BEFORE DELETING ANY DEPLOYMENT, check the "Deployments in use" list below: evals call deployments by name and
  the CLI can't show which eval uses which. 01:30: deleting v4s30-eval broke the 5 feature evals for ~15 min
  (fixed by redeploying the same name; lab/games.py drops games with relay/HTTP errors).
- Deployments in use (keep updated): opsd4-s30 = LIVE arena entry; base = opponent for sh/games12 datasets;
  v4s30-eval (infer_ac58d18d0a7d) = policy for games12-* feature evals (until they finish); rl-s10 = eval_ecbb6ba6dfb7.

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

- Live entry (solver-spot step 20, 29 games): survives (1 bust) but weak-tight — limps 14% vs open-raises 12%,
  3-bets 3%, never raises post-flop, loses on the BTN, wins 69% of showdowns. Hint v4 targets this (+ never limp).

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
   22:48: teammate RL stopped (broken, user OK). 23:08: v3mix stopped (user: "just do v4"). RUNNING: OPSD v4mix
   train_c2f55e32f7e3 (pokerbots-opsd-v4mix, hint v4, 30 steps, ckpts 10/20/30). HINT default = v4 now.
   23:50: v4 DONE (ckpts s9 ckpt_4677eda7d3d9, s19 ckpt_0f58046aa3f9, s29 ckpt_db759b762b5d). Deployed
   vals-hackathon-matt/opsd4-s30 (infer_496116c2999d) and opsd4-s20 (infer_eb929696902d); sh-select evals +
   play_local checks launching automatically (bg job). Local v3 vs v4: v4 looser (VPIP 51%, 3 busts/6) — watch busts.
   23:28: v4 at step 8/30 (~1 min/step). base redeployed (infer_74f29225a15a). After v4: deploy s20 + s30 as vals-hackathon-matt/opsd4-s20, -s30, play_local check, eval on
   sh-select, pick best by mean score + busts + VPIP (target: Pete-like ~30% VPIP / ~20% PFR, busts << base).
4. [RUNNING 00:21] GRPO from v4 s29 (ckpt_db759b762b5d, rl/train.py --from-checkpoint) on pokerbots-bots-train
   (all-scripted tables: nit, 2 shovers, maniac, station, random; no opponent GPU), 40 steps, ckpts every 10,
   concurrency 32: train_63646e72d9bc. Deleted opsd4-s20 + my base to free GPUs (redeploy base before sh evals).
   v4 s20 on sh-select (241 hands): score 9.14 vs base 8.845, busts 4.6 vs 13.6/100, weak call-offs 2.9 vs 15.4.
   00:28: freed 3 idle teammate deployments; base back (infer_6156223eb22d) + v4s30-eval copy (infer_6fecaa0f2f29)
   -> full sh-select eval of v4 s29 launching (bg). 1 GPU kept free for RL ckpt copies.
   (old) GRPO on pokerbots-sh-train (bust penalty grader), concurrency 16, after OPSD or when 6 GPUs free.
5. Deploy best checkpoints (in vals-hackathon-matt), leaks.py + paired score vs base on sh-select; best = candidate.

## Feature experiments (user asked: table talk, opponent memory, other edges; revert if no gain)
- Second harness/project for experiments: `vals-hackathon-matt2` (prj_9166711eab18), checkout
  ~/Worker/projects/vals-poker-matt2 (branch matt-project2), launchd com.mallhw.vals-harness2 (127.0.0.1:8785) +
  com.mallhw.vals-connector2. Has `system_hint` (policy-only prompt add-on) and lab/games.py.
- 00:50: 36 identical 12-hand games per variant (LLM base opponents + 1 shover), policy = v4s30-eval copy:
  control eval_aea14fba8798, talk eval_fdb75841326c, notes eval_d64b4bc20a1a, short eval_83d67c6df934,
  endgame eval_e1607639bcbf. Analyze: `uv run --project harness python lab/games.py --evals control=... talk=...`
  (run in vals-poker-matt2). Adopt a variant only if it beats control (place, chips, busts); then fold it into
  the hint (v5) and OPSD/RL it. Otherwise keep v4 — the live entry stays the fallback.
- Token-cap forced folds are rare in the arena (v4 s29: 0.28% of decisions; base ~0.5%); ~2.2% on shover tables.
- v4 s29 full sh-select benchmark (copy, 16 parallel): eval_44490d61c2c5.

## Candidate to submit
**BEST CHECKED CANDIDATE (01:45): vals-hackathon-matt/rl-s10** (infer_96d341a589e2; GRPO from v4 s29, step 10,
ckpt_52d6b4b511f4). sh-select 500 hands: score 9.397 vs v4 9.003, busts 4.4 vs 4.6, weak call-offs 3.4 vs 4.8,
fmt err 0.89% vs 3.2%, EV +1.32 vs +0.09 BB/hand (paired diff +1.23 +-1.33). play_local 17 decisions, 0 failed.
Submit as "Alea iacta est". Caveat: VPIP 11% (tight). Keep rl-s10 deployed; once submitted it becomes LIVE (never eval it).
**LIVE since ~00:08 Nov 15: vals-hackathon-matt/opsd4-s30 (infer_496116c2999d, OPSD v4 step 29) — submitted by the
user/teammate. NEVER delete it and NEVER eval it directly** (its sh-select eval eval_8c0e96ff4dad was stopped the
moment it went live; no arena games overlapped). To eval v4 s30, deploy a second copy of ckpt_db759b762b5d.
Submission name for every model: **Alea iacta est** (user's request).
- none of mine yet (arena entry = teammate's solver-spot step 20, infer_5fe0799751ef).
