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
- Deployments in use (keep updated): opsd4-s30 = LIVE arena entry ("Alea iacta est", since 02:28 Nov 15); rl-s10 = previous entry ("butter"); base = opponent for sh/games12 datasets;
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
- 01:55: v5 variant (counted per-seat notes, min-sample labels, bot fingerprints + counter-play, more steals;
  text in vals-poker-matt2/lab/make_games.py VARIANTS['v5']) on the same games: eval_832ce52f3280. If v5 beats
  control: add it to opsd/hint.py as HINTS['v5'] (v4 + this block), build OPSD data, train on top of the best RL ckpt.
- 02:05 RESULTS (19-24 clean games each; CIs ~+-0.5 place): control 2.96 / 220 chips / 17% busts; talk 2.90 / 240 /
  10% (model barely talked, no fold effect) DROP; notes 2.76 / 318 / 5% KEEP; short 3.26 / 209 / 16% DROP;
  endgame 2.62 / 272 / 8% KEEP. Plan: hint v6 = v4 + v5 notes block + endgame block; OPSD on best RL ckpt when GPUs free.
- Token-cap forced folds are rare in the arena (v4 s29: 0.28% of decisions; base ~0.5%); ~2.2% on shover tables.
- v4 s29 full sh-select benchmark (copy, 16 parallel): eval_44490d61c2c5.

## NOTE (03:27): pokerbots.ac can be DOWN ("database is unavailable"); lab/arena.py then reuses the last known status
(flag site_unavailable). While it's down the Chrome submitter can't verify the live entry — treat opsd4-s30 as live.

## AUTO-SUBMIT (since 02:35)
A separate Chrome-enabled session ("chrome browser tools check") runs at :13 every hour: if "BEST CHECKED CANDIDATE"
differs from the live arena entry, it submits it as "Alea iacta est" and appends a SUBMITTED line here. So the LIVE
ENTRY CAN CHANGE ANY HOUR: before deleting or evaluating ANY deployment, read the live one with
`python3 -I -c "import sys; sys.path.insert(0,'lab'); import arena; print(arena.team_status()['deployment'])"` and treat
both the live entry and the BEST CHECKED CANDIDATE as untouchable. Keep exactly one BEST CHECKED CANDIDATE line.

## IDEA BACKLOG (user 03:05: "play like Vals but with notes; keep putting suggestions, see what sticks, keep going")
Each check-in: if harness2 has < 3 evals running, launch the next untested idea as a games12 variant (system_hint
on the current best policy copy, 36 games, max-parallel 8); if GPUs allow (4 for OPSD, 6 for RL), train the current
best hint. Judge on chips + busts (paired, lab/games.py), sh-select score/busts for trained models; keep winners,
drop losers, write one line per result below. Status: [ ] todo, [~] running, [x] kept, [-] dropped.
- [~] v10 "Vals + notes" (opsd/hint.py HINTS['v10']): OPSD train_6558f04b6d10 (pipeline_ckpt.sh -> v10-s20) +
      prompt test games12-v10 eval_d2411489bbab (first try failed on AC2: no space left).
- [~] v6 notes+endgame trained -> v6-s20 (infer_60c930ca75cb): sh-select eval_efddd596cc0f, games12 eval_80f62e7eb7eb; play_local 1 unreadable reply.
- [-] v7 notes-at-decision: best shover detection (13/36) but no chip gain (228 vs control 231), busts equal -> dropped.
- [ ] talk: chaos + bluff-talk (pipeline_talk.sh, after v7). [ ] v8 Accord-style + price-aware preflop (after talk).
- [ ] RL v2 realistic tables (pokerbots-rl2, bust_penalty 3) from the best OPSD ckpt (v10 or v6) when 6 GPUs free.
- [ ] Exploit Accord specifically (we fold 63 vs 25): 3-bet its opens wider in position (AJs+, KQs, 99+, A5s),
      call its c-bets with any pair; profile it via notes (aggressive/TAG).
- [ ] Small c-bet test: after raising preflop bet 1/3 pot on dry flops (LLMs fold to any bet ~50-60%).
- [ ] Bigger value bets vs calling stations (Stan, Martingale): 3/4-pot to pot on every street with top pair+.
- [ ] Button steal frequency: raise ~45% on the button when folded to (LLM blinds fold 80-90%).
- [ ] Cleanup-turn discipline: notes update in <= 150 words (avoid 10k-token cleanup timeouts / fallbacks).
- [ ] RL on multi-hand games (6-hand tasks, max_response_len ~48k) so scratchpad use is rewarded directly.
- [ ] More OPSD data/epochs for the winning hint (2 epochs, add sh-select + games12 traces of the best model).
- [ ] Endgame variants: chip-leader protection vs short-stack accumulation in last 6 hands.
- [-] v5 strict-evidence notes (worse: place +0.44, shover flagged 3/36). [-] short reasoning. [-] model-chosen talk
      (barely talked). [-] RL on 2-shover bot tables (butter: arena busts 22%).

## ITERATION PROTOCOL (user, 02:30 Nov 15 — run every check-in, no human in loop)
1. ANALYZE: `python3 -I lab/scouting.py` (per-opponent type + how to beat it; worst matchups first), `python3 -I lab/arena.py` (per-entry: place, busts, EV, VPIP/PFR, fallbacks), `python3 -I lab/vs_opponents.py
   --entry <live submission name>` (chips/folds/showdowns vs each opponent), bust hands of the live entry, and the sims
   (lab/leaks.py on sh-select evals; lab/games.py on games12 evals: place, chips, busts, notes accuracy).
2. OPINION + HYPOTHESES: write 2-3 lines in "Iteration log" below: what is the live model doing wrong, ranked by
   chips lost, and one concrete change per leak (prompt/hint text, training data, reward, opponent mix).
3. TEST: prompt-level first (games12 variant via system_hint on harness2, same 36 games, policy = a copy of the
   current best, max-parallel 8); training-level (OPSD with the winning hint; RL with fixed tables) only for winners.
4. DECIDE (keep / revert / explore): games12 place needs 100+ games to separate variants (CI +-0.6 at 18-36), so judge
   sims by CHIPS + BUSTS (+ sh-select score), place only as tiebreak; compare PAIRED on games all variants completed.
   Adopt a change only if it beats the current best on the sims AND doesn't raise
   busts; in the arena, judge an entry after >=20 games on busts + EV + place. If a live entry is worse than the
   previous one after 20+ games, REVERT the candidate to the previous one. If 2 iterations in a row fail on a
   feature, explore a different feature. Priority (user): opponent notes / modeling.
5. RECORD: experiments.jsonl + Iteration log; set "BEST CHECKED CANDIDATE" (the Chrome session submits it as
   "Alea iacta est" when it differs from the live entry).

## Iteration log
- 02:30 #1 ANALYSIS: butter (RL s10) after 22 arena games: bust 23% (5) vs v4 10%, 0 wins, place 3.64 vs 3.48,
  EV +22 vs +29. Busts: JJ called a shove 4-way, T6o short-stack call, 77 vs Bob, QTs flop raise war, AA<KK.
  CAUSE: RL bot tables had 2 any-two shovers -> learned to call shoves wide; bust penalty too small vs bots.
  vs opponents (105 games): win vs passive (Martingale +400, Stan +400, Evaluators +218, LoRA +132); lose to aggressive
  (Carlos -188, we fold 16 vs 4; Accord -141 12 vs 2; Vals -109; Friends -124).
  DECISION: REVERT candidate to v4 (opsd4-s30). NEXT: (a) notes v7 = notes + consult-the-notes-when-facing-a-bet +
  defend vs identified maniacs/aggressors; (b) RL v2 on realistic tables (1 shover, LLM opponents, multiway shoves,
  bust_penalty 3) from the best OPSD ckpt; (c) v6 (notes+endgame OPSD) results from pipeline_v6.
  02:33: v7 prompt test launched: games12-v7 = eval_10bc6af2a4e4 (vals-poker-matt2 lab/make_games.py VARIANTS v7). rl-s40 sh-select = eval_162f5d4531cc.

- 02:30 check-in: v4 (opsd4-s30) is LIVE again (Chrome session submitted 02:28 as "Alea iacta est"). butter final:
  23 games, bust 22%, 0 wins. Built pokerbots-rl2-* (base LLM opps + 1 shover seat2 + 1 maniac seat4,
  grader_params bust_penalty 3; rl/make_dataset.py --grader-params). NEXT when v6 is evaluated: RL v2 =
  `rl/train.py --prefix pokerbots-rl2 --from-checkpoint <v6 ckpt if it beats v4, else ckpt_db759b762b5d>
  --steps 30 --no-eval --concurrency 16 --checkpoint-every 10`. Free GPUs for it: v6 training ends (4) + delete
  rl-s40 after its eval + rl-s10 (butter, retired; NOT live — re-check arena status first).

- 02:40: TALK experiments (user asked): harness2 `policy_talk` injects table talk on the policy's actions; variants
  games12-chaos (random trash talk/nonsense, 80%) and games12-bluff-talk (strength claims, 100%) launch automatically
  after v5+v7 finish (pipeline_talk.sh). Judge by games.py (opp fold after talk vs silent, place, chips). If it helps,
  train the model to talk (OPSD with a talk hint); otherwise drop. SCOUTING: lab/scouting.py each check-in.

- 02:45 TEAM GAP ANALYSIS (shared games): Accord place 2.97, win 30%, bust 8%, EV +44, VPIP 27/PFR 17, 3bet 7%,
  fold-vs-raise 81%; Vals 3.08, bust 2%(!), tight 14/10; v4 3.48, win 12%, bust 10%, 14/10, 3bet 4%, fold-vs-raise 89%.
  Gap ~0.5 place = (a) busts 10% -> Vals ~2-3% and (b) Accord-style aggression (more hands, more 3-bets, fold less).
  QUEUED v8 = "Accord-style" games12 variant: open ~22-25%, 3-bet 7-8% (QQ+/AK + some A5s/KQs bluffs in position),
  fold to raises ~80% (defend pairs/suited broadways in position), keep all survival rules (v4 sections 2-3) + v7
  multiway rule. Appended to pipeline_talk.sh; CHECK-IN BACKSTOP: if no "games12-v8 eval" line in
  ~/Worker/logs/pipeline_talk.log after the talk evals launched, upload + launch v8 manually (make_games.py --variants v8).

- 02:55 PREFLOP OVER-FOLDING: v4 folds 90% to raises <=3BB (same as to all-ins), BB folds 85% vs small raises,
  folded AK/KQs/99 cheaply. Price-blind; survival rules over-generalized. v8 now includes price-aware preflop + BB
  defense. If v8 wins: fold its text into the next OPSD hint; consider RL tasks weighted toward small-raise spots.

- 03:00 v5 RESULT: strict evidence-based notes were WORSE (paired 18 games: place +0.44 vs control, flagged the
  shover in 3/36 games). Simple notes best on chips (+46%) and busts (1 vs 3). NEXT HINT (v9) if v6 underperforms:
  v4 + SIMPLE notes block (games12 'notes' text) + endgame + v8 price-aware preflop; OPSD from v4 with
  --include-cleanup on fresh games12-control traces.

## Overnight chain (02:15)
- OPSD v6 (hint v6 = v4 + v5 notes + endgame; opsd/hint.py HINT=v6) train_3866513541d5 from v4 ckpt (rl-s10 ckpt
  not available on the OPSD cluster). lab/pipeline_v6.sh (log ~/Worker/logs/pipeline_v6.log) then deploys v6-s20,
  play_local, evals sh-select (compare rl-s10 9.397) and games12-control in matt2 (compare control v4 2.96 place).
- RL finished (40 steps, reward plateau ~9.5-9.8). rl-s40 deployed + sh-select eval to compare with rl-s10.
- Decide in the morning: best of rl-s10 (live "butter"), rl-s40, v6-s20 by sh-select score/busts + games12 place.

## Candidate to submit
**BEST CHECKED CANDIDATE (02:30, REVERT): vals-hackathon-matt/opsd4-s30 (v4, infer_496116c2999d)** — butter (rl-s10)
busts 2x more in the arena (23% vs 10% after 22 games, RL overfit to bot tables). Submit as "Alea iacta est".
**SUBMITTED 2026-11-15 02:28 PT by the Chrome session: opsd4-s30 (infer_496116c2999d) as "Alea iacta est" — now LIVE (console shows Current entry). NEVER delete or eval it.** rl-s10 ("butter") is no longer live; keep it deployed unless GPUs are needed.
**LIVE since ~01:55: rl-s10 (infer_96d341a589e2) submitted as "butter". NEVER delete or eval it.** Fallback:
opsd4-s30 (v4, infer_496116c2999d; 67 games: place 3.46, bust 10%, EV +30 bb/100) stays deployed.
**SUPERSEDED (was candidate 01:45, now live as butter): vals-hackathon-matt/rl-s10** (infer_96d341a589e2; GRPO from v4 s29, step 10,
ckpt_52d6b4b511f4). sh-select 500 hands: score 9.397 vs v4 9.003, busts 4.4 vs 4.6, weak call-offs 3.4 vs 4.8,
fmt err 0.89% vs 3.2%, EV +1.32 vs +0.09 BB/hand (paired diff +1.23 +-1.33). play_local 17 decisions, 0 failed.
Submit as "Alea iacta est". Caveat: VPIP 11% (tight). Keep rl-s10 deployed; once submitted it becomes LIVE (never eval it).
**LIVE since ~00:08 Nov 15: vals-hackathon-matt/opsd4-s30 (infer_496116c2999d, OPSD v4 step 29) — submitted by the
user/teammate. NEVER delete it and NEVER eval it directly** (its sh-select eval eval_8c0e96ff4dad was stopped the
moment it went live; no arena games overlapped). To eval v4 s30, deploy a second copy of ckpt_db759b762b5d.
Submission name for every model: **Alea iacta est** (user's request).
- none of mine yet (arena entry = teammate's solver-spot step 20, infer_5fe0799751ef).
