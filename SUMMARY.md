# Hackathon summary: team pineapple, AC × Vals Pokerbots (Nov 13-15, 2026)

We fine-tuned **Qwen3.6-35B-A3B** to play 6-max no-limit hold'em in the
[pokerbots.ac](https://www.pokerbots.ac) arena: 30-hand elimination games, 100 BB stacks, one `submit_action`
tool call per decision, a 5,000-token budget per move and a private 5,000-character scratchpad that persists
across hands. Training and serving ran on Applied Compute (AC2); the poker harness ran on our own machine.

## Result

The ranked ladder froze at ~14:07 PT on Nov 15.

| | |
|---|---|
| Final position | **8th of 11 teams** (11th of 17 entries incl. house bots) |
| Rating | 1059 (OpenSkill), 767 ranked matches, +7.4 bb/100, luck-adjusted EV +11.8 bb/100 |
| Peak | **#2, rating 1191** (13:30 PT Nov 15) |
| Winner | Accord, 1238 |

The peak came from a teammate's "talk round s20" model; a different model that went live just before the freeze
collapsed in its last 33 games and cost ~130 rating points an hour before ranked play stopped.

## Our arena entries

| Entry | How it was trained | Games | Mean place | Win | Last | Bust | EV bb/100 |
|---|---|---|---|---|---|---|---|
| **talk round s20** (teammate) | GRPO round with a table-talk penalty | 92 | **2.95** | 26% | 9% | 5% | +28 |
| v4 / opsd4-s30 ("Alea iacta est") | OPSD on a survival hint (`opsd/hint.py` v4) | 294 | 3.37 | 12% | 13% | 5% | +11 |
| v10-s9 ("Alea iacta est") | OPSD on "play like Vals + keep opponent notes" (v10) | 38 | 3.39 | 16% | 16% | 5% | -2 |
| solver-GRPO r4 s10 (teammate) | GRPO on solver-agreement spots | 76 | 3.41 | 18% | 17% | 11% | +17 |
| final round s20 (teammate) | GRPO with a small-pot reward | 60 | 3.78 | 13% | 25% | 3% | +19 |
| butter / rl-s10 | GRPO on all-bot tables with two any-two shovers | 23 | 3.6 | 0% | — | 22% | +22 |

Mean place is out of 6 (lower is better); with ~40-60 games a mean place has a 95% interval of about ±0.5.

## What we learned

1. **Survival is necessary but not sufficient.** Our prompt-distilled models (v4, v10) cut busts from 13.6 to
   0.6-4.6 per 100 hands on our shove benchmark, but they fold 89-96% of the time to a raise and finish last by
   *fading*: 4 of v10-s9's 6 last places were games it survived with the shortest stack.
2. **Aggression won.** The best entry (talk round s20) played like the strongest LLM team (Accord): raise-first,
   3-bet 8.5%, steal the blinds 47% of the time when folded to, rarely limp. It finished last only 9% of the time.
   See [`reports/v10-s9-vs-solver-grpo.md`](reports/v10-s9-vs-solver-grpo.md), sections 8-9.
3. **Calling the any-two shover (Bob the Buster) is +chips but -place.** solver-GRPO netted +1,426 chips on
   15 of 63 such calls, but 4 of the 7 busts we examined came from them. In an elimination format a 55/45 for the stack is a
   bad deal unless it's a real favourite.
4. **RL on unrealistic tables overfits.** GRPO on all-bot tables with two shovers ("butter") learned to call shoves
   wide and busted 22% of games. We then built realistic tables (`pokerbots-rl2`: LLM opponents + one shover + one
   maniac, bust penalty 3), but our runs there were preempted four times before their first checkpoint.
5. **Reward shaping matters more than anything else.** The starter reward (raw EV) let one lucky all-in dominate.
   We reshaped it (signed-log EV, loss weight 1.25, bigger failure gap) and added a bust penalty
   (`harness/src/poker_harness/grading.py`). A small-pot reward (teammate's final round) taught the model to avoid
   big pots and it faded.
6. **Prompt-only tweaks stop working on a fine-tuned model.** On v10-s9, the v8 (aggression), v12 (shover call
   range) and v13 (steals, push-or-fold) prompts all came out as noise on paired games; v13's rules were ignored
   (preflop raise rate went *down*). New behaviour has to be trained in.
7. **Opponent notes help.** In paired 12-hand games, a simple "one line per seat" scratchpad format beat the
   default (318 vs 220 chips, 1 vs 3 busts); stricter structured notes did worse. Including scratchpad turns in
   OPSD training confused the model's tool use (5.6% format errors) until we dropped them.
8. **Table talk does nothing.** Opponents' fold rates after our trash talk vs silence: 54% vs 52%; chaotic talk
   slightly hurt.
9. **Operations matter.** Evaluating the live arena deployment overloaded its single GPU and caused forced folds
   in real games; deleting a deployment an eval was using broke ~14% of decisions; preemptible AC2 runs were
   stopped by teammates' jobs. Rules: never eval the live entry, check usage before deleting, checkpoint often.

## Timeline (Nov 14-15, PT)

- **Nov 14 evening:** stripped the starter to bare bones; reward reshaping + bust penalty; diagnosed the arena
  leak (13.6 busts/100 hands at shover tables); OPSD with survival hints v3/v4. v4 went live at 00:08 → team #8.
- **Nov 15 00:00-03:00:** GRPO on bot tables ("butter", reverted after 22% busts); opponent profiles and scouting
  reports; feature A/B tests on paired 12-hand games (notes ✓, endgame ✓, talk ✗, short reasoning ✗).
- **03:00-08:00:** OPSD v6 (notes + endgame; format errors) and v10 ("Vals + notes"); v10-s9 best on the shove
  benchmark (score 9.106, 0.6 busts/100) → live 08:18, team #5.
- **08:00-12:30:** teammates' solver-GRPO and talk models take over the entry; prompt tests v12/v13 (null);
  RL v3 from v10-s9 preempted four times.
- **12:30-14:07:** talk round s20 → team **#2 (1191)**; final round s20 collapses; ranked ladder freezes at 1059.

## What's in this repo

| Path | What |
|---|---|
| `harness/` | Poker server, grader (reshaped reward + bust penalty), scripted opponents incl. a `shover`, `system_hint` and table-talk injection for prompt tests |
| `rl/` | GRPO datasets and launcher (`--from-checkpoint`, scripted seats, per-task grader params) |
| `opsd/` | Prompt distillation: `hint.py` (hints v2-v10), dataset builder (`--include-cleanup` for scratchpad turns) |
| `vals_tools/` | Deploy / eval / doctor helpers |
| `lab/` | Experiment framework: arena tracker (`arena.py`), entry comparison (`compare.py`), leak analysis (`leaks.py`), opponent profiles and scouting, paired game tests (`make_games.py`, `games.py`), automation pipelines, the lab notebook (`LAB.md`) and experiment log (`experiments.jsonl`) |
| `reports/` | v10-s9 vs solver-GRPO vs talk/final-round analysis |

The trained weights lived on AC2 and are not included (35B parameters; AC2 access ended with the event).
The lab scripts assume our local paths and AC2 project names; adapt them before reuse.

Team pineapple: [@Mallhw](https://github.com/Mallhw), [@nzhao721](https://github.com/nzhao721) and teammates. Built on the organizers' starter, [Applied-Compute/pokerbots-starter](https://github.com/Applied-Compute/pokerbots-starter).
Dashboard: [matthewli.org/pokerbot-lab](https://matthewli.org/pokerbot-lab).
