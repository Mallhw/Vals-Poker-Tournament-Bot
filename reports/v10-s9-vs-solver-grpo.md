# v10-s9 vs solver-GRPO round 4 step 10: what's different, what each does well

Arena data through 2026-11-15 ~10:45 PT (public match logs on pokerbots.ac, team pineapple). Both models are
Qwen3.6-35B-A3B fine-tunes playing 6-max NLHE, 30-hand elimination games, 100 BB starting stacks.

| | **v10-s9** ("Alea iacta est") | **solver-GRPO r4 s10** (live entry now) | v4 / opsd4-s30 (reference) |
|---|---|---|---|
| Deployment | `vals-hackathon-matt/v10-s9` (`infer_f617937bbaa3`) | `vals-hackathon/ho-ee668d-step10` (`infer_7fe0e70e868b`) | `infer_496116c2999d` |
| How it was trained | OPSD (prompt distillation) from the v4 checkpoint on the "v10" coach hint: play like Vals (tight, raise-only, never bust) + keep one note line per opponent in the scratchpad. Step 9 of 20 (final save failed; step 19 was tested and was worse). | GRPO on single decision spots (`pbspot*` datasets, 3000 train / 300 select), reward = agreement with a solver; 61.7% agreement on the select set. *(From the run and dataset names. Nathan, please correct.)* | OPSD on the survival hint, step 30 |
| Arena games | 38 | 56 | 294 |

**Sample size:** at ~40-55 games a mean place has a 95% interval of about ±0.45, so differences in place below
~0.5 are not significant yet. Style stats (VPIP, fold-to-raise, …) are based on 1,100-1,600 hands and are solid.

## 1. Results

| | v10-s9 | solver-GRPO | v4 |
|---|---|---|---|
| Mean place (lower is better) | 3.39 | **3.38** | 3.37 |
| Place 1/2/3/4/5/6 | 6 / 4 / 12 / 7 / 3 / 6 | 11 / 8 / 12 / 8 / 8 / 9 | 34 / 71 / 56 / 56 / 40 / 37 |
| Win rate | 15.8% | **19.6%** | 11.6% |
| Top-2 rate | 26.3% | 33.9% | **35.7%** |
| Last-place rate | 15.8% | 16.1% | **12.6%** |
| Bust rate | 5.3% | 12.5% | **4.8%** |
| Net bb/100 | -12.2 | **+45.5** | +5.2 |
| All-in-adjusted EV bb/100 | -2.1 | **+17.4** | +11.1 |
| Failed/forced moves | 1.6% | **0.4%** | 1.8% |

**Average place is tied.** solver-GRPO wins more and makes far more chips; v10-s9 busts less than half as often.
But **both finish last 16% of the time, and mostly without busting**: 4 of v10-s9's 6 last places and 7 of
solver-GRPO's 9 were games where it survived but ended as the shortest stack. (A bust only means last place if nobody
else has busted yet; 5 of solver-GRPO's 7 busts still finished 3rd-5th.) Fading to the shortest stack is the
bigger shared leak.

## 2. Style

| | v10-s9 | solver-GRPO | v4 |
|---|---|---|---|
| VPIP (hands played) | 10.8% | **23.1%** | 14.9% |
| PFR (hands raised) | 9.7% | 11.4% | 9.9% |
| Limps/cold calls (VPIP − PFR) | 1.1% | **11.7%** | 5.0% |
| Fold to a preflop raise | **95.5%** | 79.3% | 88.9% |
| 3-bet when facing a raise | 2.5% | 4.7% | 4.0% |
| Steal (raise when folded to on CO/BTN/SB) | 20% | 23% | 22% |
| Flop c-bet after raising preflop | **86%** | 65% | 86% |
| Showdowns per 100 hands | 3.1 | 6.3 | 3.7 |
| Won at showdown | 50% | **67%** | 57% |
| Calls of ≥ half the stack, per 100 hands | 0.18 | 1.07 | 0.23 |
| Bob the Buster all-ins called / faced | 0 / 16 | 14 / 61 | 12 / 76 |
| Net chips on those calls | 0 | **+1,223** | +189 |

- **v10-s9 is a raise-or-fold nit.** It almost never limps or calls preflop, c-bets 86%, and never puts its stack in
  without a premium hand. It folds 95.5% of the time to a raise, which a thinking opponent can exploit by raising
  every pot.
- **solver-GRPO is loose-passive preflop, sticky post-flop.** It plays twice as many hands, about half of them by
  limping or calling, defends against raises much more, and goes to showdown twice as often. It wins those showdowns
  (67%), so its calls are mostly good, but it takes all-ins for its whole stack when they're merely +EV.

## 3. Who each one beats (net BB in shared pots, split between the opponents in the pot)

| Opponent | v10-s9 | solver-GRPO |
|---|---|---|
| Bob the Buster (any-two shover) | -11.5 | **+418.9** |
| 🥵🥵🥵 | -19.2 | **+187.1** |
| Accord (strongest LLM team) | **-112.8** | **+114.4** |
| The Evaluators | -29.7 | +83.1 |
| Vals | -12.8 | +55.2 |
| Martingale | -88.2 | +22.7 |
| Stubborn Stan (calling station) | **+52.8** | -41.1 |
| LoRA | **+60.8** | -58.7 |
| Tilted Tim | -1.2 | -52.4 |
| Chatty Carlos | +1.2 | -47.1 |
| Friends | -22.7 | -48.4 |

The two models are close to mirror images. **solver-GRPO beats the aggressive players** (shovers, Accord, Vals):
it calls their raises and all-ins with good-enough hands and wins the showdowns. **v10-s9 beats the passive
players** (Stan, LoRA), who pay off its value bets, but gets run over by aggression (Accord -113: it folds to almost
every raise). Per-opponent numbers rest on few pots; treat them as direction, not size.

## 4. Why solver-GRPO busts (7 busts in 56 games)

| Hand | Stack | What happened | Verdict |
|---|---|---|---|
| A3o | 100 BB | called Bob's preflop all-in, lost to 88 | loose |
| QJo | 100 BB | called Bob's preflop all-in, lost to 95o | marginal |
| J7o | 100 BB | **limped, then called** Bob's all-in, lost to QTo | bad |
| 66 | 82 BB | checked to the river, **called Bob's 85×-pot river shove** on K-Q-J board, lost to Q6 | bad |
| QQ | 27 BB | shoved, called by A5, ace on the river | fine, unlucky |
| K7o | 14 BB | re-shoved over Chatty Carlos's raise, ran into A4 | loose but OK short |
| 99 | 100 BB | three-way all-in **after another player had already called** the shove, lost to T5's straight | marginal |

4 of 7 are calls of Bob the Buster's all-ins. Against an any-two shover A3o/QJo are slight favourites, so calling
him is +chips (that's where most of its +1,223 and many of its wins come from). In an elimination format, though,
a 55/45 for the whole stack means elimination almost half the time (its 7 bust games finished 3rd once, 4th or 5th four times, 6th twice).

## 5. Strengths

**v10-s9**
- Survival: 5% busts, 0.18 big calls/100, never calls an any-two shove.
- Discipline: raise-or-fold preflop, 86% c-bet; few marginal decisions.
- Beats passive opponents and calling stations.
- Opponent notes in the scratchpad (one line per seat).
- Best score on our survival benchmark (sh-select 9.106 vs v4 9.003; 0.6 busts/100 hands vs 4.6).

**solver-GRPO r4 s10**
- Chips and wins: +17 EV bb/100, 20% win rate, highest of any pineapple entry.
- Defends against raises (79% fold vs 95%) and isn't exploitable by constant aggression.
- Beats the strongest aggressive opponents (Accord, Vals, Bob, 🥵🥵🥵).
- Wins showdowns (67%): its calls are well chosen in general.
- Cleanest output: 0.4% failed moves (v10-s9 1.6%).

## 6. Weaknesses → what to change

| Model | Weakness | Fix to try |
|---|---|---|
| solver-GRPO | Calls all-ins for the stack when merely +EV (4 of 7 busts vs Bob) | Bust-aware reward (e.g. our `bust_penalty` in `grading.py`), or a rule: call an any-two shover only with 77+, A9s+, ATo+, KQ; never after limping; never multiway; post-flop all-in calls only with top pair good kicker+ |
| solver-GRPO | Limps 11.7% of hands | Raise-or-fold preflop outside the big blind |
| solver-GRPO | Loses to calling stations (Stan, LoRA) | Value-bet thinner and bigger; fewer bluffs into stations |
| v10-s9 | Folds 95.5% to raises; Accord takes -113 BB | Defend the big blind and in position (pairs, suited broadways, Ax suited) |
| both | Fade to last without busting (v10-s9 4 of 6 last places, solver-GRPO 7 of 9) | Pick up blinds: steal more on CO/BTN (20% now) and when short (≤15 BB) shove instead of folding |
| v10-s9 | Never calls Bob (0/16): leaves his chips to others | Same shover call range as above |

**Combining both:** solver-GRPO's preflop defence and showdown judgment, plus v10-s9's bust rules (and its
opponent notes so it can tell a shover from a normal raiser), should beat both. Two ways to get there:
(a) GRPO from the solver-GRPO checkpoint on the realistic `pokerbots-rl2` tables (LLM opponents + a shover +
a maniac) with the bust penalty in the reward; (b) OPSD from solver-GRPO with a short rules hint (shover call range,
no limp-calls, no multiway all-ins).

## 7. In flight (Matt's project, 10:45)

- **v12 prompt test**: v10 + the shover call range above (77+, A9s+, ATo+, KQ vs an identified shover), paired
  with v10 on the same 36 games with a shover seat. Evals `eval_1e3b7fd45eff` (v12) vs `eval_da78b93972ac` (v10).
- **RL v3**: GRPO from v10-s9 on `pokerbots-rl2` (`train_f47a076b7b24`), waiting for GPUs.

## Questions for Nathan

1. What exactly does the solver reward score: agreement on action only, or action + size? Which solver/spots?
2. Which checkpoint did round 4 start from, and does any round include a stack-risk term?
3. Would you be up for running (a) above from your checkpoint? We can share the `pokerbots-rl2` datasets and the
   `bust_penalty` grader params.

---
*Method: `lab/compare.py` on the `matt-project` branch (reads the public match JSONs). Bust hands from the match
logs. Opponent nets split each pot's result evenly among the opponents who put chips in.*

---

## 8. Update 13:30 PT: the talk and final-round models (team now #2, rating 1191)

| | **talk round s20** (any-talk penalty) | **final round s20** (small-pot reward, live) | solver-GRPO r4 s10 | v10-s9 |
|---|---|---|---|---|
| Games | 76 | 27 | 76 | 38 |
| **Mean place** | **2.93** | 3.19 | 3.41 | 3.39 |
| Win / top-2 / last | **26%** / 42% / **9%** | 26% / **44%** / 15% | 18% / 33% / 17% | 16% / 26% / 16% |
| Bust rate | 5.3% | **3.7%** | 10.5% | 5.3% |
| EV bb/100 | +28.7 | **+67.4** | +17.0 | -2.1 |
| VPIP / PFR | 27.5% / **21.2%** | 21.0% / 16.7% | 21.4% / 10.9% | 10.8% / 9.7% |
| Limps/cold calls (VPIP−PFR) | **6.3%** | 4.3% | 10.5% | 1.1% |
| 3-bet facing a raise | **8.5%** | 6.8% | 3.9% | 2.5% |
| Fold to a raise | 78% | 85% | 82% | 96% |
| Steal when folded to (CO/BTN/SB) | **47%** | 32% | 23% | 20% |
| Flop c-bet | 74% | 64% | 63% | 86% |
| Won at showdown | 66% | **77%** | 68% | 50% |
| Bob all-ins called / faced | 6/29 (+191) | 3/16 (+189) | 15/63 (+1,426) | 0/16 |
| Failed moves | 1.2% | 0.7% | 4.6% | 1.6% |

**Why the talk model is the best entry so far:** it plays the "Accord style" (the strongest LLM team) that we
tried and failed to get from prompting alone (v8): raise-first instead of limping, 3-bets twice as often as
solver-GRPO, steals the blinds almost half the time, and keeps pressure with c-bets. It finishes last only 9% of
the time, which fixes the fading leak from section 5: by stealing blinds it never drifts to the shortest stack.
And it calls Bob far more selectively than solver-GRPO (6 of 29 vs 15 of 63), so it busts half as often.

**It beats** Lincoln (+222 BB), Accord (+212), Martingale (+133), Stan (+91), Bob (+80).
**It loses to** 🥵🥵🥵 (-141), LoRA (-87), Spark (-55), Chatty Carlos (-50), Professor Pete (-47); these are
aggressive or tricky players who fight back against steals. Its busts come early (average hand 7.5 of 30).

**final round s20** is tighter (VPIP 21%) and wins more of its showdowns (77%). Early numbers are excellent, but
27 games is too few to say it beats the talk model (place 95% interval ±0.6).

**Suggestions for the next round:**
1. Keep the talk-round s20 deployment (`infer_a5aaffeb4304`) as the fallback; don't delete it. If final round s20
   is above ~3.2 mean place after 40+ games, switch back.
2. Next training round from the talk model: add a penalty for early busts (its busts come at hand ~7, when a fold
   costs nothing), and practise spots against re-stealers (🥵🥵🥵, LoRA, Spark), e.g. 4-bet or fold, not call.
3. Retire v10-s9 as a candidate: everything the talk model does differently (steal, 3-bet, defend) is what v10-s9
   is missing.

## 9. Update 14:30 PT: final round s20 collapsed; talk round s20 is back

final round s20 (small-pot reward) after 60 games: mean place **3.78**, 15 last places, but only 2 busts.

| final round s20 | games | mean place | last places | busts | net BB/game | VPIP / PFR | failed moves |
|---|---|---|---|---|---|---|---|
| first 27 (12:42-13:12) | 27 | 3.04 | 3 | 0 | +25.3 | 20.9% / 16.9% | 0.7% |
| next 33 (13:13-13:47) | 33 | **4.39** | **12** | 2 | -13.7 | 20.5% / 16.7% | 0.6% |

Same style, same failure rate, similar opponents. 10 of the 12 late last places were **fades, not busts**: the
small-pot reward seems to teach it to avoid big pots, so when it falls behind it never wins a big pot back. The
early 3.04 was probably partly luck (it also busted 0 times in 27 games). Team rating fell 1191 -> 1059 (#2 -> #12).

The live entry is back to talk round s20 (16 more games, mean place 3.06; 92 total at ~2.95). **Keep it.**
For the next round: train from talk round s20, and don't reward small pots; reward finishing position (e.g. our
`bust_penalty` plus a penalty for ending as the shortest stack).
