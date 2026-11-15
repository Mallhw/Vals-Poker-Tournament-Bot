# Final pick: final round step 20 vs step 30

2026-11-15. The two candidates for the final arena entry, compared on 900 held-out PokerBench spots, the 509 held-out
opening and c-bet spots, paired talk copies of the held-out spots, 200 held-out facing-a-shove spots, and the arena.

- **Step 20:** `ckpt_726f3f0dd6bc`, the final round (`train_36fa887620e6`, dataset `pbspot14`, started from round 4 step
  10 `ckpt_77ae64487b93`). Deployed as `vals-hackathon/ho-7620e6-step20` (`infer_1daa9f6ac6d3`), the arena entry since
  20:42 UTC.
- **Step 30:** `ckpt_9128a55961f0`, the 10-step extension of step 20 without the any-talk penalty
  (`train_01fc927b5d7d`, dataset `pbspot15`). Deployed as `vals-hackathon/ho-7b5d7d-step10` (`infer_ad05b325ab21`).

## Where step 20 is stronger

- **Postflop big calls where the solver folds:** 7 of 46 against 9 of 46. It's a small gap, within noise, but it's
  the main bust-risk measure.
- **Postflop solver match in big pots:** 58% against 53%.
- **Sway from weakness claims** ("I'm bluffing, call me"): it calls 4.0 points more often, a shift that isn't
  significant. Step 30 calls 6.7 points more, which is.
- **Discipline preflop:** fewer junk opens (2–3 per position against 7–9), less over-aggression (2% of spots against
  5%), and higher solver match in spots with 0-1 raises before hero (65.3% against 62.2%).
- **Arena evidence:** 27 matches, 16 at real-player tables, with no busts against real players and a mean place of
  2.81 there. Step 30 has none.

## Where step 30 is stronger

- **Opening:**

  | First in | Step 20 opens | Step 30 opens | Target |
  |---|---|---|---|
  | Cutoff | 22% | 37% | 40% |
  | Button | 56% | 67% | 65% |
  | Small blind | 26% | 50% | 60% |

  It also misses fewer hands it should open, and limps a little less from the small blind (5 against 8).
- **Less passive preflop:** it calls or checks where the solver raises in 4% of spots, against 7%.
- **Fewer bad big preflop calls:** 11 of 59 against 14 of 59.
- **Fewer bad shove calls:** 18 of 195 against 24 of 195.
- **C-bets:** they match the label 71% of the time against 47%, at a size slightly closer to the third-pot target
  (34% of the pot against 36%).

## Where they're the same

Both never talk or leak, both have very few format failures (1–2), and overall solver match is about equal (59.2% and
59.7%).

## In short

Step 30 is more aggressive in the right places preflop, steals more blinds and folds more to shoves. Step 20 is a
little more careful postflop, holds up a bit better against weakness claims, and has proven itself in live play.
