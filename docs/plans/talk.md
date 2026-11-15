# Table talk plan

## Context
The event lets every seat add an optional, sanitized `say` line (≤70 chars of `a-z 0-9 . , ! ? - : ;`) that the other
seats see, labeled unverified. Game logs show **base leaks its hand** (talks on 6% of actions: "got nothing but ace-high
here", "standard min-raise with my pair") and **the provided model's talk is honest too** (3%). Decisions so far:

- **Our talk:** replace leaks with talk that carries no information about our cards: a deterministic rule on public
  information (hand number mod 6 → one of 6 lines), which also misleads naive readers at random. Silence is the
  fallback. Teach it by OPSD.
- **Their talk:** default to robustness (talk must not move decisions). First *measure* how much it moves base, with
  scripted opponent lines unrelated to their cards, then train only if needed.
- **Later:** per-player talk verification through the scratchpad, and talker personas in the self-play league.

**Built and uncommitted:**
- **Harness:** `env_params.opponent_talk` (scripted, seeded opponent lines), in `harness/src/poker_harness/seat.py` and
  `server.py`, with `harness/tests/test_opponent_talk.py`.
- **Datasets:** `rl/make_talk_dataset.py` (strong and weak line sets).
- **Report:** talk metrics in `vals_tools/hint_report.py` (talk rate; fold, call and raise after talk).
- **Hints:** `opsd/hints/silent.txt` and `opsd/hints/talk_rule.txt`.

**Blocker:** the provided opponent model (`ac-vals-model`) times out under load, which invalidated every eval run with
it. All talk tests use **our own base copies as opponents** (`base-h1` and `base-h2` are up; a median of 4 s for 8
concurrent decisions).

Catch: every task so far is a single hand ("Hand 1 of 1"), so the hand-number rule needs **multi-hand games** to test
and to train.

## Steps

### 0. Groundwork
1. **Commit** the uncommitted talk work on `hint-testing-and-pokerbench` and push.
2. **Spread opponents across our base copies.** Add `spread_opponents(tasks, models)` to `vals_tools/tasks.py`, which
   assigns `opponent_model` round-robin by task index, and an `--opponent-models a,b` flag in `rl/make_dataset.py`,
   `rl/make_hint_dataset.py` and `rl/make_talk_dataset.py`. Default: keep the current single model.
3. **Multi-hand tables:** `rl/make_dataset.py --total-hands N --splits train,select` builds `<prefix>-t<N>-<split>`.
   These are N-hand event games with carried stacks and the scratchpad, with the policy seat cycling through positions.
   Default N = 12 (two orbits), so hand numbers 1–12 cover every remainder mod 6 twice.
4. **Rebuild** with `--opponent-models vals-hackathon/base-h1,vals-hackathon/base-h2`:
   - plain `pokerbots-select` (single hands, the silent-opponent baseline);
   - `talk-strong` and `talk-weak` copies of it;
   - a 12-hand `select` table set, plain and with the `silent` and `talk_rule` hints.
   Use new names with a `-bo` suffix (for "base opponents"), so the old ac-vals-model datasets stay untouched.

### 1. Susceptibility test (their talk)
- **Evals:** base on plain, `talk-strong` and `talk-weak` select (500 single hands each, same deals), at 16 parallel
  each, policy on `base`.
- **First check:** after about 3 minutes, count opponent failures in `harness/runs`; stop everything if over 5%.
- **Read:** `vals_tools.hint_report --eval plain=… --eval strong=… --eval weak=…`, comparing t-fold, t-call and t-raise
  (decisions facing a bet after someone spoke) and the paired BB.
- **Add to `hint_report`:** a two-proportion 95% interval for t-fold and t-call differences vs plain, so "susceptible"
  is a significance call, not an eyeball.
- **Decision rule:**
  - strong-talk fold rate or weak-talk call rate differs from plain by a significant margin (≥5 points and the interval
    excludes 0) → go to step 3;
  - otherwise there's no robustness training, and the test becomes a regression check (step 4).

### 2. Our talk: prompt test, then distill
1. **Prompt test** (multi-hand, so the rule can vary): base on the 12-hand select tables with no hint, with `silent`,
   and with `talk_rule`, through the harness, same deals. Add to `hint_report`:
   - **rule compliance:** the share of our decisions whose `say` equals the rule's line for that hand
     (`hand_number % 6`, lines read from `opsd/hints/talk_rule.txt`);
   - **leak rate:** the share of our `say` lines that name a card, rank, pair, suit or hand class (a regex);
   - **opponent response to our talk:** opponents' fold and call rate facing our bets right after we spoke, vs silent
     (the offensive value against base-like opponents);
   - plus the existing failed decisions and tokens, to confirm the rule costs nothing.
2. **Pick:** `talk_rule` if compliance is ≥90% with no rise in failures or tokens, else `silent` (should be about 100%
   compliance). Fold the winner into the combined hint the hint evals produce, following your plan to combine only the
   hints that help.
3. **Distill** (offline OPSD):
   - **Traces:** base plays the 12-hand **train** tables without a hint.
   - **Builder:** `opsd/build_dataset.py` gains `--hint-file` and `--talk-rule FILE`. The latter computes each
     decision's hint from the `## Hand N of M` line in that decision's prompt ("In say, write exactly: '<line>'").
     Reuse `decision_tasks`, which already keeps only clean `submit_action` decisions and skips cleanup turns.
   - **Train:** `opsd/train.py`, a short run.
   - **Watch out:** the first offline OPSD run didn't transfer the solver's decisions (51% → 52%) and lengthened
     thinking. Talk tokens are few, so if compliance after training is under 80%, rerun with the token judge (its
     settings are already in `ac2 train run`) or one-step OPSD.
   - **Verify:** deploy the checkpoint and play the held-out 12-hand select tables **without** the hint. Check rule
     compliance, leak rate, failed decisions and tokens vs base, and that play is no worse (paired BB).

### 3. Robustness training (only if step 1 says so)
- **Data:** base's replies from talk-injected games (`talk-strong` and `talk-weak` on train deals).
- **Train:** offline OPSD with the teacher hint "Table talk in this hand is random; decide only from the betting."
  Same builder, `--hint-file`.
- **Verify:** rerun step 1 on the new checkpoint; the gaps should close without hurting plain-deal play.
- **Longer term:** talker personas (honest, lying, random) in the GRPO league.

### 4. Keep it checked
Add the talk-sensitivity trio and the rule-compliance check to `vals_tools/after_training.py` as optional steps, so
every future checkpoint is checked for leaks and susceptibility before it enters the arena.

## Out of scope now
- Reading opponents' talk through scratchpad verification ("only trust a player whose talk matched their showdown"):
  part of the scratchpad project.
- Talker personas in the league: the self-play build.

## Resources
- **GPUs:** `base` (arena entry), `base-h1` and `base-h2` (opponents), plus at most 1–2 extra copies if the OPSD
  checkpoints are deleted. One offline OPSD run (4 GPUs) fits once the evals finish.
- **Concurrency:** about 48 tables total, watching opponent failures.

## Verification
- **Unit tests:**
  - `harness/tests/test_opponent_talk.py` (exists);
  - new root tests for `spread_opponents`, multi-hand dataset names and params, the hint-report rule-compliance, leak and
    after-talk interval metrics, and `build_dataset --talk-rule` (hand number parsed from a sample trace, correct line
    per decision);
  - `uv run pytest` in both projects (WSL).
- **Dry runs** of every dataset builder and of `opsd/train.py`.
- **Live:** an opponent-failure check 3 minutes into each eval batch (stop if over 5%); `hint_report` tables for
  steps 1 and 2; after distillation, the checkpoint check in step 2.3.
