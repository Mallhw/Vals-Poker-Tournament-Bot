"""The prompted strategy hint that OPSD distills into the weights.

At the event the harness owns the system prompt, so a strategy you want the model to follow has to live in its
weights. OPSD's teacher sees this text appended to the system prompt; the student never does.

This hint targets the leaks found in pineapple's first 32 arena games (base Qwen3.6-35B-A3B, 862 hands; see
~/Worker/research/reports/Why pineapple is losing.md):

- 7 busts in 32 games (22%), every one from calling off a stack with a marginal hand (T9o vs a hand-1 shove, T6s vs
  a 77 BB raise, JTo calling a 5-bet jam, 86s raising into a flop re-raise). Elimination ranking puts a busted seat
  below everyone still seated, so survival is worth more than chips.
- Calling station: flat-called 26% of preflop raises but 3-bet only 9%; called down two or three streets with one
  pair (77 vs KK, AT on a four-flush board vs a set).
- UTG was its only losing seat: -307 BB over 120 hands (86s, 88, ATo opened, then called 3- and 4-bets).
- 1.4% forced folds: tried to check facing a bet, unreadable tool calls, and replies cut off at the 5,000-token cap.

Things it already did well (39% VPIP / 19% PFR, aggressive late-position steals, value betting) are left alone.
"""

SEPARATOR = "\n\n"

HINT = """Coach's notes for this game. Follow them.

1. Read the spot before deciding. Start your thinking with one line:
   "Me: <hand>, <position>. Board: <cards>. I have: <made hand / draw>. Pot <P>, to call <C>, my stack <S>."
   If "to call" is more than 0, checking is not legal: your choices are fold, call or raise.
   Keep the rest of your thinking short. Decide, then make exactly one submit_action call that matches the decision.

2. Survive. A busted seat ranks below every player still seated, so losing your whole stack is the worst result in
   this game, much worse than folding a decent hand.
   - Do not put more than 40% of your stack into one pot unless you hold a premium hand: before the flop QQ+ or AK;
     after the flop two pair or better, or top pair with a top kicker in a pot that was not re-raised before the flop.
   - Facing an all-in or a raise of 25 BB or more before the flop: call only with TT+ or AQ+. Against a player your
     notes mark as a frequent shover, also call with 88-99 and AJ. Fold everything else, suited or not.
   - Early in the game (the first orbit) avoid marginal all-ins entirely; you have 30 hands to find better spots.

3. Before the flop, raise or fold; call rarely.
   - From the first seat to act: open only 99+, AJs+, AQo+, KQs. Fold the rest.
   - Middle seats: add 77-88, ATs, KJs, QJs. Cutoff and button: open wider and steal the blinds, as you already do.
   - Facing a raise: re-raise with JJ+, AK, AQs; call with pocket pairs and suited broadway cards only when you are in
     position and the raise is small. Fold weak offsuit hands, small suited connectors and weak aces.

4. After the flop, stop paying off.
   - With one pair, do not call bets on both the turn and the river unless it is top pair with a good kicker.
   - When a passive opponent suddenly raises or bets big on two streets, they usually have it: fold medium hands.
   - Bet your strong hands for value; check and give up when you miss against more than one opponent.

5. Scratchpad. After each hand, keep short per-seat notes: hands played, raises, all-ins, and any cards they showed
   (for example "seat3: 5 all-ins in 8 hands, showed 33 and Q7o = shover"). Use these notes to decide who is
   bluffing and who is not, and keep your own stack and the hands remaining up to date.

6. Table talk is optional. If you talk, say nothing about your cards."""
