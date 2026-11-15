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

HINTS = {
    "v2": """Coach's notes for this game. Follow them.

1. Read the spot, privately, before deciding. In your reasoning (never in your reply), first note:
   my hand, the board, what I have, the pot, the amount to call, and my stack.
   If the amount to call is more than 0, checking is not legal: your choices are fold, call or raise.
   Keep your reasoning short. Your reply is exactly one submit_action call and nothing else. A bet or raise must
   include an amount (the total you are betting to); fold, check and call take no amount.

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

6. Table talk is optional. If you talk, say nothing about your cards.""",
    "v3": """Coach's notes for this game. Follow them.

1. Read the spot privately first. In your reasoning (never in your reply) note: my hand, board, what I have, pot,
   amount to call, my stack, and "busted if I lose: yes/no". If the amount to call is more than 0, checking is not
   legal: choose fold, call or raise. Keep reasoning short. Your reply is exactly one submit_action call. A bet or
   raise needs an amount (the total you bet to); fold, check and call take none.

2. Survive. Busting ranks you below every player still seated, and folding every hand costs only ~15 chips a game,
   so a short stack is never desperate. Any call or shove that can bust you needs about 15-20 points MORE equity
   than the pot odds say. In the last few hands, if you are not last in chips, take no marginal all-ins.

3. Facing an all-in or a raise of 25 BB+ before the flop (you would put in 40%+ of your stack):
   - unknown or tight player (most bots that shove big have it): call with AA, KK only.
   - a player your notes show shoving very often (e.g. 3+ shoves, showed weak hands): 99+, AJs+, AQo+.
   - someone has already called the shove: KK+ only.
   - short (20-40 BB): QQ+, AK vs tight; TT+, AQ+ vs a frequent shover.
   Offsuit junk (T9o, J8o, Q8o, K6o, 96o) and weak aces never call off, whoever shoved.

4. Before the flop raise or fold; flat-call rarely. Open: first seat 66+, ATs+, KJs+, QJs, AQo+; middle seat add
   55, A9s, A5s, KTs, QTs, JTs, AJo, KQo; cutoff/button wider. Facing a raise: 3-bet QQ+, AK; call JJ-77, AQs, AJs,
   KQs only in position; fold the rest. Facing a 3-bet: continue with QQ+, AK. Don't open-shove 20 BB+.

5. After the flop, stop paying off. One pair facing big bets on turn and river, a check-raise, or any river raise:
   fold. Two pair or a set facing a river raise when the board has four to a straight, four of a suit, or a pair:
   call small at most, never re-raise. Other bots rarely bluff big: treat their big bets and raises as strong.
   Value-bet your strong hands bigger; bluff rarely, they call too much.

6. Scratchpad: per seat, count hands played, raises, all-ins, and cards shown ("seat3: 4 all-ins in 7 hands,
   showed 33, Q7o = frequent shover"). Track your stack and hands left.

7. Table talk: say nothing about your cards.""",
}
HINT = HINTS["v3"]
