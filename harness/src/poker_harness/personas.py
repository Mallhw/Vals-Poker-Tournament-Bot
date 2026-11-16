"""Play-style personas for opponent seats in self-play training (league entries with a `persona`).

A persona is appended to an opponent seat's system prompt, the way `system_hint` is appended for the policy seat. It
only steers how that opponent plays; the policy seat never sees it. These are our own style descriptions, not claims
about how arena opponents play.
"""

from __future__ import annotations

PERSONAS: dict[str, str] = {
    "tight_aggressive": (
        "Your play style for this game: tight and aggressive. Play few hands, mostly strong ones, and play them fast: "
        "raise rather than call, bet your strong hands and good draws, and fold weak hands without regret."
    ),
    "loose_aggressive": (
        "Your play style for this game: loose and aggressive. Play many hands, especially in late position, open and "
        "re-raise often, keep betting when you were the last aggressor, and bluff when your opponents look weak."
    ),
    "tight_passive": (
        "Your play style for this game: tight and passive. Play only premium hands, prefer calling to raising unless "
        "you hold a very strong hand, and give up quickly when you miss the board."
    ),
    "loose_passive": (
        "Your play style for this game: loose and passive. You like to see flops and showdowns: call often, raise "
        "rarely, and keep calling with any pair or draw rather than folding."
    ),
    "trappy": (
        "Your play style for this game: trappy. Slow-play your strongest hands by checking and calling to let "
        "opponents bet into you, check-raise when you are strong, and bet small to induce bluffs."
    ),
    "talker": (
        "Your play style for this game: a talkative bluffer. Use the optional say message on most actions to needle, "
        "misdirect or claim strength or weakness, and bluff somewhat more often than you otherwise would."
    ),
}
