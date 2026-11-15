"""Scouting reports on every arena opponent: type, how it plays, how we do against it, how to beat it.

    python3 -I lab/scouting.py      # runs lab/profiles.py + lab/vs_opponents.py logic, writes lab/data/scouting.{json,md,html}

Built from all of pineapple's arena matches (public logs). Refresh it every check-in; publish.py copies the HTML to the
dashboard (scouting.html).
"""

from __future__ import annotations

import html
import json
import subprocess
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parent
DATA = LAB / "data"


def classify(p: dict) -> tuple[str, str, str]:
    """(type, how it plays, how to beat it) from a profile."""
    v, r, f, agg, sh = p["vpip"], p["pfr"], p["fold_vs_raise"], p["post_aggression"], p["shove"]
    bluff = p.get("big_river_bluff_rate")
    sd = p.get("showdown_chen")
    if sh >= 0.05:
        return ("shover", f"goes all-in preflop in {sh:.0%} of hands, with wide hands (avg shown strength {sd})",
                "call its all-ins only with 99+, AJs+, AQo+ and only when nobody else has called; never with junk")
    if v <= 0.10:
        return ("nit", f"plays only {v:.0%} of hands; shown hands are strong (avg strength {sd})",
                "steal its blinds every time; fold anything but premiums when it bets or raises")
    if r <= 0.05 and v >= 0.35:
        return ("calling station", f"plays {v:.0%} but raises {r:.0%}, limps a lot, almost never bets after the flop",
                "value-bet big on every street with any decent hand; never bluff it; raise its limps")
    if r >= 0.25 and agg >= 2.5:
        return ("maniac / loose-aggressive", f"plays {v:.0%}, raises {r:.0%}, bets {agg:.1f}x as often as it calls"
                + (f"; {bluff:.0%} of its big river bets shown down were bluffs" if bluff is not None else ""),
                "don't fold decent hands to it: call down with pairs, 3-bet it with 99+/AJ+/KQ, let it bluff")
    if v <= 0.25 and f >= 0.85:
        return ("tight-passive", f"plays {v:.0%}, folds to {f:.0%} of raises",
                "raise and steal relentlessly; when it does bet big, believe it")
    if v >= 0.30 and r < 0.20:
        return ("loose-passive", f"plays {v:.0%} but raises only {r:.0%}, folds to {f:.0%} of raises",
                "isolate it with raises, value-bet thinner, bluff less")
    return ("tight-aggressive", f"plays {v:.0%}, raises {r:.0%}, folds to {f:.0%} of raises",
            "respect its raises with medium hands; steal when it is not in the pot")


def main() -> None:
    subprocess.run([sys.executable, "-I", str(LAB / "profiles.py")], capture_output=True)
    subprocess.run([sys.executable, "-I", str(LAB / "vs_opponents.py")], capture_output=True)
    prof = json.loads((DATA / "profiles.json").read_text())
    vs = json.loads((DATA / "vs_opponents.json").read_text())["opponents"]
    rows = []
    for name, p in prof.items():
        typ, how, beat = classify(p)
        h2h = vs.get(name, {})
        rows.append({"opponent": name, "type": typ, "how_it_plays": how, "how_to_beat": beat,
                     "hands_seen": p["hands"], "vpip": p["vpip"], "pfr": p["pfr"], "fold_vs_raise": p["fold_vs_raise"],
                     "our_net_bb": h2h.get("net_bb"), "pots": h2h.get("pots"),
                     "we_fold_to_them": h2h.get("we_fold_to_them"), "they_fold_to_us": h2h.get("they_fold_to_us")})
    rows.sort(key=lambda x: (x["our_net_bb"] if x["our_net_bb"] is not None else 0))
    stamp = time.strftime("%Y-%m-%d %H:%M")
    (DATA / "scouting.json").write_text(json.dumps({"updated": stamp, "opponents": rows}, indent=1))
    md = [f"# Scouting reports (updated {stamp})", "", "Sorted by how many chips pineapple loses to them (worst first).", ""]
    for x in rows:
        md += [f"## {x['opponent']} — {x['type']}", f"- How it plays: {x['how_it_plays']}.",
               f"- Us vs them: {x['our_net_bb']:+.1f} BB over {x['pots']} contested pots; we fold to them "
               f"{x['we_fold_to_them']}x, they fold to us {x['they_fold_to_us']}x." if x["pots"] else "- Us vs them: no contested pots yet.",
               f"- How to beat it: {x['how_to_beat']}.", ""]
    (DATA / "scouting.md").write_text("\n".join(md))
    cards = "".join(
        f"<section class=card><h2>{html.escape(x['opponent'])} <span class=tag>{html.escape(x['type'])}</span></h2>"
        f"<p><b>How it plays:</b> {html.escape(x['how_it_plays'])}.</p>"
        f"<p><b>Us vs them:</b> " + (f"<span class={'bad' if (x['our_net_bb'] or 0) < 0 else 'good'}>{x['our_net_bb']:+.1f} BB</span> over {x['pots']} pots; "
        f"we fold to them {x['we_fold_to_them']}×, they fold to us {x['they_fold_to_us']}×" if x["pots"] else "no contested pots yet") + "</p>"
        f"<p><b>How to beat it:</b> {html.escape(x['how_to_beat'])}.</p></section>" for x in rows)
    page = f"""<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width, initial-scale=1">
<title>Scouting Reports</title><style>
:root{{--bg:#0d1117;--panel:#161b22;--line:#30363d;--text:#e6edf3;--mute:#8b949e;--red:#f85149;--green:#3fb950;--gold:#e3b341}}
@media (prefers-color-scheme: light){{:root:not([data-theme="dark"]){{--bg:#f6f8fa;--panel:#fff;--line:#d0d7de;--text:#1f2328;--mute:#57606a}}}}
:root[data-theme="dark"]{{--bg:#0d1117;--panel:#161b22;--line:#30363d;--text:#e6edf3;--mute:#8b949e}}
body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}}main{{max-width:900px;margin:auto;padding:16px;display:grid;gap:12px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px 16px}}h1{{margin:8px 0}}h2{{font-size:17px;margin:0 0 6px}}p{{margin:4px 0}}
.tag{{font-size:12px;color:var(--gold);border:1px solid var(--gold);border-radius:10px;padding:1px 8px;margin-left:6px}}.bad{{color:var(--red)}}.good{{color:var(--green)}}a{{color:#58a6ff}}
</style></head><body><main><h1>Scouting reports</h1><p class=mute>Updated {stamp} (Pacific) from pineapple's arena games, worst matchups first.
<a href="index.html">← dashboard</a></p>{cards}</main></body></html>"""
    (DATA / "scouting.html").write_text(page)
    print(f"{len(rows)} opponents; worst: " + ", ".join(f"{x['opponent']} ({x['type']}, {x['our_net_bb']:+.0f})" for x in rows[:4] if x["our_net_bb"] is not None))


if __name__ == "__main__":
    main()
