"""Team fable, thread `yaniv`: probes for contest_fable_lab.paired. They only describe positions and name
alternative moves; nothing is fitted here.

`calls` (hero = hm5): at every hero decision where Yaniv is legal
  - the hero calls: alternatives  net_throw (the hero net's best throw row), hi_open (throw the highest-points
    group, take the open card), hi_deck (the same throw, deck draw), single_open (highest single, take open);
  - the hero plays on: alternative  call.
Every record carries the position (std_info + extras) and whether a call would be caught (true hands).

    python -m training.yaniv.contest_fable_yaniv_probe run OPP N SEED [HERO]   -> scratch pickle
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

HERO = "hm5"
SCRATCH = Path("/tmp/claude-0/-home-user-yaniv-master/ca258bc3-d453-5237-b2d9-a66c415800ab/scratchpad")


def _info(view, state):
    from .contest_fable_lab import std_info
    from .engine import POINTS_OF, cards_from_mask, rank_of
    from .features import state_features

    info = std_info(view, state)
    f = state_features(view)
    info["sure"] = bool(f[86])
    info["unseen_low"] = float(f[78])
    info["unseen_avg"] = float(f[77]) * 10
    info["caught"] = info["opp_true"] <= info["pts"]
    info["opp_hand"] = sorted(POINTS_OF[c] for c in cards_from_mask(state.hands[1 - view.seat]))
    info["open_joker"] = rank_of(view.discard[-1]) == 13
    return info


def _groups(view):
    """(points, n, mask) of every same-rank group in my hand (whole group) and every single."""
    from .engine import POINTS_OF, cards_from_mask, rank_of

    by_rank: dict = {}
    for c in cards_from_mask(view.my_hand):
        by_rank.setdefault(rank_of(c), []).append(c)
    return by_rank, POINTS_OF


def calls(view, actions, chosen, state):
    from .contest_fable_lab import describe, player  # noqa: F401
    from .engine import mask_points
    from .features import action_key, net_actions, score_actions

    call = next((a for a in actions if a.kind == "yaniv"), None)
    if call is None:
        return []
    info = _info(view, state)
    if chosen.kind != "yaniv":
        d = describe(chosen)
        info.update({"did_call": False, "thrown": d["pts"], "thrown_n": d["n"], "draw": d["draw"],
                     "after": info["pts"] - d["pts"] + (info["open"] if d["draw"] == "open" else -1)})
        return [("call", call, info)]
    info["did_call"] = True
    by_rank, pts_of = _groups(view)
    best_rank = max(by_rank, key=lambda r: (sum(pts_of[c] for c in by_rank[r]), -len(by_rank[r])))
    hi_mask = sum(1 << c for c in by_rank[best_rank])
    hi_pts = mask_points(hi_mask)
    top_single = max((c for cs in by_rank.values() for c in cs), key=lambda c: pts_of[c])
    plays = [a for a in actions if a.kind == "play"]

    def find(mask, draw):
        return next((a for a in plays if a.discard_mask == mask and a.draw == draw), None)

    out = []
    # the hero net's own best throw
    from .contest_fable_lab import load

    if "_net" not in globals():
        globals()["_net"] = load(HERO)
    uniq = net_actions(plays)[0]
    sc = score_actions(globals()["_net"], view, uniq)
    nb = uniq[int(sc.argmax())]
    d = describe(nb)
    base = {**info, "hi_pts": hi_pts, "hi_n": len(by_rank[best_rank]), "top_single": pts_of[top_single]}
    out.append(("net_throw", nb, {**base, "thrown": d["pts"], "thrown_n": d["n"], "draw": d["draw"],
                                  "after": info["pts"] - d["pts"] + (info["open"] if d["draw"] == "open" else -1),
                                  "row": float(sc.max())}))
    a = find(hi_mask, "open")
    if a is not None:
        out.append(("hi_open", a, {**base, "thrown": hi_pts, "thrown_n": len(by_rank[best_rank]), "draw": "open",
                                   "after": info["pts"] - hi_pts + info["open"],
                                   "same_as_net": action_key(a) == action_key(nb)}))
    a = find(hi_mask, "stock")
    if a is not None:
        out.append(("hi_deck", a, {**base, "thrown": hi_pts, "thrown_n": len(by_rank[best_rank]), "draw": "stock",
                                   "after": -1, "same_as_net": action_key(a) == action_key(nb)}))
    if len(by_rank[best_rank]) > 1 or (1 << top_single) != hi_mask:
        a = find(1 << top_single, "open")
        if a is not None:
            out.append(("single_open", a, {**base, "thrown": pts_of[top_single], "thrown_n": 1, "draw": "open",
                                           "after": info["pts"] - pts_of[top_single] + info["open"],
                                           "same_as_net": action_key(a) == action_key(nb)}))
    return out


if __name__ == "__main__":
    from .contest_fable_lab import paired

    opp, n, seed = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    hero = sys.argv[5] if len(sys.argv) > 5 else HERO
    recs, stats = paired(hero, opp, "contest_fable_yaniv_probe:calls", n, seed, workers=2)
    tag = hero if hero in ("hm5",) else "alt"
    path = SCRATCH / f"calls_{tag}_{opp}_{seed}.pkl"
    with open(path, "wb") as fh:
        pickle.dump((recs, stats), fh)
    print(path, len(recs), stats)
