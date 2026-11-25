"""Team fable, thread `survey`: where does hm5 throw or draw wrongly?  Measuring only (no weight is fitted).

`probe_all` offers EVERY distinct legal throw x draw at each non-Yaniv decision of the hero, so that any rule
stated a priori ("in situation S play move M(S)") can be scored afterwards from the same play-outs: gain =
margin(rule's move) - margin(net's move), both played to the end on the same hidden cards with the hero's
own policy afterwards.  The rules themselves are in contest_fable_survey_rules.py.

    python -m training.yaniv.contest_fable_survey collect hm5 hm5 --rounds 10000 --seed 101 --out FILE.pkl
"""

from __future__ import annotations

import argparse
import pickle
import time

from .engine import POINTS_OF, cards_from_mask, mask_points, popcount, rank_of
from .features import net_actions, pool_mask, opp_min_points, state_features


def probe_all(view, actions, chosen, state):
    if chosen.kind == "yaniv":
        return []
    unique, _ = net_actions(actions)
    hand = sorted(rank_of(c) for c in cards_from_mask(view.my_hand))
    unseen = [0] * 14
    for c in cards_from_mask(pool_mask(view)):
        unseen[rank_of(c)] += 1
    f = state_features(view)
    cm = chosen.discard_mask
    d = {"hand": tuple(hand), "open": rank_of(view.discard[-1]), "oc": view.opp_size, "ply": view.ply,
         "stock": view.stock_size, "known": tuple(sorted(rank_of(c) for c in cards_from_mask(view.opp_known))),
         "unseen": tuple(unseen), "est": float(f[94]) * 40.0, "omin": opp_min_points(view),
         "opp_true": mask_points(state.hands[1 - view.seat]),
         "opp_hand": tuple(sorted(rank_of(c) for c in cards_from_mask(state.hands[1 - view.seat]))),
         "ylegal": mask_points(view.my_hand) <= 7,
         "ch": (rank_of(cards_from_mask(cm)[0]), popcount(cm), chosen.draw),
         "id": (view.seat, view.ply, view.my_hand, len(view.discard), view.discard[-1], state.hands[1 - view.seat])}
    out = []
    for a in unique:
        if a.kind != "play":
            continue
        m = a.discard_mask
        out.append(("all", a, {"d": d, "m": (rank_of(cards_from_mask(m)[0]), popcount(m), a.draw)}))
    return out


def collect(hero, opp, rounds, seed, out, workers=2):
    from .contest_fable_lab import paired

    t0 = time.time()
    records, stats = paired(hero, opp, "contest_fable_survey:probe_all", rounds, seed, workers=workers)
    # regroup: one entry per decision = (d, margin0, {move: gain})
    dec = {}
    for r in records:
        d = r["d"]
        e = dec.setdefault(d["id"], (d, r["b0"] - r["a0"], {}))
        e[2][r["m"]] = r["gain"]
    decisions = list(dec.values())
    with open(out, "wb") as fh:
        pickle.dump({"hero": hero, "opp": opp, "seed": seed, "stats": stats, "decisions": decisions}, fh)
    print(f"{hero} vs {opp} seed {seed}: {stats['rounds']} rounds, {stats['decisions']} hero decisions, "
          f"{len(decisions)} probed, margin {stats['margin'] / stats['rounds']:+.3f}  [{time.time() - t0:.0f}s] -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd")
    p.add_argument("hero")
    p.add_argument("opp")
    p.add_argument("--rounds", type=int, default=10000)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    collect(a.hero, a.opp, a.rounds, a.seed, a.out, a.workers)
