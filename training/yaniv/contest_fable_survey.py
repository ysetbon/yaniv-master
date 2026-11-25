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


# ---------------------------------------------------------------- targeted probes (cheap: few alternatives)
def _find(actions, rank, count, draw):
    for a in actions:
        if a.kind == "play" and a.draw == draw and popcount(a.discard_mask) == count \
                and rank_of(cards_from_mask(a.discard_mask)[0]) == rank:
            return a
    return None


def probe_cand(view, actions, chosen, state):
    """The candidate classes of the first survey, for a fresh-seed check:
    triple   I hold exactly two of the open card's rank and the net does not take it onto them: throw the group
             worth most among the other ranks, take the open card
    refuse   the net takes an unpaired open card worth 3 or 4: same throw, deck
    """
    if chosen.kind == "yaniv":
        return []
    hand = cards_from_mask(view.my_hand)
    ranks = [rank_of(c) for c in hand]
    o = rank_of(view.discard[-1])
    op = POINTS_OF[view.discard[-1]]
    cr = rank_of(cards_from_mask(chosen.discard_mask)[0])
    cn = popcount(chosen.discard_mask)
    info = {"mc": len(hand), "oc": view.opp_size, "ply": view.ply, "op": op, "est": float(state_features(view)[94]) * 40,
            "pts": mask_points(view.my_hand), "cr": cr, "cn": cn}
    out = []
    if ranks.count(o) == 2 and not (chosen.draw == "open" and cr != o):
        others = {}
        for c in hand:
            if rank_of(c) != o:
                others[rank_of(c)] = others.get(rank_of(c), 0) + POINTS_OF[c]
        if others:
            best = max(others, key=lambda r: (others[r], r))
            a = _find(actions, best, ranks.count(best), "open")
            if a:
                out.append(("triple", a, info))
    if chosen.draw == "open" and op in (3, 4) and stock_ok(view):
        kept = [rank_of(c) for c in cards_from_mask(view.my_hand & ~chosen.discard_mask)]
        if o not in kept:
            a = _find(actions, cr, cn, "stock")
            if a:
                out.append(("refuse", a, info))
    return out


def stock_ok(view):
    return view.stock_size > 0


def probe_trained(view, actions, chosen, state):
    """The trained net's move where it differs from the hero's (a look at another player's choices; no weight is
    copied, nothing fitted)."""
    from .contest_fable_lab import player
    from .features import action_key

    alt = player("trained")(view, actions, None)
    if action_key(alt) == action_key(chosen):
        return []
    hand = cards_from_mask(view.my_hand)
    ranks = sorted(rank_of(c) for c in hand)
    unseen = [0] * 14
    for c in cards_from_mask(pool_mask(view)):
        unseen[rank_of(c)] += 1

    def mv(a):
        if a.kind == "yaniv":
            return ("yaniv", 0, "none")
        return (rank_of(cards_from_mask(a.discard_mask)[0]), popcount(a.discard_mask), a.draw)

    f = state_features(view)
    return [("trained", alt, {"hand": tuple(ranks), "open": rank_of(view.discard[-1]), "oc": view.opp_size,
                              "ply": view.ply, "unseen": tuple(unseen), "est": float(f[94]) * 40.0,
                              "assaf": float(f[95]), "ch": mv(chosen), "alt": mv(alt),
                              "known": tuple(sorted(rank_of(c) for c in cards_from_mask(view.opp_known)))})]
