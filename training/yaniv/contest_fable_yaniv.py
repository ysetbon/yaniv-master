"""Team fable, thread `yaniv`: the contest net with its own Yaniv units (everything else is contest_fable's).

    py:contest_fable_yaniv.build            the rule below
    py:contest_fable_yaniv.build@{...}      with other constants (keyword arguments of `add_yaniv_units`)

The rule (measured in results/contest-fable-logs/yaniv.md). Call Yaniv iff
  1. the call cannot be caught (sure_yaniv), or
  2. assaf < LOW (the uniform catch chance is under half a percent), or
  3. no lower hand is one swap away  AND  assaf < A[opp cards] x (BASE + PTS_W x my points - PLY_W x clip(ply)).
     "one swap away": my highest group is worth SWAP or more points above the open card (inputs best_shed and
     open_pts): then throwing it and taking the open card beats a call that can be caught.
The threshold on `assaf` rises with the opponent's card count falling (assaf understates the true catch rate by
x20 against 4 cards, x6 against 3, x2.5 against 2, x1 against 1), with my points (at the same assaf a 7-point
call is caught no more often than a 4-point one but playing on from 7 is worth less) and falls with the ply
(hidden cards are lower late in the round).

Nothing is trained or searched: the constants are read off the paired tables and rounded.
"""

from __future__ import annotations

from training.yaniv import contest_fable as cf
from training.yaniv.handmade_kit import Kit

A = {"1": 0.04, "2": 0.021, "3": 0.0055, "4": 0.0021}   # threshold per "point" for 1 / 2 / 3 / 4 opponent cards
LOW = 0.005          # always call under this assaf (against 5 cards this is the only risky call made)
BASE = 5.5           # threshold "points" = BASE + PTS_W x my points - PLY_W x clip(ply, PLY_LO, PLY_HI)
PTS_W = 0.5          # (= 5 + 0.5 (pts - 5) + 0.2 (15 - ply): 5 at 5 points, ply 15)
PLY_W = 0.2
PLY_LO, PLY_HI = 8.0, 20.0
SWAP = 2.0           # veto reason 3 when best_shed - open card >= SWAP points
K = 2000.0           # ramp steepness on assaf: a reason goes 0 -> 1 over 1/K = 0.0005 of assaf, centred on T
G = 4000.0           # gate: larger than K x any threshold


def add_yaniv_units(kit: Kit, a=None, low=LOW, base=BASE, pts_w=PTS_W, ply_w=PLY_W, ply_lo=PLY_LO, ply_hi=PLY_HI,
                    swap=SWAP, swap_on=True) -> list[dict]:
    a = {**A, **(a or {})}
    reasons = []
    kit.unit(1, "sure_call", {"sure_yaniv": 1.0, "yaniv": 1.0}, bias=-1.0, note="the call cannot be caught")
    reasons.append({"sure_call": 1.0})

    w = {"assaf": -K, "yaniv": G}
    kit.unit(1, "ylow_a", w, bias=K * low + 0.5 - G, note="Yaniv row: K (LOW - assaf) + 0.5")
    kit.unit(1, "ylow_b", dict(w), bias=K * low - 0.5 - G, note="... - 0.5: ylow_a - ylow_b = [assaf < LOW]")
    reasons.append({"ylow_a": 1.0, "ylow_b": -1.0})

    kit.unit(1, "yply_lo", {"ply": 48.0}, bias=-ply_lo, note="relu(ply - PLY_LO)")
    kit.unit(1, "yply_hi", {"ply": 48.0}, bias=-ply_hi, note="relu(ply - PLY_HI): clip(ply) = LO + lo - hi")
    veto = {}
    if swap_on:
        d = {"best_shed": 40.0, "open_pts": -10.0}
        kit.unit(1, "yswap_a", d, bias=-(swap - 1.0), note="relu(best_shed - open - (SWAP - 1))")
        kit.unit(1, "yswap_b", dict(d), bias=-swap, note="yswap_a - yswap_b = [my highest group >= open card + SWAP]")
        veto = {"yswap_a": -G, "yswap_b": G}
    for c in ("1", "2", "3", "4"):
        t = K * a[c]
        m = {"assaf": -K, "my_pts": t * pts_w * 40.0, "yply_lo": -t * ply_w, "yply_hi": t * ply_w,
             "yaniv": G, f"opp_has_{c}": G, **veto}
        b = t * (base - ply_w * ply_lo) - 2.0 * G
        kit.unit(2, f"ycall{c}_a", m, bias=b + 0.5, note=f"Yaniv row, opponent holds {c}, no swap: K (T - assaf) + 0.5")
        kit.unit(2, f"ycall{c}_b", dict(m), bias=b - 0.5, note="... - 0.5: the difference is [assaf < T]")
        reasons.append({f"ycall{c}_a": 1.0, f"ycall{c}_b": -1.0})
    return reasons


def build(throw=None, **kw):
    """contest_fable's net with this module's Yaniv units. `throw`: overrides for contest_fable.config."""
    cfg = cf.config(**(throw or {}))
    kit = Kit()
    cf.add_throw_units(kit, cfg)
    reasons = add_yaniv_units(kit, **kw)
    reasons += cf.add_score_units(kit)
    cf.finish_yaniv(kit, reasons)
    cf.fold_output(kit)
    net = kit.build()
    hidden = sum(net.dims[1:-1])
    if hidden > cf.MAX_HIDDEN:
        raise ValueError(f"{hidden} hidden units: the contest allows {cf.MAX_HIDDEN}")
    return net, kit


def rule(view, a=None, low=LOW, base=BASE, pts_w=PTS_W, ply_w=PLY_W, ply_lo=PLY_LO, ply_hi=PLY_HI, swap=SWAP,
         swap_on=True):
    """The intended decision, computed straight from the view (for checking the net): True = call."""
    from .engine import POINTS_OF, mask_points, max_shed_points
    from .features import state_features

    a = {**A, **(a or {})}
    f = state_features(view)
    assaf, pts = float(f[95]), mask_points(view.my_hand)
    if f[86] or assaf < low:
        return True
    if swap_on and max_shed_points(view.my_hand) - POINTS_OF[view.discard[-1]] >= swap:
        return False
    c = str(min(5, max(1, view.opp_size)))
    if c == "5":
        return False
    return assaf < a[c] * (base + pts_w * pts - ply_w * min(ply_hi, max(ply_lo, view.ply)))


def check(rounds: int = 2000, seed: int = 11, opp: str = "hm5", **kw) -> None:
    """Play rounds with the built net; at every Yaniv-legal decision compare its move with `rule` (Kit.trace
    for the reasons of the mismatches)."""
    import random

    import numpy as np

    from .contest_fable_lab import player
    from .engine import apply_action, legal_actions, make_view, new_round
    from .features import action_matrix, net_actions, score_actions, state_features

    net, kit = build(**kw)
    rng = random.Random(seed)
    o = player(opp)
    n = calls = bad = swaps = swap_taken = near = 0
    shown = 0
    for i in range(rounds):
        st = new_round([0, 0], (i // 2) % 2, rng)
        hs = i % 2
        while not st.terminal and st.ply < 48:
            v = make_view(st, st.turn)
            acts = legal_actions(v)
            if not acts:
                break
            if st.turn != hs:
                apply_action(st, o(v, acts, None))
                continue
            uniq = net_actions(acts)[0]
            sc = score_actions(net, v, uniq)
            a = uniq[int(np.argmax(sc))]
            if uniq[-1].kind == "yaniv":
                n += 1
                want = rule(v, **kw)
                got = a.kind == "yaniv"
                calls += got
                f = state_features(v)
                if not got and not want and not f[86] and float(f[95]) >= LOW:
                    from .engine import POINTS_OF, mask_points, max_shed_points
                    if max_shed_points(v.my_hand) - POINTS_OF[v.discard[-1]] >= SWAP:
                        swaps += 1
                        swap_taken += (a.draw == "open" and mask_points(a.discard_mask) == max_shed_points(v.my_hand))
                if want != got:
                    x = np.concatenate([f, action_matrix(v, [uniq[-1]])[0]])
                    tr = kit.trace(x)
                    r = sum(tr[k] for k in ("ycall1_a", "ycall2_a", "ycall3_a", "ycall4_a", "ylow_a", "sure_call")) - \
                        sum(tr[k] for k in ("ycall1_b", "ycall2_b", "ycall3_b", "ycall4_b", "ylow_b"))
                    if 0.0 < r < 1.0:
                        near += 1            # inside the 0.0005-wide ramp: either move is "intended"
                    else:
                        bad += 1
                        if shown < 5:
                            shown += 1
                            print("  MISMATCH want", want, "got", got, "assaf", float(f[95]), "pts", f[70] * 40,
                                  "oc", v.opp_size, "ply", v.ply, "reasons", r, "row", float(sc[-1]))
            apply_action(st, a)
    print(f"check vs {opp}: {rounds} rounds, {n} Yaniv-legal decisions, {calls} calls, mismatches {bad} "
          f"(+{near} inside the ramp); swap-veto positions {swaps}, of them the net throws its highest group and "
          f"takes the open card {swap_taken}")


if __name__ == "__main__":
    net, kit = build()
    print(f"layers {net.dims}, {sum(net.dims[1:-1])} hidden units (contest_fable: "
          f"{sum(cf.build()[0].dims[1:-1])})")
    check()
