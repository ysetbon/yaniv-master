"""Thread `reads`: a better read of P(the opponent calls Yaniv on its next turn) for the throw side.

Measured (results/contest-fable-logs/reads.md): against all of counter / hm5 / trained the chance that the
opponent calls next depends, beyond (its card count, opp_est), on

  * MY card count after the move (cards kept + the one drawn): with 2 cards it is about a third of the value with
    4-5 cards, with 1 card nearly zero, with 3 cards a little lower (opponents do not call against a short hand:
    their own chance-of-being-caught input is high);
  * a LOW open card (A, 2, joker) just thrown by the opponent: about half (it threw a low card on purpose, usually
    keeping a pair to throw next);
  * against 5 cards, an open card worth <= 4: about 4x (of a tiny base; not built).

Both built signals are written as a shift of the estimate that the PI buckets read:

    est_eff = opp_est + KSLOPE x relu(KKNEE - cards kept) + OPEN_LOW x [open card is A / 2 / joker]

The est-bucket steps move from layer 1 to layer 2 (they read the `few` unit); nothing else in contest_fable's
throw units changes. Unit count: 2 new units (few, openlow), +1 carry of opp_est, -10 carries of the steps.

    build()                                  -> (net, kit); all read changes on
    build(KSLOPE=0, OPEN_LOW=0)              -> contest_fable.build() exactly (same moves)
    build(PI={...}, ALPHA={...}, ...)        -> contest_fable.config overrides pass through
    build(FEW_ALPHA=0.4)                     -> exposure ALPHA x 0.4 when I will hold <= 2 cards (measured e2 ratio)
"""

from __future__ import annotations

from training.yaniv import contest_fable as cf
from training.yaniv.handmade_kit import Kit

KSLOPE = 3.75      # points of estimate per card under KKNEE: kept 2 (3 cards after the draw) +1.5, kept 1 +5.25,
KKNEE = 2.4        # kept 0 +9 (reads.md: the shift that maps the k-card rows of P(call) onto the 4-5-card row)
OPEN_LOW = 4.5     # an open A / 2 / joker: P(call next) about halves ~ one and a half est buckets
# P(the opponent calls next | its cards, est bucket), for MY 4-5 cards after the move, averaged over
# counter / hm5 / trained (30,000 rounds each, seed 4102; reads.md table B). Cells with n < 150 keep a neighbour's.
PI_AVG = {"5": [0.30, 0.20, 0.15, 0.10, 0.045, 0.004],
          "4": [0.40, 0.30, 0.22, 0.115, 0.05, 0.013],
          "3": [0.50, 0.36, 0.22, 0.10, 0.035, 0.006],
          "12": [0.62, 0.39, 0.26, 0.03, 0.0, 0.0]}


def add_read_units(kit: Kit, kslope: float, kknee: float, open_low: float) -> None:
    """Pre-create contest_fable's est-bucket steps (`est_under{t}_a/_b`) so that they read est_eff."""
    w = {"opp_est": -400.0}
    layer = 1
    if kslope:
        kit.unit(1, "few", {"n_kept": -5.0}, bias=kknee, note=f"relu({kknee:g} - cards kept): my hand will be short")
        w["few"] = -10.0 * kslope
        layer = 2
    if open_low:
        kit.unit(1, "openlow", {"open_A": 1.0, "open_2": 1.0, "open_JOKER": 1.0},
                 note="the open card (the opponent's last throw) is an A, 2 or joker")
        w["openlow"] = -10.0 * open_low
        layer = 2
    if layer == 1:
        return                                          # contest_fable makes them itself
    for t in cf.EST_CUTS:
        note = f"effective estimate of the opponent's total < {t:g}"
        kit.unit(layer, f"est_under{t:g}_a", dict(w), bias=10.0 * t, note=note + " (ramp)")
        kit.unit(layer, f"est_under{t:g}_b", dict(w), bias=10.0 * t - 1.0, note=note + " (ramp shifted)")


def add_few_alpha(kit: Kit, cfg: dict, factor: float) -> None:
    """ALPHA[c] x factor when I will hold <= 2 cards (kept <= 1): give back (1 - factor) ALPHA[c] of the heavy
    points' exposure and of the plan's ALPHA part."""
    kit.unit(1, "nkept_le1_a", {"n_kept": -50.0}, bias=16.0, note="[cards kept <= 1] (ramp)")
    kit.unit(1, "nkept_le1_b", {"n_kept": -50.0}, bias=15.0, note="[cards kept <= 1] (ramp shifted)")
    heavy_pts = {"pts_kept": 40.0, "incoming": 10.0, "hand_A": -4.0, "aces_thrown": 1.0, "took_ace_open": -1.0,
                 "hand_2": -8.0, "twos_thrown": 2.0, "took_two_open": -2.0}
    plan_layer = kit.level["plan_c5"]
    gt = {s: x for s, x in kit.layers[plan_layer - 1]["plan_c5"][0].items() if not s.startswith("opp_has_")}
    out = {}
    for c in cf.REGIMES:
        give = (1.0 - factor) * cfg["ALPHA"][c]
        gates = {g: 60.0 for g in cf.GATE_OF[c]}
        kit.unit(2, f"heavypts_few_c{c}", {**heavy_pts, **gates, "nkept_le1_a": 60.0, "nkept_le1_b": -60.0}, bias=-120.0,
                 note=f"heavy points after the move, regime {c}, I will hold <= 2 cards")
        out[f"heavypts_few_c{c}"] = give
        kit.unit(plan_layer, f"plan_few_c{c}", {**gt, **gates, "nkept_le1_a": 60.0, "nkept_le1_b": -60.0}, bias=-120.0,
                 note=f"G (+ deck gain), regime {c}, I will hold <= 2 cards")
        out[f"plan_few_c{c}"] = -give
    kit.out(out)


def _drop_unused(kit: Kit, prefixes: tuple) -> None:
    """Remove units (names starting with a prefix) that no unit and no output term reads any more."""
    while True:
        used = set(kit.output[0])
        for layer in kit.layers:
            for weights, _, _ in layer.values():
                used.update(weights)
        dead = [(k, n) for k, layer in enumerate(kit.layers) for n in layer if n.startswith(prefixes) and n not in used]
        if not dead:
            return
        for k, n in dead:
            del kit.layers[k][n]
            del kit.level[n]


def fix_catch(kit: Kit, mode: str, k3: float = 0.0) -> None:
    """The catch bonus (CATCH x PI x P(my hand after the draw <= CATCH_M)) pays only when the opponent calls
    AND I am at or under its total. Measured (reads.md table E): with my hand <= 5 after the move the net assumes
    PI ~ 0.4 against 1-2 cards, but P(it calls and I catch) is 0.00 after taking the open card (it knows the card
    and calls only when sure), 0.01-0.04 after a deck draw to 2 cards, 0.08-0.18 to 3 cards, 0.24-0.30 to 4.
      mode "off":  no catch bonus at all (the units are removed)
      mode "k4":   only when I keep >= 3 cards (4-5 after the draw)
      k3 > 0:      (with "k4") also x k3 for a DECK draw to exactly 3 cards
    """
    catch = [n for layer in kit.layers for n in layer if n.startswith("catch_c")]
    out = kit.output[0]
    if mode == "off":
        for n in catch:
            out.pop(n, None)
        _drop_unused(kit, ("catch_c", "low_", "kept_le", "held_le"))
        return
    kit.unit(1, "nk3_a", {"n_kept": 50.0}, bias=-24.0, note="[cards kept >= 3] (ramp)")
    kit.unit(1, "nk3_b", {"n_kept": 50.0}, bias=-25.0, note="[cards kept >= 3] (ramp shifted)")
    if k3:
        kit.unit(1, "nk2_a", {"n_kept": 50.0}, bias=-14.0, note="[cards kept >= 2] (ramp)")
        kit.unit(1, "nk2_b", {"n_kept": 50.0}, bias=-15.0, note="[cards kept >= 2] (ramp shifted)")
    layer = kit.layers[kit.level[catch[0]] - 1]
    for n in catch:
        weights, bias, note = layer[n]
        if k3:
            deck = {s: (k3 * x if s.startswith("low_k") else x) for s, x in weights.items() if s != "low_open"}
            kit.unit(kit.level[n], n.replace("catch_c", "catch3_c"),
                     {**deck, "nk2_a": 1.0, "nk2_b": -1.0, "nk3_a": -1.0, "nk3_b": 1.0}, bias=bias - 1.0,
                     note=note + f", deck draw to exactly 3 cards (x {k3:g})")
            out[n.replace("catch_c", "catch3_c")] = out[n]
        layer[n] = ({**weights, "nk3_a": 1.0, "nk3_b": -1.0}, bias - 1.0, note + ", and I keep >= 3 cards")


def build(KSLOPE: float = KSLOPE, KKNEE: float = KKNEE, OPEN_LOW: float = OPEN_LOW, FEW_ALPHA: float = 1.0,
          PI_TABLE: str = "", CATCH_MODE: str = "", CATCH_K3: float = 0.0, **over):
    if PI_TABLE == "avg":
        over = {"PI": {c: list(v) for c, v in PI_AVG.items()}, **over}
    cfg = cf.config(**over)
    kit = Kit()
    add_read_units(kit, KSLOPE, KKNEE, OPEN_LOW)
    cf.add_throw_units(kit, cfg)
    if FEW_ALPHA != 1.0:
        add_few_alpha(kit, cfg, FEW_ALPHA)
    if CATCH_MODE:
        fix_catch(kit, CATCH_MODE, CATCH_K3)
    reasons = cf.add_yaniv_units(kit)
    reasons += cf.add_score_units(kit)
    cf.finish_yaniv(kit, reasons)
    cf.fold_output(kit)
    net = kit.build()
    hidden = sum(net.dims[1:-1])
    if hidden > cf.MAX_HIDDEN:
        raise ValueError(f"{hidden} hidden units: the contest allows {cf.MAX_HIDDEN}")
    return net, kit


# ---------------------------------------------------------------- paired probe: hm5's move where it differs
_HM5 = []


def probe_hm5(view, actions, chosen, state):
    """Alternative = the move hm5 would make here, when it differs from the hero's. Grouped by the opponent's
    regime, my cards after each move and the open card."""
    from .contest_fable_lab import describe, player, std_info
    from .engine import popcount
    from .features import action_key

    if not _HM5:
        _HM5.append(player("hm5"))
    alt = _HM5[0](view, actions, None)
    if action_key(alt) == action_key(chosen) or chosen.kind == "yaniv" or alt.kind == "yaniv":
        return []
    info = std_info(view, state)
    c, a = describe(chosen), describe(alt)
    mc = info["mc"]
    info.update(k_hero=mc - c["n"] + 1, k_alt=mc - a["n"] + 1, hero_move=c, alt_move=a,
                reg={5: "5", 4: "4", 3: "3"}.get(info["oc"], "12"), open_low=int(info["open"] <= 2 and view.ply > 0))
    return [("hm5", alt, info)]


_NET = {}


def probe_second(view, actions, chosen, state):
    """Alternative = the hero net's own second choice (hero must be hm5 here), with the score gap and the read
    signals, to find classes of throws where the valuation is wrong."""
    import numpy as np

    from .contest_fable_lab import describe, load, std_info
    from .features import action_key, net_actions, score_actions

    if chosen.kind == "yaniv":
        return []
    if "hm5" not in _NET:
        _NET["hm5"] = load("hm5")
    unique = [a for a in net_actions(actions)[0] if a.kind != "yaniv"]
    if len(unique) < 2:
        return []
    scores = score_actions(_NET["hm5"], view, unique)
    order = np.argsort(-scores)
    if action_key(unique[order[0]]) != action_key(chosen):
        return []
    alt = unique[order[1]]
    info = std_info(view, state)
    c, a = describe(chosen), describe(alt)
    mc = info["mc"]
    info.update(k_hero=mc - c["n"] + 1, k_alt=mc - a["n"] + 1, hero_move=c, alt_move=a,
                gap=float(scores[order[0]] - scores[order[1]]),
                reg={5: "5", 4: "4", 3: "3"}.get(info["oc"], "12"), open_low=int(info["open"] <= 2 and view.ply > 0))
    return [("second", alt, info)]
