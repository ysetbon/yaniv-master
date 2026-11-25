"""Design `counts`: a hand-written net for the real game (5 cards, Yaniv at 7), organised around the two card counts.

    python -m training.yaniv.handmade5_designs.counts [OUT.json]

It is meant to play while either player still holds 5 cards (the 4-card net, training/yaniv/handmade.py, takes
over once both hold 4 or fewer), but it scores every legal move in every position.

THE IDEA. What a move is worth depends on how long the round will still last, and that is read off the
opponent's card count (regime c = 5, 4, 3, or "12" for 1-2 cards; gates are the `opp_has_k` inputs):

  value of a throw =
    - light points held after the move (aces, 2s: cards I keep to the end, always exposed)
    - ALPHA[c] x heavy points held after the move. A heavy card is thrown within a turn or two, so it is only
      exposed if the round ends before that: ALPHA[c] ~ P(the round ends, by the opponent, before my second
      next turn), which is small against 5 cards and ~1 against 1-2 cards
    + (ALPHA[c] - w) x G: my heaviest rank group G leaves on my very next turn, so it is exposed only with
      w = P(the opponent calls on its next turn | its card count, its estimated total) + W_PAD
    - CARD[c] per heavy card left (a card slot costs a turn to clear; more turns remain against 5 cards)
    + CATCH x P(it calls next | c, est) x P(my hand after the draw <= CATCH_M): its call would be an Assaf
    + CALLABLE[c] x P(my hand after the draw is <= 7): I can call next turn. Against a 5-card opponent the
      call is nearly never caught and it pays its whole 5-card hand, so reaching 7 first is the plan
    + the 4-card net's small terms (open-card pairing, open tell, low top, feeding).

    - LIVE[c] per unseen copy of the rank I throw: the opponent may hold one, take my card and throw the pair
      (a card of the race), and I give up my own chance to pair it. With heavy points nearly unexposed against
      4-5 cards, WHICH heavy single goes first is decided by this, not by its points.
With ALPHA = 1, one P-table for every c, CARD = 5 (3 for "12"), CALLABLE = 0 and LIVE = 0 this is the 4-card net
(to 0.3 points: its catch bonus leaves out the 0.01 base chance).
The Yaniv rule and the match-score units are the 4-card net's, imported unchanged.

Nothing is trained or searched: the tables below are rounded real-game measurements (scratch diag.py, the
4-card net against the card counter; results/handmade-5x7-logs/counts.md) or reasoned numbers, each with a note.
"""

from __future__ import annotations

import sys
from pathlib import Path

from training.yaniv.handmade import (
    CATCH, DECK_HEAVY, FEED, LOW_TOP, N_TYP, OPEN_TELL, PAIR_OPEN, TWO_LIGHT,
    _est_under, _max_all, _step, add_score_units, add_yaniv_units, finish_yaniv, fold_output,
)
from training.yaniv.handmade_kit import RANK_NAMES as RANKS, RANK_POINTS, Kit
from training.yaniv.network import save_net

SCRATCH = Path("/tmp/claude-0/-home-user-yaniv-master/a532e2e0-83d9-59dd-8a1a-414723f6a44f/scratchpad/counts")

REGIMES = ("5", "4", "3", "12")            # the opponent's card count; "12" = 1 or 2 cards
GATE_OF = {"5": {"opp_has_5": 1.0}, "4": {"opp_has_4": 1.0}, "3": {"opp_has_3": 1.0},
           "12": {"opp_has_1": 1.0, "opp_has_2": 1.0}}

# ---------------------------------------------------------------- constants (see counts.md for the tables)
EST_CUTS = [5, 8, 11, 14, 17]              # cuts on the opponent's estimated total (the 4-card net's)
# P(the counter calls on its next turn | its card count, its estimated total): one row per regime, one entry per
# est bucket (<5, 5-8, 8-11, 11-14, 14-17, 17+). Measured (5,000 real-game rounds, 4-card net against the counter,
# diag.py; table in counts.md) and rounded. Cells with under ~100 samples are reasoned: a 5-card hand with a low
# estimate is rare, and it must be genuinely low to be estimated low, so its row rises there. The 4-card
# net's single row (0.55/0.30/0.18/0.08/0.04/0.01) is close to all of these: opp_est already carries most of it.
PI_4CARD = [0.55, 0.30, 0.18, 0.08, 0.04, 0.01]
PI = {"5": [0.30, 0.15, 0.05, 0.03, 0.02, 0.005],
      "4": [0.40, 0.25, 0.15, 0.08, 0.05, 0.015],
      "3": [0.45, 0.27, 0.18, 0.11, 0.06, 0.01],
      "12": [0.50, 0.35, 0.27, 0.15, 0.08, 0.02]}
W_PAD = 0.05                               # the 4-card net's pad on w (the round can also end other ways)
# Exposure of heavy points that are not in G ~ P(the counter ends the round before my second next turn), measured
# 0.01-0.05 against 5 cards, 0.07-0.26 against 4, 0.23-0.38 against 3, 0.55-0.9 against 1-2; set a step above
# the measurement because the third-heaviest card stays a turn longer still.
ALPHA = {"5": 0.15, "4": 0.35, "3": 0.6, "12": 0.85}
# Charge per heavy card left: the 4-card net's 5, and 3 against 1-2 cards. Raising it to 7 / 6 against 5 / 4 cards
# changed nothing (variant G), so it stays.
CARD = {"5": 5.0, "4": 5.0, "3": 5.0, "12": 3.0}
# Charge per 2 left: half a heavy card, as in the 4-card net. Making 2s nearly free against 5 / 4 cards (variant E)
# was not better.
TWO = {c: TWO_LIGHT * CARD[c] for c in REGIMES}
# x P(my hand after the draw <= 7). 8 / 5 / 2 against 5 / 4 / 3 cards changed nothing (variant C): off.
CALLABLE = {"5": 0.0, "4": 0.0, "3": 0.0, "12": 0.0}
# Charge per unseen copy of the rank I throw. The counter takes my high card 87% of the time when it holds that
# rank (4% when it does not) and throws the pair next turn: one card of the race, worth about CARD. P(it holds the
# rank) ~ unseen copies x 5 / 40, so ~0.6 a copy, plus my own lost chance to pair that rank: ~1 a copy. Halved
# against 3 cards and dropped against 1-2 (the round is nearly over). 2 a copy (variant H) was not better.
LIVE = {"5": 1.0, "4": 1.0, "3": 0.5, "12": 0.0}
CALLABLE_SURE_VS4 = 3.0                    # the 4-card net's +3 for a SURE callable-next while the opponent has 4
CATCH_M = 5                                # "low enough to catch its call" (the 4-card net's)
CALL_AT = 7                                # the Yaniv limit of the real game


def _gated(kit: Kit, layer: int, name: str, expr: dict, c: str, big: float, note: str) -> str:
    """relu(expr + big x [regime c] - big): expr (0 <= expr <= big) when the opponent's card count is in regime c."""
    return kit.unit(layer, name, {**expr, **{g: big for g in GATE_OF[c]}}, bias=-big, note=note)


def _p_hand_le(kit: Kit, m: int, tag: str) -> dict:
    """P(my hand after the draw <= m), as a sum of layer-2 units: exact for an open card; for a deck draw
    sum_k [kept <= k] x P(card worth exactly m - k), P(card worth v) ~ unseen cards worth v / N_TYP."""
    p = {}
    for k in range(m + 1):
        le = _step(kit, f"kept_le{k}", {"pts_kept": -40.0}, float(k), f"kept points <= {k}")
        pv = {f"unseen_{r}": 4.0 / N_TYP for r, pt in zip(RANKS, RANK_POINTS) if pt == m - k}
        kit.unit(2, f"{tag}_k{k}", {**pv, **le, "draw_deck": 1.0}, bias=-2.0,
                 note=f"deck draw, kept <= {k}: chance the card is worth exactly {m - k}")
        p[f"{tag}_k{k}"] = 1.0
    held = _step(kit, f"held_le{m}", {"pts_kept": -40.0, "incoming": -10.0}, float(m),
                 f"kept points + incoming card <= {m}")
    kit.unit(2, f"{tag}_open", {**held, "take_open": 1.0}, bias=-1.0, note=f"open card taken, holding <= {m}")
    p[f"{tag}_open"] = 1.0
    return p


def add_throw_units(kit: Kit, cfg: dict) -> None:
    alpha, card, pi, callable_ = cfg["ALPHA"], cfg["CARD"], cfg["PI"], cfg["CALLABLE"]

    # ---- cards after the move: heavy-card count and light points (layer 1 pieces, linear combinations) ----
    kit.unit(1, "light_thrown", {"n_thrown": 4.0, "throw_A": 4.0, "throw_JOKER": 4.0}, bias=-4.0,
             note="cards thrown if the thrown rank is A or joker (light cards leaving), else 0")
    kit.unit(1, "aces_thrown", {"n_thrown": 4.0, "throw_A": 4.0}, bias=-4.0, note="cards thrown if they are aces")
    kit.unit(1, "twos_thrown", {"n_thrown": 4.0, "throw_2": 4.0}, bias=-4.0, note="cards thrown if they are 2s")
    kit.unit(1, "took_light_open", {"take_open": 1.0, "open_A": 1.0, "open_JOKER": 1.0}, bias=-1.0,
             note="1 if the move takes an open A or joker")
    kit.unit(1, "took_ace_open", {"take_open": 1.0, "open_A": 1.0}, bias=-1.0, note="1 if the move takes an open A")
    kit.unit(1, "took_two_open", {"take_open": 1.0, "open_2": 1.0}, bias=-1.0, note="1 if the move takes an open 2")
    kit.unit(1, "deck_heavy", {"draw_deck": 2.0 + DECK_HEAVY, "unseen_low": -1.0}, bias=-2.0,
             note="deck draw: chance the drawn card is heavy")
    heavy = {"n_kept": 5.0, "hand_A": -4.0, "hand_JOKER": -4.0, "light_thrown": 1.0,      # heavy cards after
             "hand_2": -4.0 * TWO_LIGHT, "twos_thrown": TWO_LIGHT,                         # the move (a 2 = half)
             "take_open": 1.0, "took_light_open": -1.0, "took_two_open": -TWO_LIGHT,
             "deck_heavy": 1.0}
    # heavy points after the move = held - aces - 2 x twos (a deck card counts as heavy: its average is ~5-6)
    heavy_pts = {"pts_kept": 40.0, "incoming": 10.0, "hand_A": -4.0, "aces_thrown": 1.0, "took_ace_open": -1.0,
                 "hand_2": -8.0, "twos_thrown": 2.0, "took_two_open": -2.0}

    twos = {"hand_2": 4.0, "twos_thrown": -1.0, "took_two_open": 1.0}       # 2s after the move
    out = {"pts_kept": -40.0, "incoming": -10.0}
    card_base = max(card.values())
    for s, w in heavy.items():
        out[s] = out.get(s, 0.0) - card_base * w
    for c in REGIMES:
        if card[c] != card_base:
            u = _gated(kit, 2, f"heavy_c{c}", heavy, c, 6.0, f"heavy cards after the move, opponent regime {c}")
            out[u] = card_base - card[c]
        if cfg["TWO"][c] != TWO_LIGHT * card[c]:
            u = _gated(kit, 2, f"twos_c{c}", twos, c, 6.0, f"2s after the move, opponent regime {c}")
            out[u] = TWO_LIGHT * card[c] - cfg["TWO"][c]
        if alpha[c] != 1.0:
            u = _gated(kit, 2, f"heavypts_c{c}", heavy_pts, c, 60.0,
                       f"heavy points after the move, opponent regime {c} (exposed only with ALPHA)")
            out[u] = 1.0 - alpha[c]

    # ---- open-card habits, top card (the 4-card net's) ----
    kit.unit(1, "open_tell", {"open_pts": 10.0, "take_open": 10.0, "pairs_open": -10.0}, bias=-13.0,
             note="points above 3 of an open card taken that pairs nothing")
    kit.unit(1, "callable_vs4", {"callable_next": 1.0, "opp_has_4": 1.0}, bias=-1.0,
             note="Yaniv surely legal next turn while the opponent still has 4 cards")
    out.update({"pairs_open": PAIR_OPEN, "open_tell": -OPEN_TELL, "callable_vs4": cfg["CALLABLE_SURE_VS4"],
                "low_top": -LOW_TOP})
    for r, p in zip(RANKS, RANK_POINTS):
        kit.unit(1, f"feed_{r}", {f"throw_{r}": 1.0, f"known_{r}": 4.0}, bias=-1.0,
                 note=f"leaves a {r} face up while the opponent is known to hold one (it takes a set)")
        out[f"feed_{r}"] = -(FEED[0] + FEED[1] * p)
    kit.out(out)

    # ---- live ranks: unseen copies of the rank I throw (the opponent may hold one and take my card for a set,
    #      and I give up my own chance to pair it) ----
    if any(cfg["LIVE"][c] for c in REGIMES):
        live = {}
        for r in RANKS:
            kit.unit(1, f"live_{r}", {f"unseen_{r}": 4.0, f"throw_{r}": 10.0}, bias=-10.0,
                     note=f"unseen copies of rank {r} if {r} is the rank thrown")
            live[f"live_{r}"] = 1.0
        out = {}
        for c in REGIMES:
            if cfg["LIVE"][c]:
                u = _gated(kit, 2, f"live_c{c}", live, c, 6.0, f"unseen copies of the thrown rank, regime {c}")
                out[u] = -cfg["LIVE"][c]
        kit.out(out)

    # ---- the plan: (ALPHA[c] - w(c, est)) x G ----
    groups = []
    for r, p in zip(RANKS, RANK_POINTS):
        if p == 0:
            continue
        kit.unit(1, f"grp_keep_{r}", {f"hand_{r}": 4.0 * p, f"throw_{r}": -40.0},
                 note=f"points of my {r}s if none are thrown")
        kit.unit(1, f"grp_left_{r}", {f"hand_{r}": 4.0 * p, "n_thrown": -4.0 * p, f"throw_{r}": 40.0}, bias=-40.0,
                 note=f"points of my {r}s left if {r}s are thrown")
        kit.unit(1, f"grp_open_{r}", {f"open_{r}": p, "take_open": p}, bias=-p,
                 note=f"{p} if the move takes an open {r}")
        groups.append({f"grp_keep_{r}": 1.0, f"grp_left_{r}": 1.0, f"grp_open_{r}": 1.0})
    G, L = _max_all(kit, 2, "gmax", groups)
    gt = dict(G)                                  # G + E[relu(drawn card - G)] for a deck draw
    for t in range(1, 11):
        kit.unit(L + 1, f"g_under{t}_a", {s: -w for s, w in G.items()}, bias=float(t), note=f"relu({t} - G)")
        kit.unit(L + 1, f"g_under{t}_b", {s: -w for s, w in G.items()}, bias=float(t - 1), note=f"relu({t - 1} - G)")
        pge = {f"unseen_{r}": 4.0 / N_TYP for r, p in zip(RANKS, RANK_POINTS) if p >= t}
        kit.unit(L + 2, f"deck_over{t}", {**pge, f"g_under{t}_a": 1.0, f"g_under{t}_b": -1.0, "draw_deck": 1.0},
                 bias=-2.0, note=f"deck draw and G < {t}: chance the drawn card is worth {t}+")
        gt[f"deck_over{t}"] = 1.0
    out = {}
    for c in REGIMES:
        w = [p + cfg["W_PAD"] for p in pi[c]]
        u = _gated(kit, L + 3, f"plan_c{c}", gt, c, 50.0, f"G (+ deck gain), opponent regime {c}")
        out[u] = alpha[c] - w[-1]
        for i, t in enumerate(EST_CUTS):
            if w[i] == w[i + 1]:
                continue
            e = _est_under(kit, t)
            kit.unit(L + 3, f"plan_c{c}_est{t}",
                     {**gt, **{s: 50.0 * x for s, x in e.items()}, **{g: 50.0 for g in GATE_OF[c]}}, bias=-100.0,
                     note=f"G (+ deck gain), opponent regime {c} and its estimate under {t}")
            out[f"plan_c{c}_est{t}"] = -(w[i] - w[i + 1])
    kit.out(out)

    # ---- catch bonus: CATCH x P(it calls next | c, est) x P(my hand after the draw <= CATCH_M) ----
    p_low = _p_hand_le(kit, cfg["CATCH_M"], "low")
    out = {}
    for c in REGIMES:
        if pi[c][-1] > 0:
            u = _gated(kit, 3, f"catch_c{c}", p_low, c, 1.0, f"P(my hand <= {cfg['CATCH_M']}), opponent regime {c}")
            out[u] = CATCH * pi[c][-1]
        for i, t in enumerate(EST_CUTS):
            if pi[c][i] == pi[c][i + 1]:
                continue
            e = _est_under(kit, t)
            kit.unit(3, f"catch_c{c}_est{t}", {**p_low, **e, **GATE_OF[c]}, bias=-2.0,
                     note=f"P(my hand after the draw <= {cfg['CATCH_M']}), regime {c}, its estimate under {t}")
            out[f"catch_c{c}_est{t}"] = CATCH * (pi[c][i] - pi[c][i + 1])
    kit.out(out)

    # ---- callable next turn: CALLABLE[c] x P(my hand after the draw <= 7) ----
    if any(callable_[c] for c in REGIMES):
        p_call = _p_hand_le(kit, CALL_AT, "cn")
        out = {}
        for c in REGIMES:
            if callable_[c]:
                u = _gated(kit, 3, f"cancall_c{c}", p_call, c, 1.0,
                           f"P(I can call Yaniv next turn), opponent regime {c}")
                out[u] = callable_[c]
        kit.out(out)


def config(**over) -> dict:
    cfg = {"ALPHA": dict(ALPHA), "CARD": dict(CARD), "PI": {c: list(v) for c, v in PI.items()},
           "CALLABLE": dict(CALLABLE), "TWO": None, "LIVE": dict(LIVE), "W_PAD": W_PAD, "CATCH_M": CATCH_M, "CALLABLE_SURE_VS4": CALLABLE_SURE_VS4}
    for k, v in over.items():
        if k not in cfg:
            raise KeyError(k)
        if isinstance(cfg[k], dict) and k != "TWO":
            cfg[k].update(v)
        else:
            cfg[k] = v
    two = {c: TWO_LIGHT * cfg["CARD"][c] for c in REGIMES}
    two.update(cfg["TWO"] or {})
    cfg["TWO"] = two
    return cfg


def build(**over):
    cfg = config(**over)
    kit = Kit()
    add_throw_units(kit, cfg)
    reasons = add_yaniv_units(kit)
    reasons += add_score_units(kit)
    finish_yaniv(kit, reasons)
    fold_output(kit)
    return kit.build(), kit


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    out_path = Path(args[0]) if args else SCRATCH / "counts.json"
    net, kit = build()
    if "-v" in sys.argv:
        print(kit.describe())
    save_net(net, out_path, meta={"rules": {"hand": 5, "yaniv_at": 7}, "handmade": True,
                                  "source": "training/yaniv/handmade5_designs/counts.py"})
    print(f"{out_path}: layers {net.dims}, {sum(net.dims[1:-1])} hidden units, written by hand")
