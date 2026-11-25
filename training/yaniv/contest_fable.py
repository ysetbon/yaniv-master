"""Team fable's contest net for the real game (5 cards each, Yaniv at 7 or less). Weights written by hand.

    python -m training.yaniv.contest_fable        # writes public/model/yaniv-net-contest-fable.json
    python -m training.yaniv.contest_fable -v     # also prints every unit with its purpose

One plain network.MLP, shape [125, ..., 1], ReLU hidden layers, linear output, at most 1,020 hidden units.
Nothing is trained, fitted or searched: every unit is written with handmade_kit.Kit from a stated purpose, and
the constants are rounded measurements or reasoned numbers (results/contest-fable.md has every variant measured).
"""

from __future__ import annotations

import sys
from pathlib import Path

from training.yaniv.handmade_kit import RANK_NAMES as RANKS, RANK_POINTS, Kit
from training.yaniv.network import save_net

MODEL_PATH = Path(__file__).resolve().parents[2] / "public" / "model" / "yaniv-net-contest-fable.json"
MAX_HIDDEN = 1020

# ---------------------------------------------------------------- throw constants
TWO_LIGHT = 0.5     # a 2 counts as half a heavy card (the trained net throws 22 over a 3, keeps it over a 5+)
PAIR_OPEN = 5.0     # taking an open card onto a kept card of its rank: = CARD, it leaves with its twin next turn
                    # (the set's value itself is in the plan's G; reader's 9 double-counted with it)
OPEN_TELL = 1.5     # per point above 3 of an unpaired open card taken (trained takes unpaired 4s 35%, 5s 7%)
LOW_TOP = 1.0       # leaving a card worth <= 4 face up feeds the counter (it takes those 95% of the time)
DECK_HEAVY = 1.07   # chance a deck card is heavy ~ 1.07 - share of unseen worth <= 3 (the 3s count as heavy)
FEED = (1.0, 0.5)   # (flat, per point) for leaving face up a rank the opponent is KNOWN to hold: it takes a
                    # set-maker (clone's rollouts: ~2.5 points)
N_TYP = 38.0        # typical unseen count mid-round: P(deck card worth v) ~ unseen cards worth v / N_TYP

CATCH = 30.0        # an Assaf against the opponent: +30 to it

# ---------------------------------------------------------------- Yaniv constants
YBIG = 1000.0       # the Yaniv row is -YBIG or +YBIG
T_EARLY = 0.025     # reader's threshold on assaf: T_FLOOR + relu(T_EARLY - T_SLOPE * ply) + gap and 6-7 point terms:
T_SLOPE = 0.00125   # ... 0.04 at ply 0, the floor from ply 20 (opponents' hidden cards are lower late)
T_FLOOR = 0.015
GAP_SLOPE = 0.002   # threshold change per point of (opponent estimate - my points) ...
GAP_CENTER = 5.0    # ... around a gap of 5 ...
GAP_CAP = 15.0      # ... with the gap capped here
HIGH_CALL = 0.005   # per point above 5: the trained net calls a little more readily at 6-7 points
C1_SLOPE = 0.4      # caller's C1: call if opp_est - pts > C1_SLOPE pts + C1_BASE + C1_ASSAF assaf
C1_BASE = 1.3
C1_ASSAF = 10.0

# ---------------------------------------------------------------- score constants (scorer.py)
GATE = 200.0        # gate strength for the score units: larger than any value a gated sum takes
OPEN_CUT = {50: (1.5, 18.0), 100: (1.5, 18.0)}  # (flat, x threat) for an open take landing on 50 / 100:
                    # ~50 x P(opponent calls next turn) (scorer.py's measurement). scorer halved it at 100; the
                    # match-win table says a 100-cut is worth ~55 points-equivalent (50-cut: 24-51), so same weight
TELL_ON_CUT = True  # an open take that lands my score + hand on 50/100 pays no OPEN_TELL (see add_score_units)
DECK_CUT = (20.0, 20.0)  # the same for a deck draw, x P(the drawn card lands exactly on the cut)


# ---------------------------------------------------------------- helpers
def _step(kit: Kit, name: str, weights: dict, bias: float, note: str) -> dict:
    """Two layer-1 units whose difference is 1 if (weights . x + bias) >= 0, ramping to 0 within -1."""
    if name + "_a" not in kit.level:
        kit.unit(1, name + "_a", weights, bias + 1.0, note=note + " (ramp)")
        kit.unit(1, name + "_b", weights, bias, note=note + " (ramp shifted)")
    return {name + "_a": 1.0, name + "_b": -1.0}


def _est_under(kit: Kit, t: float) -> dict:
    """[opp_est < t points] (opp_est <= t - 0.1, width 0.1 point)."""
    return _step(kit, f"est_under{t:g}", {"opp_est": -400.0}, 10.0 * t - 1.0, f"opponent's estimated total < {t:g}")


def _max_all(kit: Kit, layer: int, name: str, items: list[dict]) -> tuple[dict, int]:
    """max of non-negative linear combinations, as a tournament: max(a, b) = relu(a - b) + relu(b).
    Returns the max (a combination of units of the last layer used) and that layer."""
    while len(items) > 1:
        nxt = []
        for i in range(0, len(items) - 1, 2):
            a, b = items[i], items[i + 1]
            diff = dict(a)
            for s, w in b.items():
                diff[s] = diff.get(s, 0.0) - w
            kit.unit(layer, f"{name}{layer}_{i}d", diff, note="max step: relu(a - b)")
            kit.unit(layer, f"{name}{layer}_{i}b", b, note="max step: relu(b) = b")
            nxt.append({f"{name}{layer}_{i}d": 1.0, f"{name}{layer}_{i}b": 1.0})
        if len(items) % 2:
            kit.unit(layer, f"{name}{layer}_last", items[-1], note="max step: odd one carried")
            nxt.append({f"{name}{layer}_last": 1.0})
        items = nxt
        layer += 1
    return items[0], layer - 1
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


# ---------------------------------------------------------------- Yaniv
def add_yaniv_units_hm5(kit: Kit) -> list[dict]:
    """The 'call' reasons, each a combination of units that is 0 or 1 (and 0 on every throw row)."""
    reasons = []
    kit.unit(1, "sure_call", {"sure_yaniv": 1.0, "yaniv": 1.0}, bias=-1.0, note="the call cannot be caught")
    reasons.append({"sure_call": 1.0})

    # reader: assaf < T(ply, gap, pts)
    kit.unit(1, "early", {"ply": -T_SLOPE * 48}, bias=T_EARLY,
             note="threshold part that falls with the ply: T_EARLY at ply 0, 0 from ply T_EARLY/T_SLOPE")
    kit.unit(1, "gap", {"opp_est": 40.0, "my_pts": -40.0}, note="opponent's estimated total minus mine, >= 0")
    kit.unit(1, "gap_cap", {"opp_est": 40.0, "my_pts": -40.0}, bias=-GAP_CAP, note="the gap above GAP_CAP")
    kit.unit(1, "high_call", {"my_pts": 40.0}, bias=-5.0, note="my points above 5")
    t_terms = {"early": 1.0, "gap": GAP_SLOPE, "gap_cap": -GAP_SLOPE, "high_call": HIGH_CALL}
    margin = {s: 500.0 * w for s, w in t_terms.items()}
    margin.update({"assaf": -500.0, "yaniv": 100.0})
    b = -100.0 + 500.0 * (T_FLOOR - GAP_SLOPE * GAP_CENTER)
    kit.unit(2, "call", margin, bias=b, note="Yaniv row: 500 (T - assaf)")
    kit.unit(2, "call_sat", dict(margin), bias=b - 1.0, note="call - 1: call - call_sat = [assaf < T]")
    reasons.append({"call": 1.0, "call_sat": -1.0})

    # caller's C1, x10 so the ramp is 0.1 point wide: 10 (opp_est - pts - C1_SLOPE pts - C1_BASE - C1_ASSAF assaf)
    w = {"opp_est": 400.0, "my_pts": -400.0 * (1.0 + C1_SLOPE), "assaf": -10.0 * C1_ASSAF, "yaniv": YBIG * 10}
    kit.unit(1, "c1_a", w, bias=-10.0 * C1_BASE + 1.0 - YBIG * 10, note="C1 margin ramp on the Yaniv row")
    kit.unit(1, "c1_b", dict(w), bias=-10.0 * C1_BASE - YBIG * 10, note="ramp shifted")
    reasons.append({"c1_a": 1.0, "c1_b": -1.0})
    return reasons


# ---------------------------------------------------------------- Yaniv (this net's rule)
Y_A = {"1": 0.04, "2": 0.021, "3": 0.0055, "4": 0.0021}   # threshold per "point" for 1 / 2 / 3 / 4 opponent cards
Y_LOW = 0.005          # always call under this assaf (against 5 cards this is the only risky call made)
Y_BASE = 5.5           # threshold "points" = BASE + PTS_W x my points - PLY_W x clip(ply, PLY_LO, PLY_HI)
Y_PTS_W = 0.5          # (= 5 + 0.5 (pts - 5) + 0.2 (15 - ply): 5 at 5 points, ply 15)
Y_PLY_W = 0.2
Y_PLY_LO, Y_PLY_HI = 8.0, 20.0
Y_SWAP = 2.0           # veto reason 3 when best_shed - open card >= SWAP points
Y_K = 2000.0           # ramp steepness on assaf: a reason goes 0 -> 1 over 1/K = 0.0005 of assaf, centred on T
Y_G = 4000.0           # gate: larger than K x any threshold


def add_yaniv_units(kit: Kit, a=None, low=Y_LOW, base=Y_BASE, pts_w=Y_PTS_W, ply_w=Y_PLY_W, ply_lo=Y_PLY_LO,
                    ply_hi=Y_PLY_HI, swap=Y_SWAP, swap_on=True) -> list[dict]:
    """Call Yaniv iff (measured in results/contest-fable-logs/yaniv.md)
      1. the call cannot be caught (sure_yaniv), or
      2. assaf < Y_LOW, or
      3. no lower hand is one swap away AND assaf < Y_A[opp cards] x (Y_BASE + Y_PTS_W x my points - Y_PLY_W x clip(ply)).
         "One swap away": my highest group is worth Y_SWAP or more points above the open card (best_shed, open_pts):
         throwing it and taking the open card beats a call that can be caught (+4 points each time).
    The uniform catch chance `assaf` understates the true catch rate about x20 against 4-5 cards, x6 against 3,
    x2.5 against 2 and x1 against 1 (the same for the counter, hm5 and the trained net), so the threshold is
    set by the opponent's card count; the break-even true catch rate is about 20%."""
    a = {**Y_A, **(a or {})}
    reasons = []
    kit.unit(1, "sure_call", {"sure_yaniv": 1.0, "yaniv": 1.0}, bias=-1.0, note="the call cannot be caught")
    reasons.append({"sure_call": 1.0})

    w = {"assaf": -Y_K, "yaniv": Y_G}
    kit.unit(1, "ylow_a", w, bias=Y_K * low + 0.5 - Y_G, note="Yaniv row: Y_K (LOW - assaf) + 0.5")
    kit.unit(1, "ylow_b", dict(w), bias=Y_K * low - 0.5 - Y_G, note="... - 0.5: ylow_a - ylow_b = [assaf < LOW]")
    reasons.append({"ylow_a": 1.0, "ylow_b": -1.0})

    kit.unit(1, "yply_lo", {"ply": 48.0}, bias=-ply_lo, note="relu(ply - PLY_LO)")
    kit.unit(1, "yply_hi", {"ply": 48.0}, bias=-ply_hi, note="relu(ply - PLY_HI): clip(ply) = LO + lo - hi")
    veto = {}
    if swap_on:
        d = {"best_shed": 40.0, "open_pts": -10.0}
        kit.unit(1, "yswap_a", d, bias=-(swap - 1.0), note="relu(best_shed - open - (SWAP - 1))")
        kit.unit(1, "yswap_b", dict(d), bias=-swap, note="yswap_a - yswap_b = [my highest group >= open card + SWAP]")
        veto = {"yswap_a": -Y_G, "yswap_b": Y_G}
    for c in ("1", "2", "3", "4"):
        t = Y_K * a[c]
        m = {"assaf": -Y_K, "my_pts": t * pts_w * 40.0, "yply_lo": -t * ply_w, "yply_hi": t * ply_w,
             "yaniv": Y_G, f"opp_has_{c}": Y_G, **veto}
        b = t * (base - ply_w * ply_lo) - 2.0 * Y_G
        kit.unit(2, f"ycall{c}_a", m, bias=b + 0.5, note=f"Yaniv row, opponent holds {c}, no swap: Y_K (T - assaf) + 0.5")
        kit.unit(2, f"ycall{c}_b", dict(m), bias=b - 0.5, note="... - 0.5: the difference is [assaf < T]")
        reasons.append({f"ycall{c}_a": 1.0, f"ycall{c}_b": -1.0})
    return reasons


# ---------------------------------------------------------------- the triple take
TRIP = 40.0          # larger than the points the pair-throw line is ahead by on hm5's own count (~3p + 5)
TRIP_MIN = 7         # ranks worth this much or more
TRIP_OPP = (4, 5)    # opponent card counts where it applies
TRIP_MIN5 = 5        # ranks worth this much or more when the opponent has 5 cards (5s and 6s: only then)


def add_triple_units(kit: Kit, trip=TRIP, trip_min=TRIP_MIN, trip_opp=TRIP_OPP, trip_min5=None) -> None:
    """The triple take (results/contest-fable-logs/survey.md): the open card has the rank of a PAIR I hold. Keep the
    pair, throw another card and take the open card: three of a kind leave together next turn, one card fewer than
    throwing the pair now. Worth +3 to +5 points each time for ranks worth 7+ against 4-5 cards and for 5s / 6s
    against 5 cards; it loses against 1-3 cards. One layer-1 unit per rank:
        trip_r = relu(3 open_r + 2 take_open + 4 hand_r - 10 throw_r + [gate] - 7)
    = 1 when I hold two r, take the open r and do not throw r."""
    trip_min5 = trip_min if trip_min5 is None else trip_min5
    out = {}
    gate = {f"opp_has_{c}": 1.0 for c in trip_opp}
    for r, p in zip(RANKS, RANK_POINTS):
        if p < min(trip_min, trip_min5):
            continue
        gate = {f"opp_has_{c}": 1.0 for c in (trip_opp if p >= trip_min else (5,))}
        kit.unit(1, f"trip_{r}", {f"open_{r}": 3.0, "take_open": 2.0, f"hand_{r}": 4.0, f"throw_{r}": -10.0, **gate},
                 bias=-7.0, note=f"takes an open {r} onto a kept pair of {r}s, opponent has {trip_opp} cards")
        out[f"trip_{r}"] = trip
    kit.out(out)


def finish_yaniv(kit: Kit, reasons: list[dict]) -> None:
    """Yaniv row = -YBIG + 2 YBIG * min(1, sum of reasons): no reason can push it past +YBIG, and throws
    (the reasons are all 0 on throw rows) never get near either end."""
    total: dict = {}
    for r in reasons:
        for s, w in r.items():
            total[s] = total.get(s, 0.0) + w
    layer = max(kit.level[s] for s in total) + 1
    kit.unit(layer, "ycall_a", total, note="sum of call reasons (0 on throw rows)")
    kit.unit(layer, "ycall_b", dict(total), bias=-1.0, note="ycall_a - ycall_b = min(1, reasons)")
    kit.out({"yaniv": -YBIG, "ycall_a": 2 * YBIG, "ycall_b": -2 * YBIG})


# ---------------------------------------------------------------- score units
def _threat(kit: Kit) -> dict:
    """threat in 0..1: chance-like weight that the opponent calls next turn (1 at opp_est <= 9, 0 at >= 15)."""
    if "threat_a" not in kit.level:
        kit.unit(1, "threat_a", {"opp_est": -40.0}, bias=15.0, note="relu(15 - opponent's estimated total)")
        kit.unit(1, "threat_b", {"opp_est": -40.0}, bias=9.0, note="(threat_a - threat_b)/6: 1 at <= 9, 0 at >= 15")
    return {"threat_a": 1.0 / 6, "threat_b": -1.0 / 6}


def add_score_units(kit: Kit) -> list[dict]:
    """Match-score units (scorer.add_score_units; 100-cut weight raised to the 50-cut's, OPEN_TELL given back on a
    cut-landing open take). Returns an extra Yaniv reason.

    1. open card: my score + held points after taking it == 50 or 100 (exact: s = 120 my_score + 40 pts_kept
       + 10 incoming, a tent of width 1). If the opponent calls next turn my score is cut by 50:
       bonus (flat + x threat) from OPEN_CUT.
    2. deck draw: lands on the cut if the drawn card is worth exactly target - score - kept;
       bonus DECK_CUT x P(that value) x threat, the product as relu(P + 0.45 threat - 0.45) (P <= ~0.45).
    3. assaf on cut: my score + my points + 30 == 50 or 100: even a caught call lowers my score (it is
       cut), and an uncaught one costs nothing, so call. scorer.py gives this +400 on the Yaniv row; here it is
       a call REASON fed into the saturating Yaniv unit, so it always lifts the row to +YBIG (= 1000) and
       no other term can outvote it. Throw-side bonuses are at most ~20 points, far inside +-YBIG.
    """
    threat = _threat(kit)
    s = {"my_score": 120.0, "pts_kept": 40.0, "incoming": 10.0, "take_open": GATE}
    for at, (flat, x_threat) in OPEN_CUT.items():
        pieces = (("lo", -(at - 1), 1.0), ("mid", -at, -2.0), ("hi", -(at + 1), 1.0))
        for name, off, wt in pieces:
            kit.unit(1, f"opencut{at}_{name}", s, bias=off - GATE,
                     note=f"tent piece: my score + held points after taking the open card == {at}")
            kit.out({f"opencut{at}_{name}": flat * wt})
        w = {f"opencut{at}_{name}": GATE * wt for name, _, wt in pieces}
        w.update(threat)
        kit.unit(2, f"opencut{at}_threat", w, bias=-GATE, note=f"threat when the open-card take lands on {at}")
        kit.out({f"opencut{at}_threat": x_threat})
        if TELL_ON_CUT:
            # OPEN_TELL charges an unpaired high open card as a lasting heavy card / a tell; a card taken to land
            # exactly on the cut is wanted for its value this turn, so the charge is given back (diagnosis: the
            # charge, 10.5 for a 10, out-weighed the cut bonus and halved the cut takes vs merge_exact).
            w = {f"opencut{at}_{name}": 10.0 * wt for name, _, wt in pieces}
            w["open_tell"] = 1.0
            kit.unit(2, f"opencut{at}_tell", w, bias=-10.0, note=f"open_tell when the open-card take lands on {at}")
            kit.out({f"opencut{at}_tell": OPEN_TELL})
    s = {"my_score": 120.0, "pts_kept": 40.0, "draw_deck": GATE}
    for at, weight in zip((50, 100), DECK_CUT):
        chance = {}
        for v in range(0, 11):
            pieces = (("lo", -(at - v - 1), 1.0), ("mid", -(at - v), -2.0), ("hi", -(at - v + 1), 1.0))
            for name, off, wt in pieces:
                kit.unit(1, f"deckcut{at}_{v}_{name}", s, bias=off - GATE,
                         note=f"tent piece: deck draw and my score + kept == {at} - {v}")
            w = {f"deckcut{at}_{v}_{name}": GATE * wt for name, _, wt in pieces}
            for r, p in zip(RANKS, RANK_POINTS):
                if p == v:
                    w[f"unseen_{r}"] = 4.0 / 34.0
            kit.unit(2, f"deckcut{at}_{v}", w, bias=-GATE, note=f"chance the deck card is worth {v} (lands on {at})")
            chance[f"deckcut{at}_{v}"] = 1.0
        kit.unit(3, f"deckcut{at}", {**chance, **{k: 0.45 * x for k, x in threat.items()}}, bias=-0.45,
                 note=f"~ chance the deck draw lands on {at} x threat")
        kit.out({f"deckcut{at}": weight})
    reason = {}
    a = {"my_score": 120.0, "my_pts": 40.0, "yaniv": GATE}
    for at in (50, 100):
        for name, off, wt in (("lo", -(at - 31), 1.0), ("mid", -(at - 30), -2.0), ("hi", -(at - 29), 1.0)):
            kit.unit(1, f"assafcut{at}_{name}", a, bias=off - GATE,
                     note=f"tent piece: Yaniv row and my score + my points + 30 == {at}")
            reason[f"assafcut{at}_{name}"] = wt
    return [reason]


# ---------------------------------------------------------------- assembly
def fold_output(kit: Kit) -> None:
    """Bookkeeping, no new weights: instead of carrying every output term forward layer by layer (one
    pass-through unit per term per layer), sum the terms available at each layer into a running total kept
    as two units, relu(total) and relu(-total). Exact, and it keeps the net small."""
    terms, bias = kit.output
    depth = max(kit.level[s] for s in terms)
    acc: dict = {}
    for layer in range(1, depth + 1):
        part = dict(acc)
        for s, w in terms.items():
            if kit.level[s] == layer - 1:
                part[s] = part.get(s, 0.0) + w
        kit.unit(layer, f"total{layer}_pos", part, note="running output total, positive part")
        kit.unit(layer, f"total{layer}_neg", {s: -w for s, w in part.items()}, note="... negative part")
        acc = {f"total{layer}_pos": 1.0, f"total{layer}_neg": -1.0}
    final = {s: w for s, w in terms.items() if kit.level[s] == depth}
    for s, w in acc.items():
        final[s] = final.get(s, 0.0) + w
    kit.output = (final, bias)


def build(yaniv: str = "new", triple: bool = True, **over):
    """The contest net. `yaniv="hm5"`, `triple=False` and the table overrides in `config` rebuild earlier variants
    (yaniv="hm5", triple=False is handmade5 exactly)."""
    cfg = config(**over)
    kit = Kit()
    add_throw_units(kit, cfg)
    if triple:
        add_triple_units(kit, TRIP, TRIP_MIN, TRIP_OPP, TRIP_MIN5)
    reasons = add_yaniv_units(kit) if yaniv == "new" else add_yaniv_units_hm5(kit)
    reasons += add_score_units(kit)
    finish_yaniv(kit, reasons)
    fold_output(kit)
    net = kit.build()
    hidden = sum(net.dims[1:-1])
    if hidden > MAX_HIDDEN:
        raise ValueError(f"{hidden} hidden units: the contest allows {MAX_HIDDEN}")
    return net, kit


if __name__ == "__main__":
    net, kit = build()
    if "-v" in sys.argv:
        print(kit.describe())
    save_net(net, MODEL_PATH, meta={"rules": {"hand": 5, "yaniv_at": 7}, "handmade": True,
                                    "source": "training/yaniv/contest_fable.py"})
    print(f"{MODEL_PATH.name}: layers {net.dims}, {sum(net.dims[1:-1])} hidden units, written by hand")
