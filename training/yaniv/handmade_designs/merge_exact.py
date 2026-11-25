"""`merge_exact`: clone.py's exact rebuild of the card counter, plus the best findings of the other round-1 designs.

Every weight is written by hand from reasoning; nothing is trained, fitted or searched. The variants tried and
their dev-test lines are in SCRATCH/logs/merge_exact.md.

Base (copied from clone.py BEST, see its docstring for the derivations). A throw is scored, in points, by
    - [ w * E(points held after the draw) + (1 - w) * E(points after my NEXT best throw) ]
    - top penalty - feeding penalty + catch bonus
  * E(points after the next throw) = points - E[maxgroup], maxgroup the heaviest single-rank group, exact over the
    draw: the open card's group, or the deck average via the one-hot of the unseen count n (exact 1/n);
  * w = the chance the opponent calls before my next turn (+0.05), read from its lean-low estimate opp_est;
  * catch bonus: 30 * P(opponent calls next) * P(my hand after the draw <= its likely total): an Assaf for it;
  * feeding penalty: leaving face up a rank the opponent is known to hold gives it a set.

Additions in this file (each a reasoned change, measured on the dev test; BEST = what helped):
  add_throw_units  heavy-card charge 3 x (1 - w) per heavy card held (reader)          dev margin +0.89 -> +1.18  [BEST]
                   open-card tell, callable-next vs a 4-card opponent (reader)         off: no gain / -0.10
  add_yaniv_units  caller's C1 margin rule, needing 2.3 points of margin (not 1.3):
                   calls with less margin are caught ~14% and playing on (the catch
                   bonus's Assaf trap) is worth as much                               +1.18 -> +1.26 -> +1.30  [BEST]
                   clone's / reader's / reader-AND-C1 rules kept as options            +1.18 / +1.26 / +1.24
  add_score_units  scorer.py's match-score units (copied)                              matches 60.0% -> 68.3%   [BEST]
Dev test: 12,000 rounds seed 52000, 3,000 matches seed 52001; trained net +1.39, matches 60.5% there.

    python -m training.yaniv.handmade_designs.merge_exact      # writes SCRATCH/nets/merge_exact.json
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("YANIV_HAND_SIZE", "4")
os.environ.setdefault("YANIV_CALL_AT", "7")

from training.yaniv.handmade_kit import RANK_NAMES, RANK_POINTS, Kit  # noqa: E402
from training.yaniv.network import save_net  # noqa: E402

SCRATCH = "/tmp/claude-0/-home-user-yaniv-master/ecf1221d-8c18-559f-9f3b-91b1597261e5/scratchpad"

R13 = list(range(13))  # ranks A..K (jokers are worth 0 and never form a heavy group)
P = RANK_POINTS
HAND = [f"hand_{r}" for r in RANK_NAMES]
THROW = [f"throw_{r}" for r in RANK_NAMES]
UNSEEN = [f"unseen_{r}" for r in RANK_NAMES]
OPEN = [f"open_{r}" for r in RANK_NAMES]
KNOWN = [f"known_{r}" for r in RANK_NAMES]

# ------------------------------------------------------------------ constants (with their reasoning)
# The counter's top penalty: a low card left face up is a cheap card for the opponent.
TOP_PEN = [1.8 if p <= 1 else 1.0 if p <= 3 else 0.35 if p <= 5 else 0.0 for p in RANK_POINTS]
# The counter's call threshold on assaf by my points 0..7 (0.40,0.40,0.28,0.18,0.12,0.08,0.06,0.04),
# written as a bias plus slope changes at 1..5 points.
THR = {"bias": 0.40, 1: -0.12, 2: 0.02, 3: 0.04, 4: 0.02, 5: 0.02}
# opp_est buckets (points) and, per bucket: w = weight of current points = P(opponent calls next) + 0.05;
# pi = P(opponent calls next) (clone's measurement); m = its typical total when it calls (I catch it at <= m).
EST_CUTS = [5, 8, 11, 14, 17]
EST_W = [0.60, 0.35, 0.23, 0.13, 0.09, 0.06]
EST_PI = [0.55, 0.30, 0.18, 0.08, 0.04, 0.01]
EST_M = [3, 5, 5, 6, 6, 6]
CATCH = 30.0  # an Assaf against the opponent swings ~30 + its points
FEED = (1.0, 0.5)  # leaving face up a rank the opponent holds: 1 + 0.5 * its points

DEFAULT = dict(
    # ---- throws
    heavy=0.0,          # charge per heavy card (not A / joker; a 2 is half) held after the move, x (1 - w)
    open_tell=0.0,      # per point above 3 of an unpaired open card taken
    callable_vs4=0.0,   # bonus for a move that makes Yaniv sure next turn while the opponent holds 4 cards
    # ---- Yaniv
    call="clone",       # "clone" | "reader" | "c1" | "blend"
    c1_bias=1.3,        # C1: call when opp_est - 1.4 my_pts - 10 assaf >= c1_bias (points)
    # ---- score
    score=False,
)
BEST = dict(DEFAULT, heavy=3.0, call="c1", c1_bias=2.3, score=True)  # see the log: dev +1.30, matches with score units


# ================================================================== shared helper units
def common_units(kit: Kit) -> None:
    """Integer decodings used by several parts (layers 1-2)."""
    for r in R13:
        kit.unit(1, f"tk{r}", {"n_thrown": 1.0, THROW[r]: 1.0}, -1.0, "cards of rank r thrown / 4")
    for k in range(0, 52):
        kit.unit(1, f"rn{k}", {"unseen_n": 40.0}, -float(k), "relu(n - k), n = unseen cards")
    for s in R13:
        for j in range(4):
            kit.unit(1, f"us{s}_{j}", {UNSEEN[s]: 4.0}, -float(j), "relu(unseen_s - j)")
    for c in range(1, 6):
        kit.unit(1, f"pr{c}", {"my_pts": 40.0}, -float(c), "relu(my points - c)")
    for k in range(1, 51):
        kit.unit(2, f"ni{k}", {f"rn{k-1}": 1.0, f"rn{k}": -2.0, f"rn{k+1}": 1.0}, 0.0, "1 if n == k")
    for s in R13:
        for j in range(1, 4):
            kit.unit(2, f"ui{s}_{j}", {f"us{s}_{j-1}": 1.0, f"us{s}_{j}": -1.0}, 0.0, "1 if unseen_s >= j")


def est_buckets(kit: Kit, cuts: list[float]) -> list[str]:
    """One-hot (layer 3) of the bucket of opp_est (in points) between the cuts."""
    if "eb0" in kit.level:
        return [f"eb{b}" for b in range(len(cuts) + 1)]
    Kst = 1000.0
    for i, t in enumerate(cuts):
        kit.unit(2, f"es{i}_p", {"opp_est": 40.0 * Kst}, -Kst * t + 1.0, "ramp est - t")
        kit.unit(2, f"es{i}_n", {"opp_est": 40.0 * Kst}, -Kst * t, "ramp shifted")
    names = []
    for b in range(len(cuts) + 1):
        w = {}
        if b > 0:
            w.update({f"es{b-1}_p": 1.0, f"es{b-1}_n": -1.0})
        if b < len(cuts):
            w.update({f"es{b}_p": -1.0, f"es{b}_n": 1.0})
        kit.unit(3, f"eb{b}", w, 0.0 if b > 0 else 1.0, "1 if opp_est in bucket b")
        names.append(f"eb{b}")
    return names


def p_le_units(kit: Kit, m: int) -> dict[str, float]:
    """P(points after the draw <= m), as weights on layer <= 4 units (exact: one-hot on kept points and n).

    K = points kept. Deck: #(unseen worth <= m - K) / n. Open card: 1 if K + open points <= m.
    """
    if "kr0" not in kit.level:
        for j in range(-1, 9):
            kit.unit(1, f"kr{j}", {"pts_kept": 40.0}, -float(j), "relu(K - j)")
        for k in range(0, 8):
            kit.unit(2, f"kh{k}", {f"kr{k-1}": 1.0, f"kr{k}": -2.0, f"kr{k+1}": 1.0}, 0.0, "1 if K == k")
    tag = f"m{m}"
    if f"{tag}pc1" in kit.level:
        return {**{f"{tag}pc{n}": 1.0 for n in range(1, 51)}, f"{tag}oc": 1.0}
    kit.unit(1, f"{tag}o1", {"pts_kept": -40.0, "open_pts": -10.0}, m + 1.0, "relu(m + 1 - K - open)")
    kit.unit(1, f"{tag}o0", {"pts_kept": -40.0, "open_pts": -10.0}, float(m), "relu(m - K - open)")
    kit.unit(2, f"{tag}oc", {f"{tag}o1": 1.0, f"{tag}o0": -1.0, "take_open": 1.0}, -1.0,
             "takes the open card and lands on m or less")
    for k in range(0, m + 1):
        low = {UNSEEN[r]: 4.0 for r in range(14) if RANK_POINTS[r] <= m - k}
        kit.unit(3, f"{tag}lc{k}", {**low, f"kh{k}": 60.0}, -60.0, "unseen cards that keep me at m or less")
    for n in range(1, 51):
        w = {f"{tag}lc{k}": 1.0 / n for k in range(m + 1)}
        w[f"ni{n}"] = 100.0
        w["draw_deck"] = 100.0
        kit.unit(4, f"{tag}pc{n}", w, -200.0, "P(hand <= m) for a deck draw, if n == unseen count")
    return {**{f"{tag}pc{n}": 1.0 for n in range(1, 51)}, f"{tag}oc": 1.0}


# ================================================================== throws
def add_throw_units(kit: Kit, v: dict) -> None:
    """The counter's expected-hand evaluation, rebuilt exactly (clone.py), with clone's and the new terms."""
    out: dict[str, float] = {}

    # ---- kept groups G_r (points of rank r kept), and the open card's rank group
    for r in R13:
        kit.unit(2, f"G{r}", {HAND[r]: 4.0 * P[r], f"tk{r}": -4.0 * P[r]}, 0.0, "points of rank r kept")
    for s in R13:
        kit.unit(3, f"go{s}", {f"G{s}": 1.0, OPEN[s]: 40.0}, -40.0, "kept points of the open card's rank")

    # ---- M = max_r G_r by a tournament, max(a, b) = relu(a - b) + b (layers 3-6)
    def tourney(layer, vals, tag):
        nxt = []
        for i in range(0, len(vals) - 1, 2):
            a, b = vals[i], vals[i + 1]
            d = dict(a)
            for k, w in b.items():
                d[k] = d.get(k, 0.0) - w
            kit.unit(layer, f"{tag}d{i}", d, 0.0, "relu(a - b)")
            kit.unit(layer, f"{tag}c{i}", dict(b), 0.0, "b")
            nxt.append({f"{tag}d{i}": 1.0, f"{tag}c{i}": 1.0})
        if len(vals) % 2:
            kit.unit(layer, f"{tag}x", dict(vals[-1]), 0.0, "carried")
            nxt.append({f"{tag}x": 1.0})
        return nxt

    vals = [{f"G{r}": 1.0} for r in R13]
    layer = 3
    while len(vals) > 1:
        vals = tourney(layer, vals, f"t{layer}")
        layer += 1
    Mlin, L = vals[0], layer  # L = 7
    negM = {k: -w for k, w in Mlin.items()}

    # ---- layer 7: pieces of E[maxgroup]
    kit.unit(L, "M", Mlin, 0.0, "heaviest kept group")
    for t in range(0, 11):
        kit.unit(L, f"a{t}", dict(negM), float(t), "relu(t - M)")
    for s in R13:
        kit.unit(L, f"e{s}", {**negM, f"G{s}": 1.0}, float(P[s]), "relu(p_s + G_s - M): group if a rank-s card arrives, above M")
    kit.unit(L, "ov", {**negM, "open_pts": 10.0, **{f"go{s}": 1.0 for s in R13}}, 0.0, "open card's group above M")

    # ---- layer 8: T = sum of gains over the unseen cards; the open card's gain
    for t in range(1, 11):
        w = {UNSEEN[r]: 4.0 for r in R13 if P[r] >= t}
        w[f"a{t}"] = 60.0
        w[f"a{t-1}"] = -60.0
        kit.unit(L + 1, f"h{t}", w, -60.0, "C(t) if M < t: unseen cards worth >= t")
    for s in R13:
        for j in range(1, 4):
            kit.unit(L + 1, f"q{s}_{j}", {f"e{s}": 1.0, f"a{P[s]}": -1.0, f"ui{s}_{j}": 40.0}, -40.0,
                     "gain from drawing the j-th unseen card of kept rank s")
    kit.unit(L + 1, "op", {"ov": 1.0, "take_open": 100.0}, -100.0, "open-card gain, if taken")

    # ---- layer 9: T / n, exact through the one-hot of n
    Tw = {f"h{t}": 1.0 for t in range(1, 11)}
    Tw.update({f"q{s}_{j}": 1.0 for s in R13 for j in range(1, 4)})
    for k in range(1, 51):
        w = {name: x / k for name, x in Tw.items()}
        w[f"ni{k}"] = 1000.0
        w["draw_deck"] = 1000.0
        kit.unit(L + 2, f"v{k}", w, -2000.0, "T / n if n == k and a deck draw")

    # ---- layer 10: (1 - w_b) * E[maxgroup], w_b by the opp_est bucket
    Ew = {"M": 1.0, "op": 1.0, **{f"v{k}": 1.0 for k in range(1, 51)}}
    inds = est_buckets(kit, EST_CUTS)
    for b, ind in enumerate(inds):
        kit.unit(L + 3, f"zb{b}", {**Ew, ind: 100.0}, -100.0, "E[maxgroup] if opp_est in bucket b")
        out[f"zb{b}"] = 1.0 - EST_W[b]

    # ---- catch bonus: CATCH * pi_b * P(my hand after the draw <= m_b)
    for b, ind in enumerate(inds):
        pm = p_le_units(kit, EST_M[b])
        kit.unit(5, f"cat{b}", {**pm, ind: 1.0}, -1.0, "P(hand <= m_b) if opp_est in bucket b")
        out[f"cat{b}"] = CATCH * EST_PI[b]

    # ---- points held after the draw (exact mean), and the counter's top penalty
    eps = 1e-3  # the counter breaks badness ties by expected points
    out["pts_kept"] = -40.0 * (1 + eps)
    out["incoming"] = -10.0 * (1 + eps)
    for r in range(14):
        if TOP_PEN[r]:
            out[THROW[r]] = -TOP_PEN[r]

    # ---- feeding penalty: leaving face up a rank the opponent is known to hold
    for r in range(14):
        kit.unit(1, f"feed{r}", {THROW[r]: 1.0, KNOWN[r]: 4.0}, -1.0, "top rank the opponent holds")
        out[f"feed{r}"] = -(FEED[0] + FEED[1] * P[r])

    # ---- heavy-card charge (reader): a hand with fewer heavy cards stays smaller for the rest of the round,
    # so a pair of 3+ is thrown rather than a higher single. It pays off only on future turns, so it is weighted
    # by (1 - w_b), like the next-throw term.
    if v["heavy"]:
        kit.unit(1, "light_thrown", {"n_thrown": 4.0, THROW[0]: 4.0, THROW[13]: 4.0}, -4.0,
                 "cards thrown if they are aces or jokers")
        kit.unit(1, "twos_thrown", {"n_thrown": 4.0, THROW[1]: 4.0}, -4.0, "cards thrown if they are 2s")
        kit.unit(1, "took_light", {"take_open": 1.0, OPEN[0]: 1.0, OPEN[13]: 1.0}, -1.0, "takes an open A or joker")
        kit.unit(1, "took_two", {"take_open": 1.0, OPEN[1]: 1.0}, -1.0, "takes an open 2")
        kit.unit(1, "deck_heavy", {"draw_deck": 3.07, "unseen_low": -1.0}, -2.0,
                 "deck draw: ~chance the card is heavy (1.07 - share of unseen worth <= 3)")
        heavy = {"n_kept": 5.0, HAND[0]: -4.0, HAND[13]: -4.0, "light_thrown": 1.0,
                 HAND[1]: -2.0, "twos_thrown": 0.5,
                 "take_open": 1.0, "took_light": -1.0, "took_two": -0.5, "deck_heavy": 1.0}
        if v.get("heavy_pairs_open", True):
            # an open card taken onto a kept card of its rank leaves with that set next turn: no lasting card
            kit.unit(1, "took_set_heavy", {"pairs_open": 1.0, OPEN[0]: -1.0, OPEN[13]: -1.0, OPEN[1]: -0.5}, 0.0,
                     "1 (0.5 for a 2) if the open card taken pairs a kept card")
            heavy["took_set_heavy"] = -1.0
        for b, ind in enumerate(inds):
            kit.unit(L + 3, f"hv{b}", {**heavy, ind: 10.0}, -10.0, "heavy cards after the move, if bucket b")
            out[f"hv{b}"] = -v["heavy"] * (1.0 - EST_W[b])

    # ---- open-card tell (reader): an unpaired open card taken is shown to the opponent (its sure-call and
    # catch counts use it) while a deck card is hidden; charge per point above 3.
    if v["open_tell"]:
        kit.unit(1, "open_tell", {"open_pts": 10.0, "take_open": 10.0, "pairs_open": -10.0}, -13.0,
                 "points above 3 of an unpaired open card taken")
        out["open_tell"] = -v["open_tell"]

    # ---- callable next turn while the opponent still holds 4 cards: a call then is almost never caught
    if v["callable_vs4"]:
        kit.unit(1, "callable_vs4", {"callable_next": 1.0, "opp_has_4": 1.0}, -1.0,
                 "makes Yaniv sure next turn while the opponent holds 4 cards")
        out["callable_vs4"] = v["callable_vs4"]

    kit.out(out)


# ================================================================== Yaniv
K_STEP = 2000.0  # step sharpness: a margin of 1/K_STEP in assaf separates call from no call


def _step(kit: Kit, layer: int, name: str, w: dict[str, float], bias: float, k: float = K_STEP) -> dict:
    """Two units whose difference is 1 if (w . x + bias) >= 0 and 0 if it is <= -1/k."""
    kw = {s: k * x for s, x in w.items()}
    kit.unit(layer, name + "_p", kw, k * bias + 1.0, "ramp: k * margin + 1")
    kit.unit(layer, name + "_n", dict(kw), k * bias, "ramp: k * margin")
    return {name + "_p": 1.0, name + "_n": -1.0}


def _clone_thr(scale: float) -> tuple[dict, float]:
    """scale * thr(my_pts) - assaf, as (weights, bias)."""
    w = {f"pr{c}": scale * THR[c] for c in range(1, 6)}
    w["assaf"] = -1.0
    return w, scale * THR["bias"]


def add_yaniv_units(kit: Kit, v: dict) -> None:
    """The Yaniv row scores -200 + 400 * ycall, ycall in {0, 1}: above every throw exactly when calling is right.

    Every rule calls a sure Yaniv (it cannot be caught).
    """
    rule = v["call"]
    terms: dict[str, float] = {}
    if rule in ("clone", "blend"):
        # clone: assaf <= 0.5 thr, or (assaf <= 1.75 thr and opp_est - my_pts >= 3); true risk is ~3x assaf, but
        # only ~1.5-2x when the opponent looks 3+ points higher.
        ca = _step(kit, 2, "ca", *_clone_thr(1.75))
        cb = _step(kit, 2, "cb", *_clone_thr(0.5))
        dg = _step(kit, 2, "dg", {"opp_est": 40.0, "my_pts": -40.0}, -3.0, k=10.0)
        kit.unit(3, "cAD", {**ca, **dg}, -1.0, "loose threshold and opponent looks high")
        kit.unit(3, "cB", cb, 0.0, "strict threshold")
        clone_terms = {"cAD": 1.0, "cB": 1.0}
    if rule in ("reader", "blend"):
        # reader: T = 0.015 + relu(0.025 - 0.00125 ply) + 0.002 (min(gap, 15) - 5) + 0.005 relu(pts - 5)
        kit.unit(1, "early", {"ply": -0.00125 * 48}, 0.025, "threshold part falling with the ply")
        kit.unit(1, "gap", {"opp_est": 40.0, "my_pts": -40.0}, 0.0, "opp_est - my points, >= 0")
        kit.unit(1, "gap_cap", {"opp_est": 40.0, "my_pts": -40.0}, -15.0, "gap above 15")
        rw = {"early": 1.0, "gap": 0.002, "gap_cap": -0.002, "pr5": 0.005, "assaf": -1.0}
        rd = _step(kit, 2, "rd", rw, 0.015 - 0.002 * 5)
        if rule == "reader":
            kit.unit(3, "cR", rd, 0.0, "reader's threshold")
        reader_terms = {"cR": 1.0}
    if rule == "clone":
        terms = clone_terms
    elif rule == "reader":
        terms = reader_terms
    if rule in ("c1", "blend"):
        # caller C1: opp_est - my_pts > 0.4 my_pts + 1.3 (+ a small assaf guard)
        c1 = _step(kit, 2, "c1", {"opp_est": 40.0, "my_pts": -40.0 * 1.4, "assaf": -10.0}, -v["c1_bias"], k=4.0)
    if rule == "c1":
        kit.unit(3, "cC1", c1, 0.0, "caller's margin rule")
        terms = {"cC1": 1.0}
    elif rule == "blend":
        # call only when BOTH reader's (ply-falling assaf threshold) and caller's C1 (margin over opp_est) agree.
        # Measured on h3_c1's own Yaniv-legal turns (calldiag + rules.py): where only C1 calls the true catch
        # rate is 13% and playing on (with the catch bonus keeping my hand low, the "Assaf trap") averaged
        # +4.8 vs +4.1 for calling; where only reader calls it is 21%, about break-even. Each rule screens out
        # the other's riskiest calls.
        kit.unit(3, "cRC", {**rd, **c1}, -1.0, "reader's AND caller's rule")
        terms = {"cRC": 1.0}
    s = {**terms, "sure_yaniv": 1.0}
    kit.unit(4, "ysum", {**s, "yaniv": 4.0}, -4.0, "number of reasons to call, Yaniv row only")
    kit.unit(4, "ysum1", {**s, "yaniv": 4.0}, -5.0, "same minus 1")
    kit.unit(5, "ycall", {"ysum": 1.0, "ysum1": -1.0}, 0.0, "1 if any reason to call (capped at 1)")
    kit.out({"yaniv": -200.0, "ycall": 400.0})


# ================================================================== match score
BIG = 200.0  # gate strength of the score units: larger than any value a gated sum takes


def add_score_units(kit: Kit, v: dict) -> None:
    """Match-score terms, copied from scorer.add_score_units (its measured defaults), in this design's points.

    1. open_cut: taking the open card leaves my score + held points exactly 50 or 100, so if the opponent calls
       next turn my score is cut by 50: worth ~50 x P(it calls next) at 50, half that at 100;
       (flat, x threat) = (1.5, 18) at 50 and (1, 9) at 100, threat = clip((15 - opp_est)/6, 0, 1).
    2. deck_cut: the same for a deck draw, weighted by the chance the card drawn is worth exactly what lands
       on the cut (unseen count of that value / 34) x threat.
    3. assaf_cut: Yaniv is legal and my score + my points + 30 is exactly 50 or 100: even a caught call cuts
       my score, so call. +400 on the Yaniv row: the row is -200 + 400 * ycall here, so +400 makes it +200 or
       more, above every throw (throws score within about -60..+50 in this design) - same effect as in scorer.
    """
    # threat: how likely the opponent calls on its next turn (~40% at opp_est <= 9, 14% at 10-14, 3% above)
    kit.unit(1, "threat_a", {"opp_est": -40.0}, 15.0, "relu(15 - opponent's estimated total)")
    kit.unit(1, "threat_b", {"opp_est": -40.0}, 9.0, "(threat_a - threat_b)/6: 1 at <= 9, 0 at >= 15")
    # 1. open card taken lands on the cut (tent of width 1 at 50 and 100)
    s = {"my_score": 120.0, "pts_kept": 40.0, "incoming": 10.0, "take_open": BIG}
    for at, (flat, threat) in {50: (1.5, 18.0), 100: (1.0, 9.0)}.items():
        pieces = (("lo", -(at - 1), 1.0), ("mid", -at, -2.0), ("hi", -(at + 1), 1.0))
        for name, off, wt in pieces:
            kit.unit(1, f"opencut{at}_{name}", s, off - BIG, f"tent piece: score + held after taking the open card == {at}")
            kit.out({f"opencut{at}_{name}": flat * wt})
        w = {f"opencut{at}_{name}": BIG * wt for name, _, wt in pieces}
        w.update({"threat_a": 1.0 / 6, "threat_b": -1.0 / 6})
        kit.unit(2, f"opencut{at}_threat", w, -BIG, f"threat (0..1) when the open-card take lands on {at}")
        kit.out({f"opencut{at}_threat": threat})
    # 2. deck draw lands on the cut if the card is worth need = target - score - kept
    s = {"my_score": 120.0, "pts_kept": 40.0, "draw_deck": BIG}
    for at, weight in ((50, 20.0), (100, 10.0)):
        chance = {}
        for val in range(0, 11):
            pieces = (("lo", -(at - val - 1), 1.0), ("mid", -(at - val), -2.0), ("hi", -(at - val + 1), 1.0))
            for name, off, wt in pieces:
                kit.unit(1, f"deckcut{at}_{val}_{name}", s, off - BIG, f"tent piece: deck draw, score + kept == {at} - {val}")
            w = {f"deckcut{at}_{val}_{name}": BIG * wt for name, _, wt in pieces}
            for r, p in zip(RANK_NAMES, RANK_POINTS):
                if p == val:
                    w[f"unseen_{r}"] = 4.0 / 34.0
            kit.unit(2, f"deckcut{at}_{val}", w, -BIG, f"chance the deck card is worth {val} (when it lands on {at})")
            chance[f"deckcut{at}_{val}"] = 1.0
        # product chance x threat ~ relu(P + 0.45 threat - 0.45) (exact at the corners; P <= ~0.45)
        kit.unit(3, f"deckcut{at}", {**chance, "threat_a": 0.45 / 6, "threat_b": -0.45 / 6}, -0.45,
                 f"~ chance the deck draw lands on {at} x threat")
        kit.out({f"deckcut{at}": weight})
    # 3. a caught call that lands on 50 / 100 still cuts my score: call
    a = {"my_score": 120.0, "my_pts": 40.0, "yaniv": BIG}
    for at in (50, 100):
        for name, off, wt in (("lo", -(at - 31), 1.0), ("mid", -(at - 30), -2.0), ("hi", -(at - 29), 1.0)):
            kit.unit(1, f"assafcut{at}_{name}", a, off - BIG, f"tent piece: Yaniv row and score + points + 30 == {at}")
            kit.out({f"assafcut{at}_{name}": 400.0 * wt})


# ================================================================== build
def build(variant: dict | None = None):
    v = dict(BEST)
    if variant:
        v.update(variant)
    kit = Kit()
    common_units(kit)
    add_throw_units(kit, v)
    add_yaniv_units(kit, v)
    if v["score"]:
        add_score_units(kit, v)
    return kit.build(), kit


if __name__ == "__main__":
    net, kit = build()
    path = Path(SCRATCH) / "nets" / "merge_exact.json"
    save_net(net, path)
    print("saved", path, net.dims, file=sys.stderr)
