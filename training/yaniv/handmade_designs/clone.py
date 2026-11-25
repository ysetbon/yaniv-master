"""`clone`: a hand-written net that rebuilds the card counter's own evaluation, then improves on it.

The counter (heuristic.py) throws the move minimising
    E_draw[ hand_badness(new hand) ] + top_penalty(top card)
with hand_badness = points - (1 - w) * maxgroup, where maxgroup is the heaviest single-rank group
(points of the best next throw) and w depends on the opponent's card count. It calls Yaniv when the
call is sure, or when the Assaf risk is under a threshold that falls with its points.

How each piece is written with ReLUs (every quantity in whole points / whole cards):
  kept count of rank r   k_r = 4*hand_r - 4*relu(n_thrown + throw_r - 1)
  kept group of rank r   G_r = p_r * k_r                       (layer 2)
  M = max_r G_r          tournament: max(a, b) = relu(a - b) + relu(b)   (layers 3-6)
  deck draw:  E[maxgroup] = M + T / n,  n = unseen cards,
     T = sum_t 1[M < t] * C(t)                    C(t) = unseen cards worth >= t points
       + sum_s u_s * (relu(p_s + G_s - M) - relu(p_s - M))   (drawing a card that grows a kept group)
     products of a 0/1 gate and a bounded value are relu(x - B*(1 - g)); 1/n is exact by gating T/k
     on the one-hot of the integer n (hat functions of relu(n - k)).
  open card:  maxgroup = M + relu(p_open + G_open - M)
  (1 - w) * E[maxgroup] by gating on opp_has_N; points part = 40*pts_kept + 10*incoming (exact mean).
  top penalty is linear in the thrown rank (the top card has the thrown rank).
  Yaniv: thr(my_pts) is piecewise linear in my_pts; call = step(thr - assaf) OR sure_yaniv,
     the Yaniv row gets -200 + 400*call so it beats every throw exactly when the counter would call.

build({}) is the pure clone (parity with the counter: official margin +0.27, 100% same throws up to exact
ties, Yaniv differences only from the counter's sampled Assaf risk). build() is BEST, the clone plus:
  - w (weight of current points) by the opponent's lean-low estimated total opp_est instead of its card
    count: w = P(opponent calls before my next turn) + 0.05, measured 0.55/0.30/0.18/0.08/0.04/0.01 for
    opp_est < 5, 5-8, 8-11, 11-14, 14-17, 17+ points;
  - catch bonus 30 * pi_b * P(my hand after the draw <= m_b): when the opponent calls and I am at or
    under its total it is Assaf for them (one-hot of kept points x one-hot of n gives P exactly);
  - feeding penalty (1 + 0.5 * points) for leaving face up a rank the opponent is known to hold;
  - Yaniv: call if sure, or assaf <= 0.5 thr(pts), or (assaf <= 1.75 thr(pts) and opp_est - pts >= 3),
    since the true risk is ~3x the uniform assaf, but only ~1.5-2x when the opponent looks 3+ points higher.
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

# Counter's constants
W_BY_OPP = {1: 0.62, 2: 0.62, 3: 0.40, 4: 0.22, 5: 0.08}
TOP_PEN = [1.8 if p <= 1 else 1.0 if p <= 3 else 0.35 if p <= 5 else 0.0 for p in RANK_POINTS]
# Call threshold by points 0..7: 0.40,0.40,0.28,0.18,0.12,0.08,0.06,0.04 -> slope changes at 1..5
THR = {"bias": 0.40, 1: -0.12, 2: 0.02, 3: 0.04, 4: 0.02, 5: 0.02}

DEFAULT = dict(
    name="clone",
    thr=THR,            # call threshold curve
    sure_call=True,     # always call a sure Yaniv
    w=W_BY_OPP,         # weight of current points by opponent card count
    extra=None,         # extra linear output terms {input: weight}, for reasoned tweaks
    callB=0.0,          # bonus for the chance of holding 7 or less after the draw (may call next turn)
)

BEST = dict(
    est={"cuts": [5, 8, 11, 14, 17], "w": [0.60, 0.35, 0.23, 0.13, 0.09, 0.06],
         "pi": [0.55, 0.30, 0.18, 0.08, 0.04, 0.01], "m": [3, 5, 5, 6, 6, 6], "catch": 30},
    feed=[1.0, 0.5],
    call2={"A": 1.75, "B": 0.5, "d": 3},
)


def build(variant: dict | None = None):
    if variant is None:
        variant = BEST
    v = dict(DEFAULT)
    if variant:
        v.update(variant)
    kit = Kit()
    hand = [f"hand_{RANK_NAMES[r]}" for r in range(14)]
    throw = [f"throw_{RANK_NAMES[r]}" for r in range(14)]
    unseen = [f"unseen_{RANK_NAMES[r]}" for r in range(14)]
    opn = [f"open_{RANK_NAMES[r]}" for r in range(14)]

    # ---------------- layer 1
    for r in R13:
        kit.unit(1, f"tk{r}", {"n_thrown": 1.0, throw[r]: 1.0}, -1.0, "cards of rank r thrown / 4")
    for k in range(0, 52):
        kit.unit(1, f"rn{k}", {"unseen_n": 40.0}, -float(k), "relu(n - k), n = unseen cards")
    for s in R13:
        for j in range(4):
            kit.unit(1, f"us{s}_{j}", {unseen[s]: 4.0}, -float(j), "relu(unseen_s - j)")
    for c in range(1, 6):
        kit.unit(1, f"pr{c}", {"my_pts": 40.0}, -float(c), "relu(my points - c)")

    # ---------------- layer 2
    for r in R13:
        kit.unit(2, f"G{r}", {hand[r]: 4.0 * P[r], f"tk{r}": -4.0 * P[r]}, 0.0, "points of rank r kept")
    for k in range(1, 51):
        kit.unit(2, f"ni{k}", {f"rn{k-1}": 1.0, f"rn{k}": -2.0, f"rn{k+1}": 1.0}, 0.0, "1 if n == k")
    for s in R13:
        for j in range(1, 4):
            kit.unit(2, f"ui{s}_{j}", {f"us{s}_{j-1}": 1.0, f"us{s}_{j}": -1.0}, 0.0, "1 if unseen_s >= j")
    thr = {(k if k == "bias" else int(k)): x for k, x in v["thr"].items()}
    K = 2000.0

    def assaf_step(name, scale):
        """Two layer-2 units whose difference is 1 if assaf <= scale * thr(my_pts), else 0."""
        tw = {f"pr{c}": K * scale * thr[c] for c in range(1, 6)}
        tw["assaf"] = -K
        kit.unit(2, name + "_p", tw, K * scale * thr["bias"] + 1.0, "ramp: scaled thr - assaf")
        kit.unit(2, name + "_n", dict(tw), K * scale * thr["bias"], "ramp shifted")
        return {name + "_p": 1.0, name + "_n": -1.0}

    sure = 1.0 if v["sure_call"] else 0.0
    if not v.get("call2"):
        c = assaf_step("c", 1.0)
        kit.unit(3, "ycall", {**c, "sure_yaniv": sure, "yaniv": 2.0}, -2.0,
                 "the Yaniv row, when the counter would call")
    else:
        c2 = v["call2"]  # call if sure, or assaf <= B*thr, or (assaf <= A*thr and opp_est - my_pts >= d)
        ca = assaf_step("ca", c2["A"])
        cb = assaf_step("cb", c2["B"])
        kit.unit(2, "dg_p", {"opp_est": 40.0 * 10, "my_pts": -40.0 * 10}, -10.0 * c2["d"] + 1.0, "ramp: est - pts - d")
        kit.unit(2, "dg_n", {"opp_est": 40.0 * 10, "my_pts": -40.0 * 10}, -10.0 * c2["d"], "ramp shifted")
        kit.unit(3, "cAD", {**ca, "dg_p": 1.0, "dg_n": -1.0}, -1.0, "loose threshold and opponent looks high")
        kit.unit(3, "cB", cb, 0.0, "strict threshold")
        kit.unit(4, "ycall", {"cAD": 1.0, "cB": 1.0, "sure_yaniv": sure, "yaniv": 2.0}, -2.0,
                 "the Yaniv row, when calling is right")

    # ---------------- layer 3: open-card group, tournament level 1
    for s in R13:
        kit.unit(3, f"go{s}", {f"G{s}": 1.0, opn[s]: 40.0}, -40.0, "kept points of the open card's rank")

    def tourney(layer, vals, tag):
        """vals: list of dicts (linear combos of previous-layer units). Returns next combos."""
        out = []
        for i in range(0, len(vals) - 1, 2):
            a, b = vals[i], vals[i + 1]
            d = {**{k: w for k, w in a.items()}}
            for k, w in b.items():
                d[k] = d.get(k, 0.0) - w
            kit.unit(layer, f"{tag}d{i}", d, 0.0, "relu(a - b)")
            kit.unit(layer, f"{tag}c{i}", dict(b), 0.0, "b")
            out.append({f"{tag}d{i}": 1.0, f"{tag}c{i}": 1.0})
        if len(vals) % 2:
            kit.unit(layer, f"{tag}x", dict(vals[-1]), 0.0, "carried")
            out.append({f"{tag}x": 1.0})
        return out

    vals = [{f"G{r}": 1.0} for r in R13]
    layer = 3
    while len(vals) > 1:
        vals = tourney(layer, vals, f"t{layer}")
        layer += 1
    Mlin = vals[0]  # max kept group, linear in layer-6 units
    L = layer  # 7

    # ---------------- layer 7
    kit.unit(L, "M", Mlin, 0.0, "heaviest kept group")
    for t in range(0, 11):
        kit.unit(L, f"a{t}", {**{k: -w for k, w in Mlin.items()}}, float(t), "relu(t - M)")
    for s in R13:
        kit.unit(L, f"e{s}", {**{k: -w for k, w in Mlin.items()}, f"G{s}": 1.0}, float(P[s]),
                 "relu(p_s + G_s - M): group if a rank-s card arrives, above M")
    kit.unit(L, "ov", {**{k: -w for k, w in Mlin.items()}, "open_pts": 10.0,
                       **{f"go{s}": 1.0 for s in R13}}, 0.0, "open card's group above M")

    # ---------------- layer 8
    for t in range(1, 11):
        ranks = [r for r in R13 if P[r] >= t]
        w = {unseen[r]: 4.0 for r in ranks}
        w[f"a{t}"] = 60.0
        w[f"a{t-1}"] = -60.0
        kit.unit(L + 1, f"h{t}", w, -60.0, "C(t) if M < t: unseen cards worth >= t")
    for s in R13:
        for j in range(1, 4):
            kit.unit(L + 1, f"q{s}_{j}", {f"e{s}": 1.0, f"a{P[s]}": -1.0, f"ui{s}_{j}": 40.0}, -40.0,
                     "gain from drawing the j-th unseen card of kept rank s")
    kit.unit(L + 1, "op", {"ov": 1.0, "take_open": 100.0}, -100.0, "open-card gain, if taken")

    # ---------------- layer 9: divide T by n
    Tw = {f"h{t}": 1.0 for t in range(1, 11)}
    Tw.update({f"q{s}_{j}": 1.0 for s in R13 for j in range(1, 4)})
    for k in range(1, 51):
        w = {name: x / k for name, x in Tw.items()}
        w[f"ni{k}"] = 1000.0
        w["draw_deck"] = 1000.0
        kit.unit(L + 2, f"v{k}", w, -2000.0, "T / n if n == k and a deck draw (T/k < 1000)")

    # ---------------- layer 10: (1 - w) * E[maxgroup]
    Ew = {"M": 1.0, "op": 1.0, **{f"v{k}": 1.0 for k in range(1, 51)}}
    out = {}
    if not v.get("est"):
        wmix = {int(n): x for n, x in v["w"].items()}
        for N in range(1, 6):
            kit.unit(L + 3, f"z{N}", {**Ew, f"opp_has_{N}": 100.0}, -100.0, "E[maxgroup] if opp has N cards")
            out[f"z{N}"] = 1.0 - wmix[N]
    else:
        # w from the opponent's lean-low estimated total (opp_est): the chance it calls before my next turn
        est = dict(v["est"])
        inds = est_buckets(kit, est["cuts"])
        if est.get("pi2"):
            # buckets of (opp_est, opponent's card count): pi2[b][N-1] = chance it calls before my next turn
            cells, ws, pis, ms = [], [], [], []
            for b, ind in enumerate(inds):
                for N in range(1, 6):
                    kit.unit(4, f"ebc{b}_{N}", {ind: 1.0, f"opp_has_{N}": 1.0}, -1.0, "opp_est bucket b and N cards")
                    cells.append(f"ebc{b}_{N}")
                    pis.append(est["pi2"][b][N - 1])
                    ws.append(est["pi2"][b][N - 1] + est.get("wplus", 0.05))
                    ms.append(est["m"][b])
            inds, est["w"], est["pi"], est["m"] = cells, ws, pis, ms
        for b, ind in enumerate(inds):
            kit.unit(L + 3, f"zb{b}", {**Ew, ind: 100.0}, -100.0, "E[maxgroup] if opp_est in bucket b")
            out[f"zb{b}"] = 1.0 - est["w"][b]
        if est.get("catch"):
            # bonus * pi_b * P(my hand after the draw <= m_b): catching the opponent's call (Assaf for them)
            for b, ind in enumerate(inds):
                if est["pi"][b] <= 0:
                    continue
                pm = p_le_units(kit, est["m"][b])
                kit.unit(5, f"cat{b}", {**pm, ind: 1.0}, -1.0, "P(hand <= m_b) if opp_est in bucket b")
                out[f"cat{b}"] = est["catch"] * est["pi"][b]
    eps = 1e-3
    out["pts_kept"] = -40.0 * (1 + eps)
    out["incoming"] = -10.0 * (1 + eps)
    for r in range(14):
        if TOP_PEN[r]:
            out[throw[r]] = -TOP_PEN[r]
    out["yaniv"] = -200.0
    out["ycall"] = 400.0
    kit.out(out)
    if v["extra"]:
        kit.out(v["extra"])
    if v.get("callB"):
        kit.out({name: v["callB"] * w for name, w in p_le_units(kit, 7).items()})
    if v.get("feed"):
        # leaving face up a rank the opponent is known to hold hands it a set
        known = [f"known_{RANK_NAMES[r]}" for r in range(14)]
        for r in range(14):
            kit.unit(1, f"feed{r}", {throw[r]: 1.0, known[r]: 4.0}, -1.0, "top rank the opponent holds")
            kit.out({f"feed{r}": -(v["feed"][0] + v["feed"][1] * P[r])})
    if v.get("stocklean"):
        # The stock is the unseen pool minus the opponent's hidden cards, which lean low, so a deck card is worth
        # about X / (n - h) more than the unseen average: X = h * avg - (opp_est - known_pts), h hidden cards.
        for j in range(-1, 6):
            kit.unit(1, f"hr{j}", {"opp_cards": 5.0, "known_n": -5.0}, -float(j), "relu(h - j)")
        for j in range(1, 5):
            kit.unit(2, f"hh{j}", {f"hr{j-1}": 1.0, f"hr{j}": -2.0, f"hr{j+1}": 1.0}, 0.0, "1 if h == j")
            kit.unit(3, f"hav{j}", {"unseen_avg": 10.0 * j, f"hh{j}": 100.0}, -100.0, "h * unseen average")
        kit.unit(4, "slX", {**{f"hav{j}": 1.0 for j in range(1, 5)}, "opp_est": -40.0, "known_pts": 40.0}, 0.0,
                 "points the hidden cards fall short of the unseen average")
        for k in range(5, 51):
            kit.unit(5, f"sl{k}", {"slX": 1.0 / (k - 3), f"ni{k}": 100.0, "draw_deck": 100.0}, -200.0,
                     "extra points of a deck card, if n == k (h taken as 3)")
            kit.out({f"sl{k}": -v["stocklean"]})
    return kit.build(), kit


def est_buckets(kit: Kit, cuts: list[float]) -> list[str]:
    """One-hot (layer 3) of the bucket of opp_est (in points) between the cuts."""
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
    unseen = [f"unseen_{RANK_NAMES[r]}" for r in range(14)]
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
        low = {unseen[r]: 4.0 for r in range(14) if RANK_POINTS[r] <= m - k}
        kit.unit(3, f"{tag}lc{k}", {**low, f"kh{k}": 60.0}, -60.0, "unseen cards that keep me at m or less")
    for n in range(1, 51):
        w = {f"{tag}lc{k}": 1.0 / n for k in range(m + 1)}
        w[f"ni{n}"] = 100.0
        w["draw_deck"] = 100.0
        kit.unit(4, f"{tag}pc{n}", w, -200.0, "P(hand <= m) for a deck draw, if n == unseen count")
    return {**{f"{tag}pc{n}": 1.0 for n in range(1, 51)}, f"{tag}oc": 1.0}


if __name__ == "__main__":
    net, kit = build()
    path = Path(SCRATCH) / "nets" / "clone.json"
    save_net(net, path)
    print("saved", path, net.dims, file=sys.stderr)
