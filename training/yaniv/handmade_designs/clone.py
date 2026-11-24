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

Variants (VARIANT dict) add reasoned changes on top of the clone; see the log.
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


def build(variant: dict | None = None):
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
    for N in range(1, 6):
        kit.unit(L + 3, f"z{N}", {**Ew, f"opp_has_{N}": 100.0}, -100.0, "E[maxgroup] if opp has N cards")

    wmix = {int(n): x for n, x in v["w"].items()}
    out = {f"z{N}": 1.0 - wmix[N] for N in range(1, 6)}
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
        callable_units(kit, v["callB"])
    return kit.build(), kit


def callable_units(kit: Kit, bonus: float) -> None:
    """bonus * P(the hand after the draw is 7 or less, so I may call Yaniv next turn).

    K = points kept (integer). Deck: P = #(unseen worth <= 7 - K) / n, one-hot on K and on n.
    Open card: 1 if K + open points <= 7.
    """
    unseen = [f"unseen_{RANK_NAMES[r]}" for r in range(14)]
    for j in range(-1, 9):
        kit.unit(1, f"kr{j}", {"pts_kept": 40.0}, -float(j), "relu(K - j)")
    kit.unit(1, "osum8", {"pts_kept": -40.0, "open_pts": -10.0}, 8.0, "relu(8 - K - open)")
    kit.unit(1, "osum7", {"pts_kept": -40.0, "open_pts": -10.0}, 7.0, "relu(7 - K - open)")
    for k in range(0, 8):
        kit.unit(2, f"kh{k}", {f"kr{k-1}": 1.0, f"kr{k}": -2.0, f"kr{k+1}": 1.0}, 0.0, "1 if K == k")
    kit.unit(2, "ocall", {"osum8": 1.0, "osum7": -1.0, "take_open": 1.0}, -1.0,
             "takes the open card and lands on 7 or less")
    for k in range(0, 8):
        low = {unseen[r]: 4.0 for r in range(14) if RANK_POINTS[r] <= 7 - k}
        kit.unit(3, f"lowc{k}", {**low, f"kh{k}": 60.0}, -60.0, "unseen cards that keep me at 7 or less")
    for n in range(1, 51):
        w = {f"lowc{k}": 1.0 / n for k in range(8)}
        w[f"ni{n}"] = 100.0
        w["draw_deck"] = 100.0
        kit.unit(4, f"pc{n}", w, -200.0, "P(callable) for a deck draw, if n == unseen count")
    kit.out({**{f"pc{n}": bonus for n in range(1, 51)}, "ocall": bonus})


if __name__ == "__main__":
    net, kit = build()
    path = Path(SCRATCH) / "nets" / "clone.json"
    save_net(net, path)
    print("saved", path, net.dims, file=sys.stderr)
