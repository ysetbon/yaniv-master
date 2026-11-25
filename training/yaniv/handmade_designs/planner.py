"""planner: a hand-written Yaniv net (4 cards, Yaniv at 7) that values a throw by what it sets up.

Every weight is chosen by reasoning, none is trained. A throw is valued in points (higher is better):

    score = - wH * H - wP * P + B - gifts - 3 * cards kept + 0.15 * pts per extra card of a rank - info

  H  points I hold after the move: kept + the card coming in (the open card's, or the unseen average).
     This is what I pay if the opponent calls before my next turn.
  P  the plan: points I hold after my NEXT best throw (H minus the heaviest rank group of the new hand).
     For a deck draw the unknown card X is either thrown next turn (if it beats my heaviest group S)
     or kept, so P = kept - S + E[min(X, S)], with E[min(X, S)] = sum_t [S > t] P(X > t) from the
     unseen counts. A high pair kept (K K) is "sheddable next turn" and makes S = 20 by itself.
  wH/wP  shift from the plan to the present as the opponent's estimated total falls (it may call first).
  B  the chance-weighted bonus for being at 7 or less next turn (I may call then): exact when the
     open card is taken, from the unseen counts when drawing from the deck.
  gifts  a low card left face up (the opponent takes it instead of a ~5.5 deck card), and a card
     of a rank the opponent is known to hold (it takes it and has a pair to shed).
  cards kept  a smaller hand stays smaller for the rest of the round; a second card of a rank is worth
     0.15 * its points (a high set sheds a lot in one throw; a low pair just clogs two of four slots).
  info  taking the open card shows it; when the opponent's estimate is <= 9, 0.3 per point taken.
  catch  when the opponent's estimate is <= 7, B counts twice (a low hand also catches its call).

Variants and their results: SCRATCH/logs/planner.md.

The Yaniv call is a separate, simple rule (add_yaniv_units).
"""

from __future__ import annotations

import os

os.environ.setdefault("YANIV_HAND_SIZE", "4")
os.environ.setdefault("YANIV_CALL_AT", "7")

from pathlib import Path  # noqa: E402

from training.yaniv.handmade_kit import RANK_NAMES, RANK_POINTS, Kit  # noqa: E402

SCRATCH = "/tmp/claude-0/-home-user-yaniv-master/ecf1221d-8c18-559f-9f3b-91b1597261e5/scratchpad"

# --- reasoning constants -----------------------------------------------------------------------
UNSEEN_N = 43.0        # typical count of unseen cards mid-round (stands in for unseen_n in P(X = rank))
W_H, W_P = 0.25, 0.75  # no danger: the plan matters most, the present a quarter
DANGER = [(11.0, 0.15), (9.0, 0.15), (7.0, 0.15), (5.0, 0.15)]  # opp_est (points) at or under t: move delta of weight P -> H
CALL_BONUS = {7: 3.0, 6: 0.8, 5: 0.8, 4: 0.8, 3: 0.8, 2: 0.8, 1: 0.8, 0: 0.8}  # B(h) = sum_{t >= h} CALL_BONUS[t]
CATCH = 1.0            # when the opponent's estimate is <= 7: the call bonus counts again (low hands catch its call)
GIFT_LOW = 0.3         # per point the face-up card is under the unseen average
GIFT_MATCH = (0.5, 0.25)  # leaving a rank the opponent is known to hold: 0.5 + 0.25 * points (its pair sheds that much more)
CARD_COST = 3.0        # per card kept (no danger)
CARD_RELIEF = 0.0      # card cost removed per danger step (tried 1.0 with CARD_COST 4.5: worse, v13)
INFO_PER_PT = 0.3      # per point of a taken open card when the opponent is near a call (it sees the card)
TAKE_INFO = 0.0        # a flat cost on taking the open card (tried 1.5: worse, v11)
DUP_VALUE = 0.15       # * rank points, per card kept beyond the first of its rank: a high set sheds a lot later;
                       # a low pair only clogs two of four slots


def _cdf(v: int) -> dict[str, float]:
    """P(a deck card is worth v or less), linear in the unseen counts (count = 4 * unseen_r)."""
    return {f"unseen_{r}": 4.0 / UNSEEN_N for r, p in zip(RANK_NAMES, RANK_POINTS) if p <= v}


def _add(*terms: dict[str, float], scale: list[float] | None = None) -> dict[str, float]:
    out: dict[str, float] = {}
    for i, t in enumerate(terms):
        k = 1.0 if scale is None else scale[i]
        for s, w in t.items():
            out[s] = out.get(s, 0.0) + k * w
    return out


def _max_tree(kit: Kit, exprs: list[dict[str, float]], layer: int, tag: str) -> tuple[dict[str, float], int]:
    """max of nonnegative linear expressions: max(a, b) = relu(a - b) + b, one layer per round."""
    level = 0
    while len(exprs) > 1:
        nxt = []
        for i in range(0, len(exprs) - 1, 2):
            a, b = exprs[i], exprs[i + 1]
            d = kit.unit(layer, f"{tag}_d{level}_{i}", _add(a, b, scale=[1.0, -1.0]), note="max: how far a is over b")
            c = kit.unit(layer, f"{tag}_c{level}_{i}", dict(b), note="max: b carried")
            nxt.append({d: 1.0, c: 1.0})
        if len(exprs) % 2:
            c = kit.unit(layer, f"{tag}_c{level}_last", dict(exprs[-1]), note="max: odd one carried")
            nxt.append({c: 1.0})
        exprs = nxt
        layer += 1
        level += 1
    return exprs[0], layer


def add_throw_units(kit: Kit) -> None:
    H = {"pts_kept": 40.0, "incoming": 10.0}         # points held after the move
    play = {"draw_deck": 1.0, "take_open": 1.0}       # 1 on every throw, 0 on the Yaniv row

    # Layer 1: what this move throws and takes, rank by rank; integer flags for thresholds.
    for r in RANK_NAMES:
        kit.unit(1, f"thr_{r}", {"n_thrown": 4.0, f"throw_{r}": 4.0}, -4.0, note=f"cards of rank {r} thrown")
        kit.unit(1, f"tk_{r}", {"take_open": 1.0, f"open_{r}": 1.0}, -1.0, note=f"1 if an open {r} is taken")
    for j in range(8):  # kept points k <= j  (deck draw: callable chance per kept total)
        kit.unit(1, f"k_le{j}_a", {"pts_kept": -40.0}, j + 1.0)
        kit.unit(1, f"k_le{j}_b", {"pts_kept": -40.0}, float(j))
    for t in range(8):  # held points h <= t (open card taken: callable for sure)
        kit.unit(1, f"h_le{t}_a", _add(H, scale=[-1.0]), t + 1.0)
        kit.unit(1, f"h_le{t}_b", _add(H, scale=[-1.0]), float(t))
    for t, _ in DANGER:  # the opponent's estimated total at or under t
        kit.unit(1, f"dz{int(t)}_a", {"opp_est": -400.0}, 10.0 * t + 1.0)
        kit.unit(1, f"dz{int(t)}_b", {"opp_est": -400.0}, 10.0 * t)
    kit.unit(1, "deck_in", {"incoming": 10.0, "draw_deck": 10.0}, -10.0, note="the unseen average, on deck draws only")
    kit.unit(1, "gift_low", {"unseen_avg": 10.0, "top_pts": -10.0, "draw_deck": 10.0, "take_open": 10.0}, -10.0,
             note="points the face-up card saves the opponent against a deck draw")
    for r in RANK_NAMES:
        kit.unit(1, f"gift_{r}", {f"throw_{r}": 2.0, f"known_{r}": 4.0}, -2.0,
                 note=f"throwing {r} while the opponent is known to hold one")

    # Layer 2: rank groups after the move (gated to throws), gated callable bonuses, danger copies of H.
    groups = []
    for r, p in zip(RANK_NAMES, RANK_POINTS):
        if p == 0:
            continue
        g = kit.unit(2, f"grp_{r}", {f"hand_{r}": 4.0 * p, f"thr_{r}": -p, f"tk_{r}": p, "draw_deck": 40.0,
                                      "take_open": 40.0}, -40.0, note=f"points of my {r}s after the move")
        groups.append({g: 1.0})
    for r, p in zip(RANK_NAMES, RANK_POINTS):
        cnt = {f"hand_{r}": 4.0, f"thr_{r}": -1.0, f"tk_{r}": 1.0, "draw_deck": 4.0, "take_open": 4.0}
        kit.unit(2, f"cnt_{r}", cnt, -4.0, note=f"my {r}s after the move")
        kit.unit(2, f"dup_{r}", cnt, -5.0, note=f"my {r}s after the move beyond the first: a set to shed together later")
        kit.unit(3, f"pairX_{r}", {f"unseen_{r}": 4.0 / UNSEEN_N, f"cnt_{r}": 1.0, f"dup_{r}": -1.0, "draw_deck": 1.0},
                 -2.0, note=f"deck draw: chance the drawn card pairs a kept {r}")
    F = []
    for j in range(9):
        F.append(_add(*[_cdf(t - j) for t in range(j, 8)], scale=[CALL_BONUS[t] for t in range(j, 8)]) if j < 8 else {})
    for j in range(8):
        diff = _add(F[j], F[j + 1], scale=[1.0, -1.0])
        kit.unit(2, f"callD_{j}", _add(diff, {f"k_le{j}_a": 20.0, f"k_le{j}_b": -20.0, "draw_deck": 20.0}),
                 -40.0, note=f"deck draw, kept <= {j}: added chance-weighted call bonus")
    for t in range(8):
        kit.unit(2, f"callT_{t}", {f"h_le{t}_a": 1.0, f"h_le{t}_b": -1.0, "take_open": 1.0}, -1.0,
                 note=f"open card taken, holding <= {t}")
    for t, _ in DANGER:
        kit.unit(2, f"dH{int(t)}", _add(H, {f"dz{int(t)}_a": 60.0, f"dz{int(t)}_b": -60.0}), -60.0,
                 note=f"H when the opponent's estimate is <= {t}")

    # Layers 3-6: S = the heaviest rank group of the new hand.
    S, layer = _max_tree(kit, groups, 3, "S")
    # Layer 7-8: E[min(X, S)] for deck draws.
    for t in range(10):
        kit.unit(layer, f"Sgt{t}_a", S, -float(t))
        kit.unit(layer, f"Sgt{t}_b", S, -float(t) - 1.0)
    mins = {}
    for t in range(10):
        m = kit.unit(layer + 1, f"minXS_{t}",
                     _add(_cdf(t), {f"Sgt{t}_a": 1.0, f"Sgt{t}_b": -1.0, "draw_deck": 1.0}, scale=[-1.0, 1.0]),
                     -1.0, note=f"deck draw: P(X > {t}) if S > {t}")
        mins[m] = 1.0
    # P = H - S - (deck: unseen avg) + E[min(X, S)]
    P = _add(H, S, {"deck_in": 1.0}, mins, scale=[1.0, -1.0, -1.0, 1.0])
    for t, _ in DANGER:
        kit.unit(layer + 2, f"dP{int(t)}", _add(P, {f"dz{int(t)}_a": 60.0, f"dz{int(t)}_b": -60.0}), -60.0,
                 note=f"P when the opponent's estimate is <= {t}")
    kit.unit(layer + 2, "plan", P, note="points after my next best throw")

    out = {"plan": -W_P, "n_kept": -CARD_COST * 5.0, "gift_low": -GIFT_LOW, "take_open": -TAKE_INFO}
    for r, p in zip(RANK_NAMES, RANK_POINTS):
        out[f"dup_{r}"] = DUP_VALUE * p
        out[f"pairX_{r}"] = DUP_VALUE * p + 0.5 * p  # a drawn pair also sheds more next turn

    out = _add(out, H, scale=[1.0, -W_H])
    for t, _ in DANGER:  # a smaller hand pays off over future turns; when the round is about to end it does not
        kit.unit(2, f"dN{int(t)}", {"n_kept": 5.0, f"dz{int(t)}_a": 5.0, f"dz{int(t)}_b": -5.0}, -5.0,
                 note=f"cards kept when the opponent's estimate is <= {t}")
        out[f"dN{int(t)}"] = CARD_RELIEF
    # Taking the open card shows it: an opponent near a call learns my minimum rises by its points and calls safely.
    kit.unit(2, "shown_pts", {"incoming": 10.0, "take_open": 10.0, "dz9_a": 10.0, "dz9_b": -10.0}, -20.0,
             note="points of the open card I take, when the opponent's estimate is <= 9")
    out["shown_pts"] = -INFO_PER_PT
    for t, delta in DANGER:
        out = _add(out, {f"dH{int(t)}": -delta, f"dP{int(t)}": delta})
    for j in range(8):
        out[f"callD_{j}"] = 1.0
    for t in range(8):
        out[f"callT_{t}"] = CALL_BONUS[t]
    for j in range(8):  # the opponent likely calls next: a low hand also catches its call (Assaf, ~40 point swing)
        kit.unit(3, f"catchD_{j}", {f"callD_{j}": 1.0, "dz7_a": 20.0, "dz7_b": -20.0}, -20.0)
        kit.unit(3, f"catchT_{j}", {f"callT_{j}": 1.0, "dz7_a": 1.0, "dz7_b": -1.0}, -1.0)
        out[f"catchD_{j}"] = CATCH
        out[f"catchT_{j}"] = CATCH * CALL_BONUS[j]
    for r, p in zip(RANK_NAMES, RANK_POINTS):
        out[f"gift_{r}"] = -(GIFT_MATCH[0] + GIFT_MATCH[1] * p)
    kit.out(out)


def add_yaniv_units(kit: Kit) -> None:
    """Call when the uniform Assaf chance is under a convex threshold of my points; always when sure.

    Calling with m points wins the opponent's hand (~10) or loses m + 30; the lower m, the more of the
    opponent's hands lose to me and the cheaper a catch is, so the tolerated risk grows faster as m falls:
    thr(m) = 0.04 at 7, .06, .09, .13, .18, .25, .33, 0.45 at 0.
    """
    kinks = {6: 0.01, 5: 0.01, 4: 0.01, 3: 0.02, 2: 0.01, 1: 0.04}  # extra slope below each kink
    for k in kinks:
        kit.unit(1, f"m_under{k}", {"my_pts": -40.0}, float(k), note=f"how far my points are under {k}")
    thr = _add({"my_pts": -0.02 * 40.0}, {f"m_under{k}": w for k, w in kinks.items()})
    x = _add(thr, {"assaf": -1.0}, scale=[100.0, 100.0])
    base = 100.0 * (0.04 + 0.02 * 7)
    kit.unit(2, "call_a", x | {"yaniv": 40.0}, base - 40.0)
    kit.unit(2, "call_b", x | {"yaniv": 40.0}, base - 40.0 - 1.0)
    kit.out({"call_a": 200.0, "call_b": -200.0, "yaniv": -100.0})


def build():
    kit = Kit()
    add_throw_units(kit)
    add_yaniv_units(kit)
    return kit.build(), kit


if __name__ == "__main__":
    from training.yaniv.network import save_net

    net, kit = build()
    save_net(net, Path(SCRATCH) / "nets" / "planner.json")
    print(net.dims)
