"""counter2: a second, stronger card counter. Plain rules, nothing trained.

Like the first counter (heuristic.py) it sees only `engine.View`: its own hand, the public table and the
scores. It remembers nothing between moves.
"""

from __future__ import annotations

from math import exp

from .engine import ASSAF_PENALTY, DECK_SIZE, POINTS, SCORE_CUTS, YANIV_LIMIT, Action, View
from .features import HIDDEN_BETA

RANK_PTS = POINTS + [0]                                   # points of the 14 ranks (the jokers are rank 13)
RANK_OF = [c % 13 for c in range(52)] + [13, 13]
ALL_CARDS = (1 << DECK_SIZE) - 1
CAP = 12                                                  # the opponent's total is modelled exactly up to here

# ---- the opponent's hidden cards
CAP_LEAK = 0.15        # weight of a kept card above the single it last threw
FRESH = 0.75           # chance one hidden card is a fresh deck draw (it took the open card otherwise)
# ---- the chance it calls on its next turn, given that it holds t points (t = 0..7)
CALL_SHARE = (1.0, 0.9, 0.8, 0.7, 0.55, 0.5, 0.5, 0.45)
# ---- what a card still in hand after my next throw costs
SLOT = 2.5
TURN = 2.7
TWO_HEAVY = 0.4
EXPOSE = {5: 0.05, 4: 0.12, 3: 0.25, 2: 0.4, 1: 0.5}
# ---- the value of an empty hand, by the opponent's card count
BASE = {5: 17.0, 4: 15.0, 3: 12.0, 2: 7.0, 1: 2.0}
# ---- feeding
FEED = 2.4
FEED_POINT = 0.5
# ---- calling
CALL_RISK = 1.0        # never call above this risk
CALL_EDGE = -0.5        # call only if it beats the best throw by this much
LATER_RISK = 1.0       # the risk of a call next turn, as a multiple of the same call now

_TABLES: dict = {}


def reset_tables() -> None:
    _TABLES.clear()


def _lean(ply: int) -> list[float]:
    beta = HIDDEN_BETA[min(ply // 2, len(HIDDEN_BETA) - 1)]
    return [exp(-beta * p) for p in range(11)]


def opp_total_cdf(view: View, unseen_pts: list[int]) -> list[float]:
    """cdf[t] = P(the opponent's total <= t) for t = 0..CAP.

    Hidden cards lean low (HIDDEN_BETA). If its last throw was a single worth v, the cards it kept are
    almost surely worth v or less; one hidden card may be a fresh draw from the deck.
    """
    known_pts = 0
    known_n = 0
    bits = view.opp_known
    while bits:
        bit = bits & -bits
        known_pts += RANK_PTS[RANK_OF[bit.bit_length() - 1]]
        known_n += 1
        bits ^= bit
    need = view.opp_size - known_n
    room = CAP - known_pts
    weight = _lean(view.ply)
    discard = view.discard
    capped = False
    if view.ply >= 1:
        top = discard[-1]
        single = len(discard) < 2 or RANK_OF[discard[-2]] != RANK_OF[top]
        if single:
            v = RANK_PTS[RANK_OF[top]]
            weight = [w if p <= v else w * CAP_LEAK for p, w in enumerate(weight)]
            capped = True
    fresh = FRESH if capped else 0.0
    if room < 0:
        return [0.0] * (CAP + 1), weight, fresh
    if need <= 0:
        return [1.0 if t >= known_pts else 0.0 for t in range(CAP + 1)], weight, fresh
    total_unseen = sum(unseen_pts)
    if need > total_unseen:
        return [1.0] * (CAP + 1), weight, fresh
    # ways[k][s]: weighted ways to hold k kept cards worth s in all (s <= room); tot[k]: any total
    ways = [[0.0] * (room + 1) for _ in range(need + 1)]
    ways[0][0] = 1.0
    tot = [0.0] * (need + 1)
    tot[0] = 1.0
    for p in range(11):
        n = unseen_pts[p]
        if not n:
            continue
        w = weight[p]
        coef = [1.0]
        for j in range(1, min(n, need) + 1):
            coef.append(coef[-1] * (n - j + 1) / j * w)
        top_j = len(coef) - 1
        for k in range(need, 0, -1):
            row = ways[k]
            t = tot[k]
            for j in range(1, min(k, top_j) + 1):
                c = coef[j]
                t += c * tot[k - j]
                shift = p * j
                if shift <= room:
                    below = ways[k - j]
                    for s in range(shift, room + 1):
                        row[s] += c * below[s - shift]
            tot[k] = t
    all_kept = [x / tot[need] for x in ways[need]] if tot[need] > 0 else [0.0] * (room + 1)
    if fresh > 0.0:
        kept = [x / tot[need - 1] for x in ways[need - 1]]
        mixed = [0.0] * (room + 1)
        for p in range(min(10, room) + 1):
            share = unseen_pts[p] / total_unseen
            if share:
                for s in range(p, room + 1):
                    mixed[s] += share * kept[s - p]
        dist = [fresh * a + (1.0 - fresh) * b for a, b in zip(mixed, all_kept)]
    else:
        dist = all_kept
    cdf = [0.0] * (CAP + 1)
    run = 0.0
    for s in range(room + 1):
        run += dist[s]
        cdf[known_pts + s] = run
    return cdf, weight, fresh


def _card_cost(opp_size: int) -> list[float]:
    key = ("cost", opp_size, SLOT, TURN, TWO_HEAVY)
    table = _TABLES.get(key)
    if table is None:
        expose = EXPOSE[max(1, min(5, opp_size))]
        by_pts = []
        for p in range(11):
            heavy = 0.0 if p <= 1 else TWO_HEAVY if p == 2 else 1.0
            by_pts.append(SLOT + heavy * TURN + expose * p + (0.15 * p if p <= 2 else 0.0))
        table = [by_pts[RANK_PTS[r]] for r in range(14)]
        _TABLES[key] = table
    return table


def _read(view: View):
    """What I hold and what I have not seen, by rank."""
    hand = view.my_hand
    count = [0] * 14
    my_pts = 0
    bits = hand
    while bits:
        bit = bits & -bits
        r = RANK_OF[bit.bit_length() - 1]
        count[r] += 1
        my_pts += RANK_PTS[r]
        bits ^= bit
    seen = hand | view.opp_known
    for card in view.discard:
        seen |= 1 << card
    unseen = [0] * 14
    unseen_pts = [0] * 11
    bits = ALL_CARDS & ~seen
    while bits:
        bit = bits & -bits
        r = RANK_OF[bit.bit_length() - 1]
        unseen[r] += 1
        unseen_pts[RANK_PTS[r]] += 1
        bits ^= bit
    return count, my_pts, unseen, unseen_pts


def _est(view: View, unseen_pts: list[int]) -> float:
    """The opponent's expected total: its known cards plus its hidden cards at the leaning average."""
    known_pts = 0
    known_n = 0
    bits = view.opp_known
    while bits:
        bit = bits & -bits
        known_pts += RANK_PTS[RANK_OF[bit.bit_length() - 1]]
        known_n += 1
        bits ^= bit
    weight = _lean(view.ply)
    top = sum(weight[p] * p * n for p, n in enumerate(unseen_pts))
    bottom = sum(weight[p] * n for p, n in enumerate(unseen_pts))
    return known_pts + max(0, view.opp_size - known_n) * (top / bottom if bottom else 0.0)


def _call_value(view: View, pts: int, cdf: list[float], est: float, risk_scale: float = 1.0) -> float:
    """My expected result if I call Yaniv holding `pts`."""
    risk = min(1.0, cdf[pts] * risk_scale)
    caught = -float(pts + ASSAF_PENALTY)
    if view.my_score + pts + ASSAF_PENALTY in SCORE_CUTS:
        caught += 50.0
    won = max(est, pts + 3.0)
    if view.opp_score + round(won) in SCORE_CUTS:
        won -= 0.0
    return (1.0 - risk) * won + risk * caught


def beliefs(view: View) -> dict:
    """What counter2 believes about the opponent (for heuristic2_study.py)."""
    count, my_pts, unseen, unseen_pts = _read(view)
    cdf, _, _ = opp_total_cdf(view, unseen_pts)
    hazard = cdf[0] * CALL_SHARE[0] + sum((cdf[t] - cdf[t - 1]) * CALL_SHARE[t] for t in range(1, 8))
    return {"risk": cdf[my_pts] if my_pts <= CAP else 1.0, "hazard": hazard, "cdf": cdf,
            "value": evaluate(view)[0]}


def evaluate(view: View, detail: bool = False):
    """(value of my best move, the move as (rank, cards, draw) or None for Yaniv)."""
    count, my_pts, unseen, unseen_pts = _read(view)
    n_unseen = sum(unseen)
    cdf, kept_weight, fresh = opp_total_cdf(view, unseen_pts)
    est = _est(view, unseen_pts)
    opp_size = max(1, min(5, view.opp_size))

    # ---- its call on its next turn: how likely, and my result by the points I hold then
    pdf = [cdf[0]] + [cdf[t] - cdf[t - 1] for t in range(1, YANIV_LIMIT + 1)]
    calls = [pdf[t] * CALL_SHARE[t] for t in range(YANIV_LIMIT + 1)]
    hazard = sum(calls)
    called = []
    for x in range(51):
        value = 0.0
        for t in range(YANIV_LIMIT + 1):
            value += calls[t] * ((ASSAF_PENALTY + t) if x <= t else -float(x))
        if view.my_score + x in SCORE_CUTS:
            value += 50.0 * (sum(calls[:x]) if x <= YANIV_LIMIT else hazard)
        called.append(value)                              # already times the chance of the call

    # ---- my next turn: call if that is worth more than throwing on
    cost = _card_cost(opp_size)
    base = BASE[opp_size]
    draw_cost = sum(unseen[d] * cost[d] for d in range(14)) / n_unseen if n_unseen else 0.0
    stay = 1.0 - hazard
    later = [_call_value(view, x, cdf, est, LATER_RISK) for x in range(YANIV_LIMIT + 1)]

    open_rank = RANK_OF[view.discard[-1]]
    need = view.opp_size - view.opp_known.bit_count()
    known_ranks = [0] * 14
    bits = view.opp_known
    while bits:
        bit = bits & -bits
        known_ranks[RANK_OF[bit.bit_length() - 1]] += 1
        bits ^= bit

    kept_total = sum(unseen_pts[p] * kept_weight[p] for p in range(11)) or 1.0
    hand_cost = sum(count[r] * cost[r] for r in range(14))
    best_key = None
    best_throw = None
    for r in range(14):
        have = count[r]
        if not have:
            continue
        # the chance the opponent holds this rank: it would take my card and throw the set
        if known_ranks[r]:
            holds = 1.0
        elif need > 0 and unseen[r] and n_unseen > need:
            # each kept card is of this rank with its leaning weight's share; the fresh one as the deck deals
            share = unseen[r] * kept_weight[RANK_PTS[r]] / kept_total
            fresh_share = unseen[r] / n_unseen
            holds = 1.0 - ((1.0 - share) ** (need - fresh)) * ((1.0 - fresh_share) ** fresh)
        else:
            holds = 0.0
        feed = (FEED + FEED_POINT * RANK_PTS[r]) * holds
        for k in range(1, have + 1):
            count[r] = have - k
            kept_pts = my_pts - k * RANK_PTS[r]
            kept_cost = hand_cost - k * cost[r]
            group = [count[x] * cost[x] for x in range(14)]
            best_group = max(group)
            floor = base - draw_cost - kept_cost
            # take the open card
            pts = kept_pts + RANK_PTS[open_rank]
            g = group[open_rank] + cost[open_rank]
            go_on = floor - cost[open_rank] + (g if g > best_group else best_group)
            if pts <= YANIV_LIMIT and later[pts] > go_on:
                go_on = later[pts]
            value_open = called[pts] + stay * go_on - feed
            key = (value_open, 1, r, k)
            if best_key is None or key > best_key:
                best_key, best_throw = key, (r, k, "open")
            if view.stock_size > 0 and n_unseen > 0:
                value = 0.0
                for d in range(14):
                    n = unseen[d]
                    if n:
                        pts = kept_pts + RANK_PTS[d]
                        g = group[d] + cost[d]
                        go_on = floor - cost[d] + (g if g > best_group else best_group)
                        if pts <= YANIV_LIMIT and later[pts] > go_on:
                            go_on = later[pts]
                        value += n * (called[pts] + stay * go_on)
                value = value / n_unseen - feed
                key = (value, 0, r, k)
                if key > best_key:
                    best_key, best_throw = key, (r, k, "stock")
        count[r] = have

    # ---- Yaniv now
    throw_value = best_key[0] if best_key is not None else None
    if my_pts <= YANIV_LIMIT:
        risk = cdf[my_pts]
        call_now = _call_value(view, my_pts, cdf, est)
        if detail:
            return {"risk": risk, "call": call_now, "throw": throw_value, "move": best_throw, "hazard": hazard}
        if (risk <= 0.0 or view.my_score + my_pts + ASSAF_PENALTY in SCORE_CUTS
                or (risk <= CALL_RISK and (throw_value is None or call_now >= throw_value + CALL_EDGE))):
            return call_now, None
    elif detail:
        return {"throw": throw_value, "move": best_throw, "hazard": hazard}
    if best_key is None:
        return 0.0, None
    return throw_value, best_throw


def _action_for(throw, actions: list[Action]) -> Action:
    if throw is None:
        return next((a for a in actions if a.kind == "yaniv"), actions[0])
    r, k, draw = throw
    for action in actions:
        if action.kind == "play" and action.draw == draw and action.discard_mask.bit_count() == k:
            low = action.discard_mask & -action.discard_mask
            if RANK_OF[low.bit_length() - 1] == r:
                return action
    return actions[0]


def choose_counter2(view: View, actions: list[Action], rng=None) -> Action:
    del rng
    return _action_for(evaluate(view)[1], actions)
