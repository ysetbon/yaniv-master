"""counter2: a second, much stronger card counter. Plain rules, nothing trained or fitted.

Like the first counter (heuristic.py, the reference every earlier result was measured against) it acts on
`engine.View` alone: its own hand, the public table and the scores. It remembers nothing between moves and
draws no random numbers, so the same view always gives the same move.

How it decides (every number is reasoned or read off a measured table; results/counter2.md has the tests):

1. THE OPPONENT'S TOTAL. An exact distribution of the opponent's points up to 12, over every way to fill its
   hidden cards from the cards I have not seen. Hidden cards lean low (features.HIDDEN_BETA), and the cards it
   kept are capped by the single it last threw: a player throws its highest card, so what stayed is lower.
2. ITS CALL. The chance it calls Yaniv on its next turn = P(it holds t) x P(it calls on t | the cards I hold).
   The second factor is a measured table: against five cards a player calls whenever it may, against one or
   two cards only on a very low total. If it calls while I hold no more than it, I catch it (an Assaf).
3. A THROW is worth the exact expectation, over the card I draw, of
     its call next turn (my points lost, or the catch; a score landing on 50 or 100 is cut by 50)
     + otherwise my next turn: the better of calling Yaniv then and throwing on.
   "Throwing on" is read from a table of the game played alone against a clock, solved exactly for every
   hand of up to five cards (`_clock_tables`): many moves ahead, in a simplified game.
4. FEEDING. Throwing a rank the opponent holds lets it take the card and throw a set; keeping a rank it is
   about to throw lets me do the same. Both are charged by the chance and by the card's size.
5. YANIV now, if the call is worth at least what the best throw is worth.
"""

from __future__ import annotations

from math import exp

from .engine import ASSAF_PENALTY, DECK_SIZE, POINTS, SCORE_CUTS, YANIV_LIMIT, Action, View
from .features import HIDDEN_BETA

RANK_PTS = POINTS + [0]                                   # points of the 14 ranks (the jokers are rank 13)
RANK_OF = [c % 13 for c in range(52)] + [13, 13]          # suits never matter: every rule is by rank
ALL_CARDS = (1 << DECK_SIZE) - 1
CAP = 12                                                  # the opponent's total is modelled exactly up to here
BY_POINTS = [[r for r in range(14) if RANK_PTS[r] == p] for p in range(10, -1, -1)]   # ranks, highest points first

# ---- 1. the opponent's hidden cards
# Weight of a kept card worth more than the single it last threw. Not zero: it may have kept a pair, or thrown
# a lower card on purpose. With 0.15 the modelled catch risk matched the true one bin by bin (calib-v1).
CAP_LEAK = 0.15
# After a single throw one hidden card is usually the card it just drew from the deck (any unseen card), unless
# it took the open card, which I cannot tell without memory: about three draws in four are from the deck.
FRESH = 0.75

# ---- 2. the chance it calls when it holds t points (t = 0..7), by the cards I hold (1..5)
# Measured on 20,000 rounds each against the old counter, the trained net and the hand-written net
# (calib-v4-callshare.txt) and set between them: all three call nearly always against 4-5 cards and nearly
# never above 3-4 points against 1-2 cards, because a small hand is likely to be at or under their total.
CALL_SHARE = {1: (1.0, 0.65, 0.4, 0.17, 0.06, 0.01, 0.0, 0.0),
              2: (1.0, 1.0, 0.75, 0.5, 0.25, 0.09, 0.03, 0.02),
              3: (1.0, 1.0, 0.95, 0.9, 0.8, 0.73, 0.6, 0.4),
              4: (1.0, 1.0, 1.0, 1.0, 1.0, 0.98, 0.96, 0.95),
              5: (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)}

# ---- 3. the game played alone against a clock (the look-ahead table)
# P(the opponent holds 7 or less) at each step; the clock moves one step each turn. The first steps are the
# measured average by turn number (hazard-by-turn.txt, times about 1.3 for "can call" against "calls"); the
# upper steps carry a position that is already dangerous on to near certainty.
LADDER = (0.0, 0.006, 0.03, 0.07, 0.12, 0.17, 0.22, 0.27, 0.33, 0.4, 0.5, 0.62, 0.76, 0.9)
# Of its hands worth 7 or less, the share worth t or less (t = 0..7): low totals are rare (measure.md c).
CLOCK_LOW = (0.01, 0.03, 0.08, 0.17, 0.30, 0.47, 0.70, 1.0)
# What it pays when my call stands, at the first step and at the last: its hand shrinks through the round
# (measure.md c: the other hand's mean total is 16 when the caller still holds 5 cards, 9-10 near the end).
CLOCK_PAYS = (16.0, 10.0)
# Where the clock stands at my next turn: P(it holds <= 7 then) = 0.04 + 1.1 x P(it does now), measured
# (calib-v6-next-turn.txt: 0.06 -> 0.10, 0.15 -> 0.19, 0.27 -> 0.32, 0.42 -> 0.54).
SECOND_TURN = (0.04, 1.1)

# ---- 4. feeding
# If the opponent holds the rank I throw it takes the card (the counter does so 87% of the time) and throws
# the set: one card and the card's points fewer for it. An opponent card is worth about 2.5 points of result
# and an opponent point about 0.6 (measure.md f2): charge (2.4 + 0.5 x points) x P(it holds the rank).
FEED = 2.4
FEED_POINT = 0.5
# The same gain is mine when its next throw pairs a card I kept, so a rank it is likely to throw is worth
# keeping one more turn: the same sum x P(its next throw is that rank) x this.
PAIR_IN = 1.0

# ---- 5. calling
# Call when the call is worth the best throw plus this. Every position where Yaniv was legal was played out
# both ways (calls-final.txt, 20,000 rounds against each of three opponents): the believed difference predicts
# the true one, and the true one crosses zero at a believed -2 (old counter) to -3 (the nets), because the
# value of throwing on is a little optimistic for a low hand.
CALL_EDGE = -2.5

_TABLES: dict = {}                                        # look-ahead tables, by the constants they were built from


def reset_tables() -> None:
    """Nothing to drop: every table is stored under the constants it was built from."""


def _lean(ply: int) -> list[float]:
    """Weight of a hidden card by its points: the opponent keeps low cards, more so as the round goes on."""
    beta = HIDDEN_BETA[min(ply // 2, len(HIDDEN_BETA) - 1)]
    return [exp(-beta * p) for p in range(11)]


def opp_total_cdf(view: View, unseen_pts: list[int]):
    """(cdf, weights, fresh): cdf[t] = P(the opponent's total <= t) for t = 0..CAP.

    `unseen_pts[p]` is the number of unseen cards worth p. A hand is weighted by the product of its cards'
    weights: exp(-beta x points) (hidden cards lean low, more so later in the round), times CAP_LEAK for a
    card above the single the opponent last threw. With a share FRESH one hidden card is instead a plain
    deck draw. Also returns the weights of a kept card by points and the fresh share, for the feeding rule.
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


def _clock_tables():
    """The value of every hand of up to five cards in a game played alone against a clock.

    The simplified game: on my turn I call Yaniv (with 7 or less) or throw one rank and draw from a full deck
    less my own cards. Then the opponent holds 7 or less with the chance of the current step of LADDER and
    calls by CALL_SHARE for the cards I hold; I lose my points, or catch it if I hold no more than it. Then
    the clock moves one step up. My own call is caught with the chance the opponent holds my total or less.

    This is not fitted to anything: it is the exact value of that game, found by working back from the top
    step (which repeats, so it is iterated until it settles). It replaces a hand-set cost per card: pairs,
    low cards, the hand's size and the race to 7 are all weighed by playing the hand out, many turns ahead.
    About 11,500 hands; built once per process in about 0.3 seconds, with NumPy only to make that fast.

    Returns ({hand code: row}, one list per step of the value of the best THROW from each hand). A hand's
    code is the sum of 5 ** rank over its cards.
    """
    key = ("clock", LADDER, CLOCK_LOW, CLOCK_PAYS, tuple(sorted(CALL_SHARE.items())), YANIV_LIMIT)
    tables = _TABLES.get(key)
    if tables is not None:
        return tables
    import itertools

    import numpy as np

    deck = [4] * 13 + [2]
    power = [5 ** r for r in range(14)]
    hands = []
    for n in range(1, 6):
        for combo in itertools.combinations_with_replacement(range(14), n):
            counts = [0] * 14
            for r in combo:
                counts[r] += 1
            if all(c <= d for c, d in zip(counts, deck)):
                hands.append(counts)
    counts = np.array(hands, dtype=np.int64)
    codes = counts @ np.array(power, dtype=np.int64)
    order = np.argsort(codes)
    counts, codes = counts[order], codes[order]
    points = counts @ np.array(RANK_PTS, dtype=np.int64)
    left = np.maximum(0, np.array(deck)[None, :] - counts).astype(np.float64)     # cards of each rank I may draw
    left /= left.sum(axis=1, keepdims=True)
    owner, succ, prob = [], [], []
    for r in range(14):
        for k in range(1, deck[r] + 1):
            rows = np.nonzero(counts[:, r] >= k)[0]
            if not len(rows):
                continue
            after = codes[rows, None] - k * power[r] + np.array(power, dtype=np.int64)[None, :]
            where = np.minimum(np.searchsorted(codes, after), len(codes) - 1)
            chance = left[rows]
            chance = np.where(codes[where] == after, chance, 0.0)
            owner.append(rows)
            succ.append(where)
            prob.append(chance)
    owner = np.concatenate(owner)
    by_owner = np.argsort(owner, kind="stable")
    succ = np.concatenate(succ)[by_owner]
    prob = np.concatenate(prob)[by_owner]
    starts = np.searchsorted(owner[by_owner], np.arange(len(codes)))

    cards = counts.sum(axis=1)
    can_call = points <= YANIV_LIMIT
    low = np.array([CLOCK_LOW[min(t, len(CLOCK_LOW) - 1)] for t in range(YANIV_LIMIT + 1)])
    at = np.diff(low, prepend=0.0)                     # of its hands worth 7 or less, the share worth exactly t
    calls = np.zeros(len(codes))                       # P(it calls | it holds <= 7), by the cards I hold
    hit = np.zeros(len(codes))                         # ... times my result when it does
    for t in range(YANIV_LIMIT + 1):
        share = at[t] * np.array([0.0] + [CALL_SHARE[n][min(t, 7)] for n in range(1, 6)])[cards]
        calls += share
        hit += share * np.where(points <= t, ASSAF_PENALTY + t, -points)

    def call_value(step: int) -> np.ndarray:
        risk = LADDER[step] * low
        pays = CLOCK_PAYS[0] + (CLOCK_PAYS[1] - CLOCK_PAYS[0]) * LADDER[step] / LADDER[-1]
        by_total = (1.0 - risk) * pays - risk * (np.arange(YANIV_LIMIT + 1) + ASSAF_PENALTY)
        return by_total[np.minimum(points, YANIV_LIMIT)]

    def sweep(step: int, after: np.ndarray):
        can = LADDER[step]
        mine = can * hit + (1.0 - can * calls) * after
        throw = np.maximum.reduceat((prob * mine[succ]).sum(axis=1), starts)
        return throw, np.where(can_call, np.maximum(call_value(step), throw), throw)

    top = len(LADDER) - 1
    value = np.zeros(len(codes))
    for _ in range(60):                                # the top step repeats: iterate until it settles
        throw, new = sweep(top, value)
        done = float(np.abs(new - value).max()) < 1e-4
        value = new
        if done:
            break
    throws = [None] * len(LADDER)
    throws[top] = throw.tolist()
    for step in range(top - 1, -1, -1):
        throw, value = sweep(step, value)
        throws[step] = throw.tolist()
    tables = ({int(code): i for i, code in enumerate(codes)}, throws)
    _TABLES[key] = tables
    return tables


def _round_values(view: View):
    """worth(mine, theirs): what a round that adds those points to the two scores is worth to me: theirs
    minus mine, plus 50 when my score lands exactly on 50 or 100 and is cut by 50."""
    score = view.my_score
    return lambda mine, theirs: float(theirs - mine) + (50.0 if score + mine in SCORE_CUTS else 0.0)


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


def _call_value(pts: int, cdf: list[float], est: float, worth) -> float:
    """My expected result if I call Yaniv holding `pts`: the opponent adds about its estimated total if the
    call stands; I add my points and the penalty if it is caught."""
    risk = cdf[pts]
    return (1.0 - risk) * max(est, pts + 3.0) + risk * worth(pts + ASSAF_PENALTY, 0)


def beliefs(view: View) -> dict:
    """What counter2 believes about the opponent (for heuristic2_study.py)."""
    count, my_pts, unseen, unseen_pts = _read(view)
    cdf, _, _ = opp_total_cdf(view, unseen_pts)
    share = CALL_SHARE[max(1, min(5, sum(count)))]
    hazard = cdf[0] * share[0] + sum((cdf[t] - cdf[t - 1]) * share[min(t, 7)] for t in range(1, YANIV_LIMIT + 1))
    return {"risk": cdf[my_pts] if my_pts <= CAP else 1.0, "hazard": hazard, "cdf": cdf,
            "value": evaluate(view)[0]}


def evaluate(view: View, detail: bool = False):
    """(value of my best move, the move): the move is (rank, cards thrown, draw), or None for Yaniv.

    The value is the expected result of the round in points (the opponent's minus mine). With `detail` a dict
    of the parts instead, for the studies.
    """
    count, my_pts, unseen, unseen_pts = _read(view)
    n_unseen = sum(unseen)
    my_cards = sum(count)
    cdf, kept_weight, fresh = opp_total_cdf(view, unseen_pts)
    est = _est(view, unseen_pts)
    worth = _round_values(view)

    # ---- 2. its call on its next turn: the chance, and my result by the points I hold then. Both depend on
    # the cards I hold after my move (throwing a set leaves fewer), so there is one table per card count.
    pdf = [cdf[0]] + [cdf[t] - cdf[t - 1] for t in range(1, YANIV_LIMIT + 1)]
    paid = [worth(x, 0) for x in range(51)]               # a score landing on 50 or 100 is cut: `worth` knows
    caught_it = [worth(0, ASSAF_PENALTY + t) for t in range(YANIV_LIMIT + 1)]
    by_cards = {}
    for n in range(1, my_cards + 1):
        share = CALL_SHARE[min(5, n)]
        calls = [pdf[t] * share[min(t, 7)] for t in range(YANIV_LIMIT + 1)]
        hazard = sum(calls)
        called = []                                       # my result x the chance of the call, by my points
        for x in range(51):
            if x <= YANIV_LIMIT:
                value = sum(calls[:x]) * paid[x]          # it holds less than I do: its call stands
                for t in range(x, YANIV_LIMIT + 1):       # it holds as much or more: I catch it
                    value += calls[t] * caught_it[t]
            else:
                value = hazard * paid[x]
            called.append(value)
        by_cards[n] = (1.0 - hazard, called)
    hazard = 1.0 - by_cards[my_cards][0]

    # ---- 3. my next turn: calling then (same risk as now, an approximation), or throwing on (the clock table)
    later = [_call_value(x, cdf, est, worth) for x in range(YANIV_LIMIT + 1)]
    index, throws = _clock_tables()
    second = SECOND_TURN[0] + SECOND_TURN[1] * cdf[YANIV_LIMIT]
    step = 0
    while step < len(LADDER) - 2 and LADDER[step + 1] <= second:
        step += 1
    between = min(1.0, max(0.0, (second - LADDER[step]) / (LADDER[step + 1] - LADDER[step])))
    lower, upper = throws[step], throws[step + 1]         # read between the two nearest steps
    hand_code = sum(count[r] * 5 ** r for r in range(14))

    # ---- 4. the chance the opponent holds each rank, and the chance its next throw is that rank
    need = view.opp_size - view.opp_known.bit_count()
    holds = [0.0] * 14
    bits = view.opp_known
    while bits:
        bit = bits & -bits
        holds[RANK_OF[bit.bit_length() - 1]] = 1.0        # it took that card from the table and still has it
        bits ^= bit
    kept_total = sum(unseen_pts[p] * kept_weight[p] for p in range(11)) or 1.0
    for r in range(14):
        if holds[r] < 1.0 and need > 0 and unseen[r] and n_unseen > need:
            # each kept card is of this rank with its weight's share; the fresh one as the deck deals
            share = unseen[r] * kept_weight[RANK_PTS[r]] / kept_total
            holds[r] = 1.0 - ((1.0 - share) ** (need - fresh)) * ((1.0 - unseen[r] / n_unseen) ** fresh)
    pair_gain = [0.0] * 14                                # it throws its highest cards first
    none_above = 1.0
    for level in BY_POINTS:
        total = sum(holds[r] for r in level)
        if total > 0.0:
            miss = 1.0
            for r in level:
                miss *= 1.0 - holds[r]
            for r in level:
                throws_next = none_above * (1.0 - miss) * holds[r] / total
                pair_gain[r] = PAIR_IN * throws_next * (FEED + FEED_POINT * RANK_PTS[r])
            none_above *= miss
    pair_all = sum(pair_gain[r] for r in range(14) if count[r])

    open_rank = RANK_OF[view.discard[-1]]
    best_key = None
    best_throw = None
    for r in range(14):
        have = count[r]
        if not have:
            continue
        feed = (FEED + FEED_POINT * RANK_PTS[r]) * holds[r]
        for k in range(1, have + 1):
            # the feeding charge, less the pairing chances of the ranks I keep
            charge = feed - pair_all + (pair_gain[r] if k == have else 0.0)
            stay, called = by_cards[my_cards - k + 1]
            kept_pts = my_pts - k * RANK_PTS[r]
            kept_code = hand_code - k * 5 ** r
            # take the open card: no chance involved
            pts = kept_pts + RANK_PTS[open_rank]
            row = index[kept_code + 5 ** open_rank]
            go_on = lower[row] + between * (upper[row] - lower[row])
            if pts <= YANIV_LIMIT and later[pts] > go_on:
                go_on = later[pts]
            key = (called[pts] + stay * go_on - charge, 1, r, k)
            if best_key is None or key > best_key:
                best_key, best_throw = key, (r, k, "open")
            # draw from the deck: every unseen card is as likely as another
            if view.stock_size > 0 and n_unseen > 0:
                value = 0.0
                for d in range(14):
                    n = unseen[d]
                    if n:
                        pts = kept_pts + RANK_PTS[d]
                        row = index[kept_code + 5 ** d]
                        go_on = lower[row] + between * (upper[row] - lower[row])
                        if pts <= YANIV_LIMIT and later[pts] > go_on:
                            go_on = later[pts]
                        value += n * (called[pts] + stay * go_on)
                key = (value / n_unseen - charge, 0, r, k)
                if key > best_key:
                    best_key, best_throw = key, (r, k, "stock")

    # ---- 5. Yaniv now: when it cannot be caught, when even a caught call lands my score on a cut, or when it
    # is worth the best throw
    throw_value = best_key[0] if best_key is not None else None
    if my_pts <= YANIV_LIMIT:
        risk = cdf[my_pts]
        call_now = _call_value(my_pts, cdf, est, worth)
        if detail:
            return {"risk": risk, "call": call_now, "throw": throw_value, "move": best_throw, "hazard": hazard}
        if (risk <= 0.0 or view.my_score + my_pts + ASSAF_PENALTY in SCORE_CUTS
                or throw_value is None or call_now >= throw_value + CALL_EDGE):
            return call_now, None
    elif detail:
        return {"throw": throw_value, "move": best_throw, "hazard": hazard}
    if best_key is None:
        return 0.0, None
    return throw_value, best_throw


def _action_for(throw, actions: list[Action]) -> Action:
    """The first legal action that makes the move (which suits go, and which stays face up, never matter)."""
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
