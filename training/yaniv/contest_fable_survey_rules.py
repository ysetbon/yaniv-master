"""Thread `survey`: rules stated a priori, scored from the play-outs collected by contest_fable_survey.probe_all.

A rule is a function(P) -> list of (class label, move) where P is a Pos (a decision with helpers) and move is
(rank, count, draw).  Its gain at a decision is the play-out margin of that move minus the net's move.

    python -m training.yaniv.contest_fable_survey_rules RULE[,RULE] [--by mc,oc,ply] hm5=FILE trained=FILE counter=FILE
"""

from __future__ import annotations

import math
import pickle
import sys

PTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 0]
NAME = "A23456789TJQK*"


class Pos:
    def __init__(self, d, gains):
        self.d, self.g = d, gains
        self.hand = d["hand"]
        self.cnt = {r: self.hand.count(r) for r in set(self.hand)}
        self.pts = sum(PTS[r] for r in self.hand)
        self.mc, self.oc, self.ply, self.open = len(self.hand), d["oc"], d["ply"], d["open"]
        self.ch = d["ch"]
        self.unseen = d["unseen"]
        self.singles = sorted((r for r, c in self.cnt.items() if c == 1), key=lambda r: (-PTS[r], self.unseen[r], r))
        self.sets = sorted((r for r, c in self.cnt.items() if c >= 2), key=lambda r: (-PTS[r] * self.cnt[r], r))
        self.stock = d["stock"]

    def has(self, move):
        return move in self.g or move == self.ch

    def gain(self, move):
        return 0.0 if move == self.ch else self.g[move]

    def kept_after(self, move):
        h = list(self.hand)
        for _ in range(move[1]):
            h.remove(move[0])
        return h


def band(x, cuts):
    for c in cuts:
        if x <= c:
            return f"<={c}"
    return f">{cuts[-1]}"


def pb(p):      # points band of a card
    return "0-1" if p <= 1 else "2" if p == 2 else "3" if p == 3 else "4" if p == 4 else "5-6" if p <= 6 else "7-9" if p <= 9 else "10"


# ------------------------------------------------------------------ rules
def swap_draw(P):
    r, n, draw = P.ch
    other = "open" if draw == "deck" or draw == "stock" else "stock"
    m = (r, n, other)
    if not P.has(m):
        return []
    kept = P.kept_after(P.ch)
    k = kept.count(P.open)
    pair = "pairs" if k == 1 else "triples" if k >= 2 else "nopair"
    lab = ("take" if other == "open" else "refuse") + f" open {pb(PTS[P.open])} {pair}"
    return [(lab, m)]


def hi_single(P):
    """throw the highest unpaired single (net's draw) when the net throws something else"""
    if not P.singles:
        return []
    m = (P.singles[0], 1, P.ch[2])
    if m[:2] == P.ch[:2]:
        return []
    r, n, _ = P.ch
    if n >= 2:
        lab = f"net set {n}x{pb(PTS[r])}"
    elif P.cnt[r] >= 2:
        lab = "net breaks pair"
    else:
        lab = f"net lower single {pb(PTS[r])}"
    return [(lab + f" -> hi single {pb(PTS[m[0]])}", m)]


def greedy(P):
    """throw the group worth the most points (net's draw)"""
    best = max(P.cnt, key=lambda r: (PTS[r] * P.cnt[r], -P.unseen[r], r))
    m = (best, P.cnt[best], P.ch[2])
    if m[:2] == P.ch[:2] or PTS[best] * P.cnt[best] == PTS[P.ch[0]] * P.ch[1]:
        return []
    r, n, _ = P.ch
    kind = "set" if n >= 2 else "pairbreak" if P.cnt[r] >= 2 else "single"
    return [(f"net {kind} {pb(PTS[r])} -> greedy {'set' if m[1] >= 2 else 'single'} {pb(PTS[best])}", m)]


def set_instead(P):
    """net throws a single; throw the whole set instead (each set held)"""
    r, n, draw = P.ch
    if n != 1:
        return []
    out = []
    for s in P.sets:
        if s == r:
            continue
        out.append((f"net single {pb(PTS[r])} -> set {P.cnt[s]}x{pb(PTS[s])}", (s, P.cnt[s], draw)))
    return out


def whole_set(P):
    """net throws part of a set: throw all of it"""
    r, n, draw = P.ch
    if P.cnt[r] > n:
        return [(f"net {n} of {P.cnt[r]} {pb(PTS[r])} -> all", (r, P.cnt[r], draw))]
    return []


def single2(P):
    """throw the second-highest unpaired single when the net throws the highest"""
    r, n, draw = P.ch
    if len(P.singles) < 2 or n != 1 or r != P.singles[0] and PTS[r] != PTS[P.singles[0]]:
        return []
    s = P.singles[1]
    if s == r:
        s = P.singles[0]
    return [(f"net hi single {pb(PTS[r])} -> 2nd single {pb(PTS[s])}", (s, 1, draw))]


def break_pair(P):
    """throw one card of a held pair (the highest pair) when the net does not"""
    r, n, draw = P.ch
    out = []
    for s in P.sets:
        if s == r:
            continue
        out.append((f"net {n}x{pb(PTS[r])} -> one of {P.cnt[s]}x{pb(PTS[s])}", (s, 1, draw)))
    return out


def pair_take(P):
    """the open card's rank is in my hand and the net does not end up taking it onto a kept twin:
    throw the best group of another rank and take the open card"""
    o = P.open
    if o not in P.cnt or (P.ch[2] == "open" and P.ch[0] != o):
        return []
    others = [r for r in P.cnt if r != o]
    if not others:
        return []
    best = max(others, key=lambda r: (PTS[r] * P.cnt[r], -P.unseen[r], r))
    r, n, draw = P.ch
    what = "throws the twin" if r == o else "draws deck"
    return [(f"net {what} ({n}x{pb(PTS[r])},{draw}) -> take open {pb(PTS[o])} onto {P.cnt[o]}", (best, P.cnt[best], "open"))]


def least_live(P):
    """net throws a single; another unpaired single of the SAME points has fewer unseen copies: throw that"""
    r, n, draw = P.ch
    if n != 1 or P.cnt[r] != 1:
        return []
    same = [s for s in P.singles if PTS[s] == PTS[r] and s != r]
    if not same:
        return []
    s = min(same, key=lambda x: (P.unseen[x], x))
    if P.unseen[s] >= P.unseen[r]:
        return []
    return [(f"net single unseen {P.unseen[r]} -> twin-value single unseen {P.unseen[s]}", (s, 1, draw))]


def most_live(P):
    """the reverse: throw the same-points single with MORE unseen copies"""
    r, n, draw = P.ch
    if n != 1 or P.cnt[r] != 1:
        return []
    same = [s for s in P.singles if PTS[s] == PTS[r] and s != r]
    if not same:
        return []
    s = max(same, key=lambda x: (P.unseen[x], x))
    if P.unseen[s] <= P.unseen[r]:
        return []
    return [(f"net single unseen {P.unseen[r]} -> twin-value single unseen {P.unseen[s]}", (s, 1, draw))]


def low_single(P):
    """net throws a low card (<= 3 points) as a single: throw the highest other unpaired single instead"""
    r, n, draw = P.ch
    if n != 1 or PTS[r] > 3:
        return []
    others = [s for s in P.singles if s != r]
    if not others:
        return []
    return [(f"net single {pb(PTS[r])} -> single {pb(PTS[others[0]])}", (others[0], 1, draw))]


def take_low_best(P):
    """net draws the deck although the open card is worth <= 4: take it (same throw unless the net throws that rank)"""
    if P.ch[2] == "open" or PTS[P.open] > 4:
        return []
    m = (P.ch[0], P.ch[1], "open")
    return [(f"take open {PTS[P.open]}", m)]


def refuse_low(P):
    """net takes an unpaired open card: draw the deck instead"""
    if P.ch[2] != "open":
        return []
    kept = P.kept_after(P.ch)
    if P.open in kept:
        return []
    return [(f"refuse open {PTS[P.open]}", (P.ch[0], P.ch[1], "stock"))]


def any_best(P):
    """NOT a rule: the best move in hindsight averaged per class of (net move kind) - only to see where spread is"""
    return []


RULES = {f.__name__: f for f in (swap_draw, hi_single, greedy, set_instead, whole_set, single2, break_pair, pair_take, least_live, most_live, low_single, take_low_best, refuse_low)}


# ------------------------------------------------------------------ scoring
def load(path):
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    return [Pos(d, g) for d, _, g in data["decisions"]], data["stats"]["rounds"]


KEYS = {"mc": lambda P: f"me{P.mc}", "oc": lambda P: f"opp{P.oc}", "ply": lambda P: "ply" + band(P.ply, (3, 9, 19)),
        "pts": lambda P: "pts" + band(P.pts, (7, 12, 20, 30)), "est": lambda P: "est" + band(P.d["est"], (8, 14, 20)),
        "stock": lambda P: "stock" + band(P.stock, (5, 15)), "yl": lambda P: "yl" if P.d["ylegal"] else "",
        "ocb": lambda P: "opp" + ("5" if P.oc == 5 else "4" if P.oc == 4 else "1-3"),
        "live": lambda P: f"live{P.unseen[P.ch[0]]}", "drw": lambda P: P.ch[2],
        "opn": lambda P: "open" + pb(PTS[P.open]), "known": lambda P: f"known{len(P.d['known'])}",
        "mcb": lambda P: "me" + ("5" if P.mc == 5 else "4" if P.mc == 4 else "1-3")}


def score(rule, sets, by=(), least=40, show_all=False, where=None):
    """sets: {opp: (positions, rounds)}. Prints per class: for each opponent n, each +-se, a round."""
    f = RULES[rule] if isinstance(rule, str) else rule
    table = {}
    for opp, (pos, rounds) in sets.items():
        for P in pos:
            if where and not where(P):
                continue
            for lab, m in f(P):
                if not P.has(m):
                    continue
                key = (lab,) + tuple(KEYS[k](P) for k in by)
                table.setdefault(key, {}).setdefault(opp, []).append(P.gain(m))
    lines = []
    for key in sorted(table):
        cells, signs, strong, big = [], set(), 0, 0
        for opp, (_, rounds) in sets.items():
            g = table[key].get(opp, [])
            if not g:
                cells.append(f"{opp}: -")
                continue
            mean = sum(g) / len(g)
            se = math.sqrt(max(1e-9, sum(x * x for x in g) / len(g) - mean * mean) / len(g))
            per_round = sum(g) / rounds
            signs.add(mean > 0)
            strong += abs(mean) >= 3 * se
            big += abs(per_round) >= 0.01
            cells.append(f"{opp}: n {len(g):5d} each {mean:+6.2f}+-{se:.2f} rd {per_round:+.4f}")
        n = max(len(v) for v in table[key].values())
        flag = "***" if len(signs) == 1 and strong == len(sets) and big == len(sets) else \
               "** " if len(signs) == 1 and strong >= 2 and big >= 2 else "*  " if len(signs) == 1 and strong >= 1 else "   "
        if n >= least and (show_all or flag.strip() or not by):
            lines.append(f"{flag} {' '.join(key):<58} | " + " | ".join(cells))
    return "\n".join(lines)


if __name__ == "__main__":
    args = sys.argv[1:]
    by, rules, sets, show_all = (), [], {}, False
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--by":
            by = tuple(args[i + 1].split(","))
            i += 1
        elif a == "--all":
            show_all = True
        elif "=" in a:
            k, v = a.split("=")
            sets[k] = load(v)
        else:
            rules += a.split(",")
        i += 1
    for r in rules:
        print(f"== {r} by {by}")
        print(score(r, sets, by, show_all=show_all))
