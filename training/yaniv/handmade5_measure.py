"""Real-game statistics for the designers of the hand-written 5-card net (results/handmade-5x7-logs/measure.md).

    python -m training.yaniv.handmade5_measure part1 [--rounds 40000]     # tables a-d
    python -m training.yaniv.handmade5_measure e [--rounds 6000]          # behaviour while someone holds 5 cards
    python -m training.yaniv.handmade5_measure f [--rounds 40000]         # what a card is worth
    python -m training.yaniv.handmade5_measure g [--matches 3000]         # score cuts per match (+ builds hm4 without score units)

Each part prints markdown. Real-game rules: YANIV_HAND_SIZE / YANIV_CALL_AT must be unset. Two processes.
Seeds are 900000 + part offset + chunk number. Nothing here fits or searches weights: it only plays and counts.

Players: "counter" (heuristic.choose_heuristic), "hm4" (the 4-card hand-written net, unchanged), "trained"
(public/model/yaniv-net.json), or a path to a net file.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .engine import (MAX_PLIES, POINTS_OF, RANK_CHARS, SCORE_CUTS, STANDARD_RULES, State, apply_action,
                     card_name, cards_from_mask, legal_actions, make_view, mask_points, new_round, popcount,
                     rank_of, resolve_showdown)
from .features import RANKS, action_key, net_actions, state_features

ROOT = Path(__file__).resolve().parents[2]
NETS = {"hm4": ROOT / "public/model/yaniv-net-4cards-yaniv7-handmade.json",
        "trained": ROOT / "public/model/yaniv-net.json"}
EST_AT, ASSAF_AT = 5 * RANKS + 19 + 5, 5 * RANKS + 19 + 5 + 1
PROCESSES = 2
_CHOOSERS: dict = {}


def chooser_of(kind: str):
    if kind not in _CHOOSERS:
        if kind == "counter":
            from .heuristic import choose_heuristic
            _CHOOSERS[kind] = choose_heuristic
        else:
            from .network import load_net
            from .train import net_chooser
            _CHOOSERS[kind] = net_chooser(load_net(Path(NETS.get(kind, kind))), 0.0)
    return _CHOOSERS[kind]


def copy_state(s: State) -> State:
    return State(hands=list(s.hands), known=list(s.known), stock=list(s.stock), discard=list(s.discard),
                 scores=list(s.scores), turn=s.turn, ply=s.ply)


def run(state: State, choosers, log=None, snap_at=()) -> tuple[State, dict]:
    """engine.play_round's loop from any state, with a hook that sees the true state (the opponent's hand)."""
    snaps = {}
    while not state.terminal:
        if state.ply >= MAX_PLIES:
            resolve_showdown(state)
            break
        view = make_view(state, state.turn)
        actions = legal_actions(view)
        if not actions:
            resolve_showdown(state)
            break
        if state.ply in snap_at:
            snaps[state.ply] = copy_state(state)
        action = choosers[state.turn](view, actions, None)
        if log is not None:
            log(state, view, actions, action)
        apply_action(state, action)
    return state, snaps


# decision columns
WHO, PLY, MY_N, OPP_N, MY_PTS, OPP_PTS, EST, ASSAF, CAN, CALLED, NEXT_CALL, NEXT_CAN, RID = range(13)
# round columns
R_PLIES, R_SHOW, R_CALLER, R_CAUGHT, R_CN, R_ON, R_CP, R_OP, R_ADD_A, R_ADD_B = range(10)


def _logged_rounds(job):
    """Rounds of A against B (seats and the lead alternate). Returns every decision and every round."""
    kind_a, kind_b, n, seed, chunk = job
    rng = random.Random(seed)
    a, b = chooser_of(kind_a), chooser_of(kind_b)
    decs, rounds = [], []
    for i in range(n):
        seat_a = i % 2
        choosers = (a, b) if seat_a == 0 else (b, a)
        state = new_round([0, 0], (i // 2) % 2, rng)
        pending = [None, None]
        rid = chunk * 1_000_000 + i

        def log(st, view, actions, action):
            seat = st.turn
            f = state_features(view)
            mine, theirs = mask_points(view.my_hand), mask_points(st.hands[1 - seat])
            called = action.kind == "yaniv"
            j = pending[1 - seat]
            if j is not None:
                decs[j][NEXT_CALL] = 1.0 if called else 0.0
                decs[j][NEXT_CAN] = 1.0 if mine <= 7 else 0.0
            decs.append([0 if seat == seat_a else 1, st.ply, popcount(view.my_hand), view.opp_size, mine, theirs,
                         float(f[EST_AT]) * 40.0, float(f[ASSAF_AT]), 1.0 if mine <= 7 else 0.0,
                         1.0 if called else 0.0, -1.0, -1.0, rid])
            pending[seat] = None if called else len(decs) - 1

        run(state, choosers, log)
        res = state.result
        if res.caller is None:
            rounds.append([state.ply, 1, -1, 0, 0, 0, 0, 0, res.added[seat_a], res.added[1 - seat_a]])
        else:
            c = res.caller
            rounds.append([state.ply, 0, 0 if c == seat_a else 1, 1 if res.assaf else 0, popcount(res.hands[c]),
                           popcount(res.hands[1 - c]), res.points[c], res.points[1 - c],
                           res.added[seat_a], res.added[1 - seat_a]])
    return np.array(decs, dtype=np.float64), np.array(rounds, dtype=np.float64)


def logged(kind_a: str, kind_b: str, rounds: int, seed: int):
    per = 1000
    jobs = [(kind_a, kind_b, min(per, rounds - s), seed + k, k) for k, s in enumerate(range(0, rounds, per))]
    with Pool(PROCESSES) as pool:
        parts = pool.map(_logged_rounds, jobs, chunksize=1)
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


# ---------------------------------------------------------------- markdown helpers
def cell(hits: float, n: int, digits: int = 2) -> str:
    """rate (n). `*` under 100 samples (rough), `**` under 30 (do not trust)."""
    if n == 0:
        return "-"
    mark = "**" if n < 30 else "*" if n < 100 else ""
    return f"{hits / n:.{digits}f} ({n}){mark}"


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


EST_CUTS = [5, 8, 11, 14, 17]
EST_NAMES = ["<5", "5-8", "8-11", "11-14", "14-17", "17+"]
ASSAF_CUTS = [0.01, 0.02, 0.04, 0.08, 0.15]
ASSAF_NAMES = ["0", "0-0.01", "0.01-0.02", "0.02-0.04", "0.04-0.08", "0.08-0.15", "0.15+"]
PLY_CUTS = [6, 10, 14, 20]
PLY_NAMES = ["ply 0-5", "6-9", "10-13", "14-19", "20+"]


def assaf_bin(x: np.ndarray) -> np.ndarray:
    """0 for exactly 0; 1.. for (0,0.01], (0.01,0.02], ..."""
    return np.where(x <= 0, 0, 1 + np.searchsorted(ASSAF_CUTS, x, side="left"))


def part_a(d: np.ndarray, me: str) -> str:
    d = d[(d[:, CALLED] == 0) & (d[:, NEXT_CALL] >= 0)]
    eb = np.searchsorted(EST_CUTS, d[:, EST], side="right")
    out = [f"**me = {me}, opponent = counter** ({len(d):,} of my throws that the opponent answered). "
           f"Cell: P(opponent calls on its next turn) (samples); `*` n<100, `**` n<30.\n"]
    rows = []
    for label, sel in [(str(n), d[:, OPP_N] == n) for n in (5, 4, 3, 2, 1)] + [("1-4", d[:, OPP_N] <= 4),
                                                                              ("all", d[:, OPP_N] > 0)]:
        rows.append([label] + [cell(d[sel & (eb == k), NEXT_CALL].sum(), int((sel & (eb == k)).sum()))
                               for k in range(6)] + [cell(d[sel, NEXT_CALL].sum(), int(sel.sum()))])
    out.append(table(["opp cards \\ opp_est"] + EST_NAMES + ["any"], rows))
    out.append("\nSame cells, P(the opponent CAN call on its next turn: it holds <= 7 then):\n")
    rows = []
    for label, sel in [(str(n), d[:, OPP_N] == n) for n in (5, 4, 3, 2, 1)] + [("1-4", d[:, OPP_N] <= 4)]:
        rows.append([label] + [cell(d[sel & (eb == k), NEXT_CAN].sum(), int((sel & (eb == k)).sum()))
                               for k in range(6)] + [cell(d[sel, NEXT_CAN].sum(), int(sel.sum()))])
    out.append(table(["opp cards \\ opp_est"] + EST_NAMES + ["any"], rows))
    out.append("\nFiner, by opp_est rounded down to a whole point (P calls next turn):\n")
    fine = np.floor(d[:, EST]).astype(int)
    cols = list(range(0, 15)) + [15]
    rows = []
    for label, sel in [("4", d[:, OPP_N] == 4), ("3", d[:, OPP_N] == 3), ("2", d[:, OPP_N] == 2),
                       ("1", d[:, OPP_N] == 1), ("1-4", d[:, OPP_N] <= 4)]:
        row = [label]
        for c in cols:
            s = sel & ((fine == c) if c < 15 else (fine >= 15))
            row.append(cell(d[s, NEXT_CALL].sum(), int(s.sum())))
        rows.append(row)
    out.append(table(["opp cards \\ est"] + [str(c) for c in cols[:-1]] + ["15+"], rows))
    out.append("\nBy ply band (opponent holding 1-4 cards), P calls next turn:\n")
    pb = np.searchsorted(PLY_CUTS, d[:, PLY], side="right")
    rows = []
    for p, name in enumerate(PLY_NAMES):
        sel = (d[:, OPP_N] <= 4) & (pb == p)
        rows.append([name] + [cell(d[sel & (eb == k), NEXT_CALL].sum(), int((sel & (eb == k)).sum()))
                              for k in range(6)])
    out.append(table(["ply \\ opp_est"] + EST_NAMES, rows))
    return "\n".join(out)


def part_b(d: np.ndarray, me: str, opp: str) -> str:
    d = d[d[:, CAN] == 1]
    caught = (d[:, OPP_PTS] <= d[:, MY_PTS]).astype(float)
    ab = assaf_bin(d[:, ASSAF])
    called = d[:, CALLED] == 1
    out = [f"**me = {me}, opponent = {opp}**: {len(d):,} positions where I hold <= 7 at my turn; I called in "
           f"{int(called.sum()):,} of them, caught {caught[called].mean():.3f}; not calling: the opponent was at or "
           f"under me in {caught[~called].mean():.3f} of {int((~called).sum()):,}.\n"
           f"Cell: real share with the opponent's hand <= mine (samples); `*` n<100, `**` n<30.\n"]
    rows = []
    for k, name in enumerate(ASSAF_NAMES):
        sel = ab == k
        row = [name, f"{d[sel, ASSAF].mean():.3f}" if sel.any() else "-"]
        for n in (5, 4, 3, 2, 1):
            s = sel & (d[:, OPP_N] == n)
            row.append(cell(caught[s].sum(), int(s.sum()), 3))
        row.append(cell(caught[sel].sum(), int(sel.sum()), 3))
        ratio = caught[sel].mean() / d[sel, ASSAF].mean() if sel.any() and d[sel, ASSAF].mean() > 0 else None
        row.append(f"{ratio:.1f}x" if ratio is not None else "-")
        rows.append(row)
    out.append(table(["assaf", "mean assaf", "opp 5", "opp 4", "opp 3", "opp 2", "opp 1", "all", "real / assaf"],
                     rows))
    out.append("\nBy ply band:\n")
    pb = np.searchsorted(PLY_CUTS, d[:, PLY], side="right")
    rows = []
    for k, name in enumerate(ASSAF_NAMES):
        sel = ab == k
        rows.append([name] + [cell(caught[sel & (pb == p)].sum(), int((sel & (pb == p)).sum()), 3)
                              for p in range(len(PLY_NAMES))])
    out.append(table(["assaf"] + PLY_NAMES, rows))
    out.append("\nBy my own total (all positions where I may call), and with assaf <= 0.04 only:\n")
    rows = []
    low = d[:, ASSAF] <= 0.04
    for pts in range(8):
        sel = d[:, MY_PTS] == pts
        rows.append([str(pts), f"{d[sel, ASSAF].mean():.3f}" if sel.any() else "-",
                     cell(caught[sel].sum(), int(sel.sum()), 3),
                     f"{d[sel & low, ASSAF].mean():.3f}" if (sel & low).any() else "-",
                     cell(caught[sel & low].sum(), int((sel & low).sum()), 3)])
    out.append(table(["my total", "mean assaf", "real caught", "mean assaf (<=0.04)", "real caught (<=0.04)"], rows))
    out.append("\nBy the opponent's estimated total `opp_est` minus my total (all assaf values), opponent 1-4 cards:\n")
    gap = d[:, EST] - d[:, MY_PTS]
    cuts = [0, 2, 4, 6, 8, 11]
    names = ["<0", "0-2", "2-4", "4-6", "6-8", "8-11", "11+"]
    gb = np.searchsorted(cuts, gap, side="right")
    rows = []
    for n in (4, 3, 2, 1):
        rows.append([str(n)] + [cell(caught[(d[:, OPP_N] == n) & (gb == k)].sum(),
                                     int(((d[:, OPP_N] == n) & (gb == k)).sum()), 3) for k in range(len(names))])
    out.append(table(["opp cards \\ est - mine"] + names, rows))
    return "\n".join(out)


def part_c(r: np.ndarray, caller: int, who_calls: str, who_catches: str) -> str:
    r = r[(r[:, R_SHOW] == 0) & (r[:, R_CALLER] == caller)] if caller >= 0 else r[r[:, R_SHOW] == 0]
    out = [f"**{who_calls} calls, {who_catches} is called on** ({len(r):,} calls). The caller's total by its card "
           f"count; `caught` = P(the other hand <= the caller's total).\n"]
    rows = []
    for label, sel in [(str(n), r[:, R_CN] == n) for n in (5, 4, 3, 2, 1)] + [("all", r[:, R_CN] > 0)]:
        n = int(sel.sum())
        if n == 0:
            rows.append([label, "0"] + ["-"] * 13)
            continue
        dist = [f"{(r[sel, R_CP] == p).mean():.0%}" for p in range(8)]
        rows.append([label, f"{n} ({n / len(r):.1%})", f"{r[sel, R_CP].mean():.2f}"] + dist
                    + [cell(r[sel, R_CAUGHT].sum(), n, 3), f"{r[sel, R_OP].mean():.1f}",
                       f"{(r[sel, R_OP] <= 7).mean():.2f}"])
    out.append(table(["caller's cards", "calls (share)", "mean total"] + [f"={p}" for p in range(8)]
                     + ["caught", "other hand's mean total", "P(other hand <= 7)"], rows))
    out.append("\nCaught rate by the caller's total (all card counts), and the share of calls with the other hand "
               "<= 7:\n")
    rows = [[str(p), cell(r[r[:, R_CP] == p, R_CAUGHT].sum(), int((r[:, R_CP] == p).sum()), 3)] for p in range(8)]
    out.append(table(["caller's total", "caught"], rows))
    return "\n".join(out)


def part_d(d: np.ndarray, r: np.ndarray, me: str) -> str:
    mine = d[d[:, WHO] == 0]
    n_rounds = len(r)
    conds = {"both <= 4": (mine[:, MY_N] <= 4) & (mine[:, OPP_N] <= 4), "mine <= 4": mine[:, MY_N] <= 4,
             "opp <= 4": mine[:, OPP_N] <= 4, "either <= 4": (mine[:, MY_N] <= 4) | (mine[:, OPP_N] <= 4)}
    out = [f"**me = {me} vs counter** ({n_rounds:,} rounds from 0-0, {len(mine):,} of my decisions, "
           f"{len(mine) / n_rounds:.2f} a round). A round 'reaches' a point when I make at least one decision "
           f"there.\n"]
    rows = []
    for name, sel in conds.items():
        first = {}
        for rid, ply in zip(mine[sel, RID], mine[sel, PLY]):
            if rid not in first:
                first[rid] = ply
        plies = np.array(sorted(first.values()))
        q = np.percentile(plies, [25, 50, 75])
        rows.append([name, f"{len(first) / n_rounds:.1%}", f"{sel.mean():.1%}",
                     f"{plies.mean():.1f}", f"{q[0]:.0f} / {q[1]:.0f} / {q[2]:.0f}"])
    out.append(table(["point", "rounds reaching it", "my decisions after it", "mean ply of my first decision there",
                      "quartiles of that ply"], rows))
    out.append("\nShare of my decisions by (my cards, opponent's cards):\n")
    rows = []
    for a in (5, 4, 3, 2, 1):
        rows.append([str(a)] + [f"{((mine[:, MY_N] == a) & (mine[:, OPP_N] == b)).mean():.1%}" for b in (5, 4, 3, 2, 1)]
                    + [f"{(mine[:, MY_N] == a).mean():.1%}"])
    rows.append(["any"] + [f"{(mine[:, OPP_N] == b).mean():.1%}" for b in (5, 4, 3, 2, 1)] + ["100%"])
    out.append(table(["my cards \\ opp cards", "5", "4", "3", "2", "1", "any"], rows))
    q = np.percentile(r[:, R_PLIES], [10, 25, 50, 75, 90])
    out.append(f"\nRound length in plies (throws by both): mean {r[:, R_PLIES].mean():.1f}, 10/25/50/75/90% = "
               + " / ".join(f"{x:.0f}" for x in q) + f"; showdowns (48 plies): {r[:, R_SHOW].mean():.2%}.")
    out.append(f"Result: I add {r[:, R_ADD_A].mean():.2f} a round, the counter {r[:, R_ADD_B].mean():.2f} "
               f"(margin {r[:, R_ADD_B].mean() - r[:, R_ADD_A].mean():+.2f}, +-"
               f"{1.96 * (r[:, R_ADD_B] - r[:, R_ADD_A]).std() / np.sqrt(n_rounds):.2f}).\n")
    rows = []
    for who, name in ((0, me), (1, "counter")):
        sel = (r[:, R_SHOW] == 0) & (r[:, R_CALLER] == who)
        row = [name, f"{sel.mean():.1%}", f"{r[sel, R_CAUGHT].mean():.1%}", f"{r[sel, R_CP].mean():.2f}"]
        row += [f"{(r[sel, R_CN] == n).mean():.1%}" for n in (5, 4, 3, 2, 1)]
        row += [f"{(r[sel, R_ON] == n).mean():.1%}" for n in (5, 4, 3, 2, 1)]
        rows.append(row)
    out.append(table(["caller", "share of rounds", "caught", "mean total"] + [f"caller has {n}" for n in (5, 4, 3, 2, 1)]
                     + [f"other has {n}" for n in (5, 4, 3, 2, 1)], rows))
    out.append("\nShare of all calls by (caller's cards, other's cards), both callers together:\n")
    calls = r[r[:, R_SHOW] == 0]
    rows = []
    for a in (5, 4, 3, 2, 1):
        rows.append([str(a)] + [f"{((calls[:, R_CN] == a) & (calls[:, R_ON] == b)).mean():.1%}" for b in (5, 4, 3, 2, 1)])
    out.append(table(["caller's cards \\ other's", "5", "4", "3", "2", "1"], rows))
    return "\n".join(out)


def part1(rounds: int) -> None:
    hm_d, hm_r = logged("hm4", "counter", rounds, 900_000)
    cc_d, cc_r = logged("counter", "counter", rounds, 901_000)
    tr_d, tr_r = logged("trained", "counter", rounds // 2, 902_000)
    print(f"# Real-game measurements (5 cards, Yaniv at 7)\n\nScript: `python -m training.yaniv.handmade5_measure "
          f"part1 --rounds {rounds}` (seeds 900000+). Rounds start from 0-0, seats and the lead alternate. "
          f"hm4 = the 4-card hand-written net playing the whole real game unchanged. `opp_est` and `assaf` are "
          f"the net's inputs 94 and 95 (x40 for opp_est) at my decision, before my throw. "
          f"Games: hm4 vs counter {len(hm_r):,} rounds, counter vs counter {len(cc_r):,} (both seats counted as "
          f"'me'), trained vs counter {len(tr_r):,}.\n\nMarks: `*` fewer than 100 samples (rough: +-0.1 or worse "
          f"on a rate), `**` fewer than 30 (do not use).\n")
    print("## a. P(the counter calls Yaniv on its next turn) by its card count and `opp_est`\n")
    print(part_a(hm_d[hm_d[:, WHO] == 0], "hm4"))
    print()
    print(part_a(cc_d, "counter"))
    print("\n## b. Real catch rate of a Yaniv call against the `assaf` input\n")
    print(part_b(hm_d[hm_d[:, WHO] == 0], "hm4", "counter"))
    print()
    print(part_b(cc_d, "counter", "counter"))
    print()
    print(part_b(tr_d[tr_d[:, WHO] == 0], "trained", "counter"))
    print("\n## c. The counter's total when it calls\n")
    print(part_c(hm_r, 1, "counter", "hm4"))
    print()
    print(part_c(cc_r, -1, "counter", "counter"))
    print()
    print(part_c(hm_r, 0, "hm4", "counter"))
    print("\n## d. Hand sizes over a round\n")
    print(part_d(hm_d, hm_r, "hm4"))
    print()
    print(part_d(tr_d, tr_r, "trained"))

# ---------------------------------------------------------------- e. behaviour while someone holds 5 cards
PLAYERS = ("counter", "hm4", "trained")


def hand_text(mask: int) -> str:
    return " ".join(card_name(c) for c in sorted(cards_from_mask(mask), key=lambda c: (POINTS_OF[c], rank_of(c))))


def key_text(key: tuple) -> str:
    if key[0] == "yaniv":
        return "calls Yaniv"
    rank = "joker" if key[1] == 13 else RANK_CHARS[key[1]]
    what = rank if key[2] == 1 else f"{key[2]} x {rank}"
    return f"throws {what}, " + ("takes the open card" if key[3] == "open" else "draws from the deck")


def throw_kind(key: tuple, counts: dict, top_pts: int) -> str:
    """set / highest single / one card of a pair / a lower single / yaniv."""
    if key[0] == "yaniv":
        return "Yaniv"
    rank, n = key[1], key[2]
    if n >= 2:
        return "a set"
    if counts[rank] >= 2:
        return "one card of a pair"
    pts = 0 if rank == 13 else POINTS_OF[rank]
    return "its highest single" if pts >= top_pts else "a lower single"


def _behaviour(job):
    source, n, seed = job
    rng = random.Random(seed)
    hero, counter = chooser_of(source), chooser_of("counter")
    players = [chooser_of(p) for p in PLAYERS]
    sets, opens, cells, kinds = Counter(), Counter(), Counter(), Counter()
    examples: dict = defaultdict(list)
    for i in range(n):
        seat = i % 2
        choosers = (hero, counter) if seat == 0 else (counter, hero)
        state = new_round([0, 0], (i // 2) % 2, rng)

        def log(st, view, actions, action):
            my_n = popcount(view.my_hand)
            if st.turn != seat or (my_n < 5 and view.opp_size < 5):
                return
            unique = net_actions(actions)[0]
            keys = [action_key(c(view, unique, None)) for c in players]
            cards = cards_from_mask(view.my_hand)
            counts = Counter(rank_of(c) for c in cards)
            pts_of = {r: (0 if r == 13 else POINTS_OF[r]) for r in counts}
            open_card = view.discard[-1]
            open_rank, open_pts = rank_of(open_card), POINTS_OF[open_card]
            opp5 = view.opp_size >= 5
            may_call = mask_points(view.my_hand) <= 7
            # 1. a set is available (exactly one rank held twice or more)
            multi = [r for r, c in counts.items() if c >= 2]
            if len(multi) == 1 and not may_call:
                r = multi[0]
                x = pts_of[r]
                others = [pts_of[q] for q in counts if q != r]
                h = max(others) if others else -1
                rel = "none" if h < 0 else "H >= 2X" if h >= 2 * x else "X < H < 2X" if h > x else "H <= X"
                for p, key in zip(PLAYERS, keys):
                    what = "set" if key[2] >= 2 else "break" if key[1] == r else "keep"
                    sets[(p, x, counts[r], rel, opp5, my_n, what)] += 1
            # 2. the open card
            if not may_call:
                for p, key in zip(PLAYERS, keys):
                    kept_pair = open_rank in counts and not (key[1] == open_rank and key[2] == counts[open_rank])
                    opens[(p, open_pts, open_rank in counts, opp5, key[3] == "open", kept_pair)] += 1
            # 3. agreement
            for a, b in ((0, 1), (0, 2), (1, 2)):
                cells[(a, b, my_n, view.opp_size, keys[a] == keys[b])] += 1
            if keys[1] != keys[2]:
                top = max(pts_of.values())
                k1, k2 = keys[1], keys[2]
                if k1[0] == "yaniv" or k2[0] == "yaniv":
                    kind = "hm4 calls Yaniv, the trained net plays on" if k1[0] == "yaniv" else \
                        "the trained net calls Yaniv, hm4 plays on"
                elif k1[1:3] == k2[1:3]:
                    taker = "hm4" if k1[3] == "open" else "the trained net"
                    band = "0-3" if open_pts <= 3 else "4-6" if open_pts <= 6 else "7-10"
                    pair = "pairing a card it holds" if open_rank in counts else "unpaired"
                    kind = f"same throw; only {taker} takes the open card (worth {band}, {pair})"
                else:
                    t1, t2 = throw_kind(k1, counts, top), throw_kind(k2, counts, top)
                    if t1 == t2 and pts_of[k1[1]] == pts_of[k2[1]] and k1[2] == k2[2]:
                        kind = (f"both throw {t1.replace('its highest', 'a ten-point')} but a different rank of the "
                                f"same points (which of T/J/Q/K goes)")
                    elif t1 == t2:
                        kind = f"both throw {t1.replace('its ', 'a ')} but of a different rank and value"
                    else:
                        kind = f"hm4 throws {t1}, the trained net throws {t2.replace('its', 'the')}"
                kinds[kind] += 1
                if len(examples[kind]) < 4:
                    examples[kind].append(
                        f"hand {hand_text(view.my_hand)} ({mask_points(view.my_hand)} pts), open {card_name(open_card)}, "
                        f"opponent {view.opp_size} cards, ply {st.ply}: hm4 {key_text(k1)}; trained {key_text(k2)}; "
                        f"counter {key_text(keys[0])}")

        run(state, choosers, log)
    return sets, opens, cells, kinds, dict(examples)


def part_e(rounds: int) -> None:
    per = 500
    jobs = [(src, min(per, rounds - s), 903_000 + 500 * j + k)
            for j, src in enumerate(("hm4", "trained")) for k, s in enumerate(range(0, rounds, per))]
    with Pool(PROCESSES) as pool:
        parts = pool.map(_behaviour, jobs, chunksize=1)
    sets, opens, cells, kinds = Counter(), Counter(), Counter(), Counter()
    examples: dict = defaultdict(list)
    for s, o, c, k, ex in parts:
        sets.update(s), opens.update(o), cells.update(c), kinds.update(k)
        for name, items in ex.items():
            examples[name] = (examples[name] + items)[:4]
    n_pos = sum(v for (a, b, *_), v in cells.items() if (a, b) == (1, 2))
    print(f"## e. Behaviour while someone holds 5 cards\n\n`python -m training.yaniv.handmade5_measure e --rounds "
          f"{rounds}`: {n_pos:,} positions (my decisions with 5 cards in my hand or the opponent's), from {rounds:,} "
          f"rounds of hm4 vs counter and {rounds:,} of trained vs counter (seeds 903000+). The three players choose "
          f"on the same positions. Cells: counter / hm4 / trained (samples).\n")

    def three(counter: Counter, match, hit) -> str:
        vals, n = [], 0
        for p in PLAYERS:
            tot = sum(v for k, v in counter.items() if k[0] == p and match(k))
            good = sum(v for k, v in counter.items() if k[0] == p and match(k) and hit(k))
            n = tot
            vals.append(f"{good / tot:.2f}" if tot else "-")
        mark = "**" if n < 30 else "*" if n < 100 else ""
        return "-" if n == 0 else " / ".join(vals) + f" ({n}){mark}"

    print("### e1. A set is available (exactly one rank held twice or more; Yaniv not legal)\n")
    print("P(throws the set), by the rank's points X and the highest other card H:\n")
    rels = ["H >= 2X", "X < H < 2X", "H <= X"]
    rows = []
    for x in range(11):
        rows.append([f"{x}" + (" (jokers)" if x == 0 else "")]
                    + [three(sets, lambda k, x=x, rel=rel: k[1] == x and k[3] == rel, lambda k: k[6] == "set")
                       for rel in rels]
                    + [three(sets, lambda k, x=x: k[1] == x, lambda k: k[6] == "set")])
    rows.append(["any"] + [three(sets, lambda k, rel=rel: k[3] == rel, lambda k: k[6] == "set") for rel in rels]
                + [three(sets, lambda k: True, lambda k: k[6] == "set")])
    print(table(["pair of X points each"] + rels + ["any H"], rows))
    print("\nWhat is done instead: P(keeps the pair and throws another card) and P(breaks the pair):\n")
    rows = []
    for name, lo, hi in (("X 0-2", 0, 2), ("X 3-5", 3, 5), ("X 6-9", 6, 9), ("X 10", 10, 10)):
        rows.append([name]
                    + [three(sets, lambda k, lo=lo, hi=hi: lo <= k[1] <= hi, lambda k, w=w: k[6] == w)
                       for w in ("set", "keep", "break")]
                    + [three(sets, lambda k, lo=lo, hi=hi, o=o: lo <= k[1] <= hi and k[4] == o, lambda k: k[6] == "set")
                       for o in (True, False)])
    print(table(["pair", "throws the set", "keeps the pair, throws another card", "breaks the pair",
                 "throws the set when the opp has 5", "throws the set when the opp has <= 4"], rows))
    print("\nThree of a kind held (or four): P(throws the whole set): "
          + three(sets, lambda k: k[2] >= 3, lambda k: k[6] == "set") + "\n")
    print("### e2. Taking the open card (Yaniv not legal)\n")
    rows = []
    for pts in range(11):
        rows.append([str(pts)]
                    + [three(opens, lambda k, pts=pts: k[1] == pts and not k[2], lambda k: k[4]),
                       three(opens, lambda k, pts=pts: k[1] == pts and not k[2] and k[3], lambda k: k[4]),
                       three(opens, lambda k, pts=pts: k[1] == pts and not k[2] and not k[3], lambda k: k[4]),
                       three(opens, lambda k, pts=pts: k[1] == pts and k[2], lambda k: k[4]),
                       three(opens, lambda k, pts=pts: k[1] == pts and k[2], lambda k: k[4] and k[5])])
    print(table(["open card's points", "unpaired: takes it", "unpaired, opp has 5", "unpaired, opp has <= 4",
                 "its rank is in my hand: takes it", "... takes it and keeps the matching card (builds a set)"], rows))
    print("\n### e3. How often the players differ, by (my cards, opponent's cards)\n")
    rows = []
    for my_n, opp_n in ((5, 5), (5, 4), (5, 3), (5, 2), (5, 1), (4, 5), (3, 5), (2, 5), (1, 5)):
        row = [f"{my_n}, {opp_n}"]
        n = 0
        for a, b in ((1, 2), (0, 1), (0, 2)):
            same = cells[(a, b, my_n, opp_n, True)]
            diff = cells[(a, b, my_n, opp_n, False)]
            n = same + diff
            row.append(f"{diff / n:.1%}" if n else "-")
        rows.append(row[:1] + [str(n) + ("**" if n < 30 else "*" if n < 100 else "")] + row[1:])
    tot = {ab: [sum(v for k, v in cells.items() if k[:2] == ab and k[4] == flag) for flag in (False, True)]
           for ab in ((1, 2), (0, 1), (0, 2))}
    rows.append(["all", str(sum(tot[(1, 2)]))] + [f"{tot[ab][0] / sum(tot[ab]):.1%}" for ab in ((1, 2), (0, 1), (0, 2))])
    print(table(["my cards, opp cards", "positions", "hm4 differs from trained", "hm4 differs from counter",
                 "trained differs from counter"], rows))
    total = sum(kinds.values())
    print(f"\n### e4. The most common kinds of hm4-vs-trained disagreement ({total:,} disagreements)\n")
    for kind, n in kinds.most_common(9):
        print(f"- **{kind}**: {n} ({n / total:.1%})")
        for ex in examples[kind][:3]:
            print(f"  - {ex}")


# ---------------------------------------------------------------- f. what a card is worth
SNAPS = (4, 8, 12)


def _worth(job):
    n, seed = job
    rng = random.Random(seed)
    pick = random.Random(seed + 7)
    hm, counter = chooser_of("hm4"), chooser_of("counter")
    snaps_out, roll_out = [], []
    for i in range(n):
        seat = i % 2
        choosers = (hm, counter) if seat == 0 else (counter, hm)
        state, snaps = run(new_round([0, 0], (i // 2) % 2, rng), choosers, None, SNAPS)
        margin = state.result.added[1 - seat] - state.result.added[seat]
        for ply, snap in snaps.items():
            my_n, opp_n = popcount(snap.hands[seat]), popcount(snap.hands[1 - seat])
            snaps_out.append([ply, my_n, opp_n, mask_points(snap.hands[seat]), mask_points(snap.hands[1 - seat]),
                              1 if snap.turn == seat else 0, margin])
            for side, who in ((0, seat), (1, 1 - seat)):
                cards = cards_from_mask(snap.hands[who])
                if len(cards) < 2:
                    continue
                card = pick.choice(cards)
                alt = copy_state(snap)
                alt.hands[who] &= ~(1 << card)
                alt.known[who] &= ~(1 << card)
                end, _ = run(alt, choosers)
                new = end.result.added[1 - seat] - end.result.added[seat]
                lowest = POINTS_OF[card] == min(POINTS_OF[c] for c in cards)
                roll_out.append([ply, side, my_n, opp_n, POINTS_OF[card], new - margin, 1 if lowest else 0,
                                 1 if snap.turn == seat else 0])
    return np.array(snaps_out, dtype=np.float64), np.array(roll_out, dtype=np.float64)


def _pm(x: np.ndarray) -> str:
    if len(x) == 0:
        return "-"
    mark = "**" if len(x) < 30 else "*" if len(x) < 100 else ""
    return f"{x.mean():+.1f} ±{1.96 * x.std() / np.sqrt(len(x)):.1f} ({len(x)}){mark}"


def part_f(rounds: int) -> None:
    per = 500
    jobs = [(min(per, rounds - s), 904_000 + k) for k, s in enumerate(range(0, rounds, per))]
    with Pool(PROCESSES) as pool:
        parts = pool.map(_worth, jobs, chunksize=1)
    s = np.concatenate([p[0] for p in parts])
    r = np.concatenate([p[1] for p in parts])
    print(f"## f. What a card is worth\n\n`python -m training.yaniv.handmade5_measure f --rounds {rounds}`: "
          f"{rounds:,} rounds of hm4 vs counter from 0-0 (seeds 904000+). Result = points the counter adds minus "
          f"points I add, at the end of the round (overall mean {s[s[:, 0] == 4, 6].mean():+.2f}). Cells: mean "
          f"±95% range (samples); `*` n<100, `**` n<30. A round's result has a standard deviation of about "
          f"{s[s[:, 0] == 4, 6].std():.0f} points, so cells with under ~1,000 samples are only good to +-1.\n")
    for ply in SNAPS:
        d = s[s[:, 0] == ply]
        print(f"### f1. Mean final result by (my cards, opp cards) at ply {ply} ({len(d):,} rounds still going)\n")
        rows = []
        for a in (5, 4, 3, 2, 1):
            rows.append([str(a)] + [_pm(d[(d[:, 1] == a) & (d[:, 2] == b), 6]) for b in (5, 4, 3, 2, 1)])
        print(table(["my cards \\ opp cards", "5", "4", "3", "2", "1"], rows))
        print()
    print("### f2. Matched comparison: least-squares fit of the final result on cards and points held\n")
    print("result ~ const + a x my cards + b x my points + c x opp cards + d x opp points (+ I am to move). "
          "Coefficients ± 95% range. `a` is what one card MORE in my hand costs at equal points.\n")
    rows = []
    for ply in SNAPS:
        d = s[s[:, 0] == ply]
        for label, sel in [("all", d[:, 2] > 0)] + [(f"opp has {b}", d[:, 2] == b) for b in (5, 4, 3, 2)]:
            e = d[sel]
            if len(e) < 200:
                continue
            cols = [np.ones(len(e)), e[:, 1], e[:, 3], e[:, 4], e[:, 5]] + ([e[:, 2]] if label == "all" else [])
            x = np.stack(cols, axis=1)
            beta, *_ = np.linalg.lstsq(x, e[:, 6], rcond=None)
            resid = e[:, 6] - x @ beta
            cov = np.linalg.inv(x.T @ x) * resid.var()
            se = 1.96 * np.sqrt(np.diag(cov))
            rows.append([str(ply), label, str(len(e)), f"{beta[1]:+.2f} ±{se[1]:.2f}", f"{beta[2]:+.2f} ±{se[2]:.2f}",
                         f"{beta[5]:+.2f} ±{se[5]:.2f}" if label == "all" else "-",
                         f"{beta[3]:+.2f} ±{se[3]:.2f}", f"{beta[4]:+.2f} ±{se[4]:.2f}"])
    print(table(["ply", "rounds", "n", "per card of mine", "per point of mine", "per opp card", "per opp point",
                 "I am to move"], rows))
    print("\n### f3. Rollouts: delete one card and play the round out again (same deck order, same players)\n")
    print("At plies 4, 8 and 12 one random card is deleted from my hand (or from the counter's), and the rest of "
          "the round is replayed; the cell is the mean change of my final result. A deleted 0-2 point card is "
          "close to the pure value of holding one card fewer.\n")
    bands = (("0-2", 0, 2), ("3-5", 3, 5), ("6-9", 6, 9), ("10", 10, 10))
    for side, title in ((0, "One of MY cards deleted: my gain, by the deleted card's points and the opponent's "
                            "card count"),
                        (1, "One of the COUNTER's cards deleted: my change (negative = my loss), by the deleted "
                            "card's points and MY card count")):
        d = r[r[:, 1] == side]
        print(f"**{title}** (plies pooled; {len(d):,} rollouts)\n")
        col = 3 if side == 0 else 2
        rows = []
        for name, lo, hi in bands:
            sel = (d[:, 4] >= lo) & (d[:, 4] <= hi)
            rows.append([name] + [_pm(d[sel & (d[:, col] == b), 5]) for b in (5, 4, 3, 2, 1)] + [_pm(d[sel, 5])])
        rows.append(["any"] + [_pm(d[d[:, col] == b, 5]) for b in (5, 4, 3, 2, 1)] + [_pm(d[:, 5])])
        other = "opp" if side == 0 else "my"
        print(table(["deleted card"] + [f"{other} cards {b}" for b in (5, 4, 3, 2, 1)] + ["any"], rows))
        print("\nBy ply and by the hand's own card count before the deletion (deleted card worth 0-2):\n")
        own = 2 if side == 0 else 3
        low = d[d[:, 4] <= 2]
        rows = []
        for ply in SNAPS:
            rows.append([str(ply)] + [_pm(low[(low[:, 0] == ply) & (low[:, own] == b), 5]) for b in (5, 4, 3, 2)]
                        + [_pm(low[low[:, 0] == ply, 5])])
        print(table(["ply"] + [f"hand had {b}" for b in (5, 4, 3, 2)] + ["any"], rows))
        x = np.stack([np.ones(len(d)), d[:, 4]], axis=1)
        beta, *_ = np.linalg.lstsq(x, d[:, 5], rcond=None)
        print(f"\nStraight-line fit over all deleted cards: change = {beta[0]:+.2f} {beta[1]:+.2f} x the card's "
              f"points.\n")


# ---------------------------------------------------------------- g. match-score units
def build_without_score_units(path: Path) -> Path:
    """handmade.build() with add_score_units left out."""
    from . import handmade
    from .handmade_kit import Kit
    from .network import save_net

    kit = Kit()
    handmade.add_throw_units(kit)
    reasons = handmade.add_yaniv_units(kit)
    handmade.finish_yaniv(kit, reasons)
    handmade.fold_output(kit)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_net(kit.build(), path, meta={"handmade": True, "note": "hm4 without add_score_units (measuring only)"})
    return path


def _cut_matches(job):
    kind, n, seed, first = job
    rng = random.Random(seed)
    hero, counter = chooser_of(kind), chooser_of("counter")
    out = []
    for i in range(n):
        seat = (first + i) % 2
        choosers = (hero, counter) if seat == 0 else (counter, hero)
        scores, leader, rounds = [0, 0], 0, 0
        cuts = [[0, 0], [0, 0]]   # [mine, counter's] x [at 50, at 100]
        how = [0, 0, 0]           # my cuts: I was called on / my own caught call / showdown
        while (max(scores) < 100 and rounds < 40) or (scores[0] == scores[1] and rounds < 43 and max(scores) >= 100):
            state, _ = run(new_round(scores, leader, rng), choosers)
            res = state.result
            for who in (0, 1):
                total = scores[who] + res.added[who]
                if total in SCORE_CUTS and res.added[who] > 0:
                    cuts[0 if who == seat else 1][0 if total == 50 else 1] += 1
                    if who == seat:
                        how[2 if res.caller is None else 1 if res.caller == seat else 0] += 1
            scores = state.scores
            leader = 1 - leader
            rounds += 1
        out.append([1 if scores[seat] < scores[1 - seat] else 0, 1 if scores[seat] > scores[1 - seat] else 0, rounds,
                    cuts[0][0], cuts[0][1], cuts[1][0], cuts[1][1]] + how)
    return np.array(out, dtype=np.float64)


def part_g(matches: int, scratch: str) -> None:
    scratch_dir = Path(scratch) if scratch else Path.cwd()
    bare = build_without_score_units(scratch_dir / "hm4-no-score-units.json")
    print(f"## g. Do the match-score units pay in the real game?\n\n`python -m training.yaniv.handmade5_measure g "
          f"--matches {matches} --scratch DIR` builds hm4 without `add_score_units` and plays {matches:,} matches "
          f"to 100 of each player against the counter on the same deals (seeds 905000+). A cut = a round after "
          f"which a score lands exactly on 50 or 100 (and drops by 50).\n")
    per = 250
    rows = []
    for kind, name in (("hm4", "hm4 (with score units)"), (str(bare), "hm4 without score units"),
                       ("counter", "counter (vs counter)")):
        jobs = [(kind, min(per, matches - s), 905_000 + k, s) for k, s in enumerate(range(0, matches, per))]
        with Pool(PROCESSES) as pool:
            m = np.concatenate(pool.map(_cut_matches, jobs, chunksize=1))
        n = len(m)
        win = m[:, 0].mean()
        rows.append([name, f"{int(m[:, 0].sum())}-{int(m[:, 1].sum())} ({win:.1%} ±{1.96 * np.sqrt(win * (1 - win) / n):.1%})",
                     f"{m[:, 2].mean():.1f}", f"{m[:, 3].mean():.3f}", f"{m[:, 4].mean():.3f}",
                     f"{(m[:, 3] + m[:, 4]).mean():.3f}", f"{(m[:, 5] + m[:, 6]).mean():.3f}",
                     f"{((m[:, 3] + m[:, 4]) > 0).mean():.1%}",
                     f"{m[:, 7].sum():.0f} / {m[:, 8].sum():.0f} / {m[:, 9].sum():.0f}"])
    print(table(["player", "matches won-lost", "rounds a match", "my cuts at 50 a match", "my cuts at 100 a match",
                 "my cuts a match", "the counter's cuts a match", "matches with a cut of mine",
                 "my cuts: called on / my own caught call / showdown"], rows))


def main() -> None:
    if not STANDARD_RULES:
        raise SystemExit("run under the real game's rules (YANIV_HAND_SIZE / YANIV_CALL_AT unset)")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("part", choices=("part1", "e", "f", "g"))
    parser.add_argument("--rounds", type=int, default=0)
    parser.add_argument("--matches", type=int, default=3000)
    parser.add_argument("--scratch", default="", help="where part g saves the net without score units")
    args = parser.parse_args()
    if args.part == "part1":
        part1(args.rounds or 40000)
    elif args.part == "e":
        part_e(args.rounds or 6000)
    elif args.part == "f":
        part_f(args.rounds or 40000)
    else:
        part_g(args.matches, args.scratch)


if __name__ == "__main__":
    sys.exit(main())
