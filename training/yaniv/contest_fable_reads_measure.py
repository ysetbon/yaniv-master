"""Thread `reads`: plain logged games (no fitting). At each of the hero's decisions record what the 125 inputs
say about the opponent and the truth (its real total, whether it calls on its next turn, whether it ends the
round before the hero's second next turn).

    python -m training.yaniv.contest_fable_reads_measure HERO OPP ROUNDS SEED OUT.npz [workers]
"""

from __future__ import annotations

import random
import sys

import numpy as np

COLS = ["round", "ply", "oc", "mc", "est", "opp_min", "known_n", "known_pts", "open_pts", "open_rank",
        "open_gone", "open_unseen", "open_mine", "stock", "my_pts", "my_low", "unseen_low", "unseen_n",
        "unseen_avg", "hero_calls", "first",           # first: the open card is not an opponent's throw (ply 0)
        "opp_last_n", "opp_last_pts", "opp_last_draw_open",   # truth about the opponent's last move (-1: none)
        "opp_true", "opp_hidden_pts", "opp_low_true",          # truth: its total, its hidden cards' points
        "e1", "h1", "e2", "hero_pays", "opp_calls_total",      # e1: opp calls on its next turn; h1: hero calls on
        "my_last_pts", "my_last_taken", "opp_true_after",      # its next; e2: opp calls on its 1st or 2nd next turn
        "k_after", "take_open", "n_thrown", "pts_kept", "pts_after", "my_known_n", "my_known_pts",   # after my move:
        "my_min_after"]   # my cards, did I take the open card, my points (truth), my cards the opponent knows


def _job(job):
    from .contest_fable_lab import player
    from .engine import (POINTS_OF, apply_action, cards_from_mask, legal_actions, make_view, mask_points, new_round,
                         popcount, rank_of, resolve_showdown)
    from .features import state_features

    hero_name, opp_name, n, seed = job
    hero, opp = player(hero_name), player(opp_name)
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        hs = i % 2
        st = new_round([0, 0], (i // 2) % 2, rng)
        mine = []                    # this round's rows (indices into rows)
        opp_last = (-1, -1, -1)
        my_last = (-1, -1)           # (points of the card I left on top, index of my row)
        opp_turns = 0
        while not st.terminal and st.ply < 48:
            v = make_view(st, st.turn)
            acts = legal_actions(v)
            if not acts:
                break
            if st.turn != hs:
                a = opp(v, acts, None)
                opp_turns += 1
                # resolve e1 / e2 for the hero rows that are waiting
                for idx, at in mine:
                    k = opp_turns - at          # 1: the opponent's next turn after that hero decision
                    if a.kind == "yaniv":
                        if k == 1:
                            rows[idx][COLS.index("e1")] = 1
                        if k <= 2:
                            rows[idx][COLS.index("e2")] = 1
                if a.kind == "yaniv":
                    for idx, at in mine:
                        rows[idx][COLS.index("opp_calls_total")] = mask_points(st.hands[1 - hs])
                else:
                    opp_last = (popcount(a.discard_mask), mask_points(a.discard_mask), int(a.draw == "open"))
                    if my_last[1] >= 0:
                        rows[my_last[1]][COLS.index("my_last_taken")] = int(a.draw == "open")
                apply_action(st, a)
                continue
            f = state_features(v)
            a = hero(v, acts, None)
            if mine:                                     # my previous decision: did I call on my next turn?
                rows[mine[-1][0]][COLS.index("h1")] = int(a.kind == "yaniv")
            open_card = v.discard[-1]
            r = rank_of(open_card)
            hand = cards_from_mask(v.my_hand)
            opp_hand = st.hands[1 - hs]
            row = [i, v.ply, v.opp_size, len(hand), float(f[94]) * 40, float(f[85]) * 40, popcount(v.opp_known),
                   mask_points(v.opp_known), POINTS_OF[open_card], r,
                   round(float(f[14 + r]) * 4), round(float(f[56 + r]) * 4), round(float(f[r]) * 4),
                   v.stock_size, mask_points(v.my_hand), sum(1 for c in hand if POINTS_OF[c] <= 3),
                   float(f[78]), round(float(f[76]) * 40), float(f[77]) * 10, int(a.kind == "yaniv"),
                   int(opp_turns == 0), *opp_last,
                   mask_points(opp_hand), mask_points(opp_hand & ~v.opp_known),
                   sum(1 for c in cards_from_mask(opp_hand) if POINTS_OF[c] <= 3),
                   0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            rows.append(row)
            mine.append((len(rows) - 1, opp_turns))
            if a.kind != "yaniv":
                my_last = (POINTS_OF[a.top], len(rows) - 1)
                row[COLS.index("my_last_pts")] = POINTS_OF[a.top]
            if a.kind != "yaniv":
                row[COLS.index("n_thrown")] = popcount(a.discard_mask)
                row[COLS.index("take_open")] = int(a.draw == "open")
                row[COLS.index("pts_kept")] = mask_points(v.my_hand & ~a.discard_mask)
            apply_action(st, a)
            if not st.terminal:
                row[COLS.index("opp_true_after")] = mask_points(st.hands[1 - hs])
                row[COLS.index("k_after")] = popcount(st.hands[hs])
                row[COLS.index("pts_after")] = mask_points(st.hands[hs])
                row[COLS.index("my_known_n")] = popcount(st.known[hs])
                row[COLS.index("my_known_pts")] = mask_points(st.known[hs])
        if not st.terminal:
            resolve_showdown(st)
        for idx, _ in mine:
            rows[idx][COLS.index("hero_pays")] = st.result.added[hs] - st.result.added[1 - hs]
    return rows


def measure(hero: str, opp: str, rounds: int, seed: int, workers: int = 2) -> np.ndarray:
    from .contest_fable_lab import _pool

    size = max(2, rounds // (workers * 8) // 2 * 2)
    jobs = [(hero, opp, min(size, rounds - at), seed * 7919 + i) for i, at in enumerate(range(0, rounds, size))]
    with _pool(workers) as pool:
        parts = list(pool.map(_job, jobs))
    out = []
    for k, part in enumerate(parts):
        for r in part:
            r[0] += k * size
        out.extend(part)
    return np.array(out, dtype=np.float32)


if __name__ == "__main__":
    hero_, opp_, rounds_, seed_, path = sys.argv[1:6]
    data = measure(hero_, opp_, int(rounds_), int(seed_), int(sys.argv[6]) if len(sys.argv) > 6 else 2)
    np.savez_compressed(path, data=data, cols=np.array(COLS))
    print(f"{hero_} vs {opp_}: {int(data[:, 0].max()) + 1} rounds, {len(data)} hero decisions -> {path}")
