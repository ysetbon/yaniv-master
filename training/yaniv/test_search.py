"""The search teacher: sampled deals, values, labels, and a tiny search run."""

from __future__ import annotations

import dataclasses
import random
import unittest

from .engine import (
    DECK_SIZE,
    POINTS_OF,
    Action,
    State,
    apply_action,
    cards_from_mask,
    legal_actions,
    make_view,
    new_round,
)
from .heuristic import choose_heuristic
from .network import default_net
from .search import MATCH_BONUS, collect_search, counter_prior, deal_from_view, evaluate_moves, round_value
from .train import Observer, TrainConfig, run_training


def advanced_state(seed: int, plies: int) -> State:
    rng = random.Random(seed)
    state = new_round([0, 0], 0, rng)
    for _ in range(plies):
        view = make_view(state, state.turn)
        apply_action(state, choose_heuristic(view, legal_actions(view)))
        if state.terminal:
            break
    return state


class Deals(unittest.TestCase):
    def test_sampled_deal_matches_what_the_player_sees(self):
        for seed in range(20):
            state = advanced_state(seed, 6)
            if state.terminal:
                continue
            view = make_view(state, state.turn)
            deal = deal_from_view(view, random.Random(seed))
            me, opp = view.seat, 1 - view.seat
            self.assertEqual(deal.hands[me], state.hands[me])
            self.assertEqual(deal.hands[opp] & view.opp_known, view.opp_known)
            self.assertEqual(deal.hands[opp].bit_count(), state.hands[opp].bit_count())
            self.assertEqual(len(deal.stock), len(state.stock))
            self.assertEqual(deal.discard, state.discard)
            seen = deal.hands[0] | deal.hands[1]
            for card in deal.stock + deal.discard:
                self.assertFalse(seen & (1 << card))
                seen |= 1 << card
            self.assertEqual(seen, (1 << DECK_SIZE) - 1)

    def test_hidden_cards_lean_low_later_in_the_round(self):
        state = new_round([0, 0], 0, random.Random(3))
        state.ply = 12
        view = make_view(state, 0)
        rng = random.Random(1)
        known = view.opp_known
        total = count = 0
        for _ in range(400):
            hidden = deal_from_view(view, rng).hands[1] & ~known
            for card in cards_from_mask(hidden):
                total += POINTS_OF[card]
                count += 1
        pool_avg = sum(POINTS_OF[c] for c in range(DECK_SIZE) if not (view.my_hand >> c) & 1
                       and c not in view.discard) / (DECK_SIZE - 5 - len(view.discard))
        self.assertLess(total / count, pool_avg - 1.0)


class Values(unittest.TestCase):
    def test_round_value_is_margin_plus_match_bonus(self):
        state = new_round([10, 20], 0, random.Random(1))
        state.hands = [1 << 0, 1 << 9]  # ace against a ten
        from .engine import resolve_showdown

        resolve_showdown(state)
        self.assertEqual(round_value(state, 0), 10.0)
        self.assertEqual(round_value(state, 1), -10.0)
        state = new_round([20, 95], 0, random.Random(1))
        state.hands = [1 << 0, 1 << 9]
        resolve_showdown(state)  # 95 + 10 = 105 ends the match
        self.assertEqual(round_value(state, 0), 10.0 + MATCH_BONUS)
        self.assertEqual(round_value(state, 1), -10.0 - MATCH_BONUS)

    def test_a_sure_yaniv_beats_a_hopeless_throw(self):
        mine, theirs = 1 << 52, (1 << 9) | (1 << 22) | (1 << 35)  # a joker against three tens
        rest = [c for c in range(DECK_SIZE) if not ((mine | theirs) >> c) & 1]
        random.Random(2).shuffle(rest)
        state = State(hands=[mine, theirs], known=[0, 0], stock=rest[1:], discard=rest[:1], scores=[0, 0],
                      turn=0, ply=6)
        view = make_view(state, 0)
        actions = legal_actions(view)
        candidates = counter_prior(view, actions)[:2] + [Action("yaniv")]
        values = evaluate_moves(view, candidates, choose_heuristic, 8, random.Random(0))
        self.assertEqual(int(values.argmax()), len(candidates) - 1)


class Labels(unittest.TestCase):
    def test_collected_labels_index_legal_moves(self):
        rows = collect_search(default_net(5), 2, 4, samples=4, width=3)
        self.assertTrue(rows)
        for view, label in rows:
            self.assertLess(label, len(legal_actions(view)))


class EndToEnd(unittest.TestCase):
    def test_tiny_search_run(self):
        cfg = dataclasses.replace(TrainConfig.search(quick=True), workers=1, cache=False, search_rounds=2,
                                  search_samples=4, eval_rounds=6, eval_matches=2, gate_rounds=6, gate_matches=2,
                                  final_rounds=6, final_matches=2)
        report = run_training(cfg, Observer(), model_path=None, report_path=None)
        self.assertEqual(report["recipe"], "search")
        searched = [h for h in report["history"] if h["phase"] == "search"]
        self.assertEqual(len(searched), 2)
        self.assertTrue(all(h["searched"] > 0 for h in searched))
        self.assertIn(report["best_generation"], (0, 1, 2))


if __name__ == "__main__":
    unittest.main()
