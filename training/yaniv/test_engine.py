"""Rules, feature layout, and a sanity check that the bench beats a greedy player."""

from __future__ import annotations

import random
import unittest

import numpy as np

from .engine import (
    ASSAF_PENALTY,
    POINTS_OF,
    YANIV_LIMIT,
    Action,
    View,
    apply_action,
    best_keep_points,
    cards_from_mask,
    is_valid_combo,
    legal_actions,
    make_view,
    mask_points,
    new_round,
    play_round,
    popcount,
)
from .features import ACTION_DIM, STATE_DIM, action_features, state_features
from .heuristic import choose_greedy, choose_heuristic
from .network import MLP


def brute_keep(mask: int) -> int:
    cards = cards_from_mask(mask)
    if not cards:
        return 0
    best = 10**9
    n = len(cards)
    for subset in range(1, 1 << n):
        chosen = [cards[i] for i in range(n) if subset & (1 << i)]
        if not is_valid_combo(chosen):
            continue
        rem = mask
        for card in chosen:
            rem &= ~(1 << card)
        best = min(best, mask_points(rem))
    return best


def conserved(state) -> bool:
    seen = 0
    for hand in state.hands:
        if seen & hand:
            return False
        seen |= hand
    if state.known[0] & ~state.hands[0]:
        return False
    if state.known[1] & ~state.hands[1]:
        return False
    for card in state.discard + state.stock:
        bit = 1 << card
        if seen & bit:
            return False
        seen |= bit
    return seen == (1 << 52) - 1


class RulesTest(unittest.TestCase):
    def test_points_and_runs(self):
        self.assertEqual(POINTS_OF[0], 1)
        self.assertEqual(POINTS_OF[8], 9)
        self.assertEqual(POINTS_OF[9], 10)
        self.assertEqual(POINTS_OF[12], 10)
        ace, two, three = 0, 1, 2
        self.assertTrue(is_valid_combo([ace, two, three]))
        king, queen, ace_spade = 12, 11, 13 * 3
        self.assertFalse(is_valid_combo([king, queen, ace_spade]))
        self.assertTrue(is_valid_combo([12, 11, 10]))
        self.assertTrue(is_valid_combo([0, 13]))
        self.assertFalse(is_valid_combo([0, 14]))
        self.assertTrue(is_valid_combo([0, 13, 26, 39]))
        self.assertFalse(is_valid_combo([1, 2]))

    def test_five_card_minimum_is_six(self):
        aces = (1 << 0) | (1 << 13) | (1 << 26) | (1 << 39)
        self.assertEqual(mask_points(aces | (1 << 1)), 6)
        self.assertGreater(mask_points(aces | (1 << 1)), YANIV_LIMIT)

    def test_best_keep_matches_brute(self):
        rng = random.Random(2)
        for _ in range(400):
            deck = list(range(52))
            rng.shuffle(deck)
            mask = 0
            for card in deck[: rng.randint(1, 5)]:
                mask |= 1 << card
            self.assertEqual(best_keep_points(mask), brute_keep(mask))

    def test_yaniv_assaf_and_whole_hand_throw(self):
        state = new_round([0, 0], 0, random.Random(1))
        state.hands = [(1 << 0) | (1 << 1), (1 << 12) | (1 << 11) | (1 << 10)]
        state.known = [0, 0]
        state.stock = [c for c in range(52) if not ((state.hands[0] | state.hands[1]) & (1 << c)) and c != 20]
        state.discard = [20]
        state.turn = 0
        state.ply = 0
        self.assertTrue(conserved(state))
        view = make_view(state, 0)
        actions = legal_actions(view)
        self.assertTrue(any(action.kind == "yaniv" for action in actions))
        apply_action(state, Action("yaniv"))
        self.assertTrue(state.result.assaf is False)
        self.assertEqual(state.result.added, [0, mask_points(state.result.hands[1])])
        self.assertEqual(state.result.winner, 0)

        state = new_round([4, 9], 1, random.Random(2))
        caller = (1 << 1) | (1 << 2)  # 2♣ + 3♣ = 5, not a run
        catcher = (1 << 13) | (1 << 14) | (1 << 27)  # A♦ + 2♦ + 2♥ = 5
        used = caller | catcher
        rest = [c for c in range(52) if not (used & (1 << c))]
        state.hands = [caller, catcher]
        state.known = [0, 0]
        state.discard = [rest[0]]
        state.stock = rest[1:]
        state.turn = 0
        state.scores = [4, 9]
        apply_action(state, Action("yaniv"))
        self.assertTrue(state.result.assaf)
        self.assertEqual(state.result.added[0], 5 + ASSAF_PENALTY)
        self.assertEqual(state.result.added[1], 0)
        self.assertEqual(state.result.winner, 1)
        self.assertEqual(state.scores, [4 + 35, 9])

        state = new_round([0, 0], 0, random.Random(3))
        run = (1 << 0) | (1 << 1) | (1 << 2)
        opp = (1 << 12) | (1 << 25)
        rest = [c for c in range(52) if c not in (0, 1, 2, 12, 25)]
        state.hands = [run, opp]
        state.known = [0, 0]
        state.discard = [rest[0]]
        state.stock = rest[1:]
        state.turn = 0
        # Throwing the whole hand still needs a draw, so the hand never empties.
        whole = [a for a in legal_actions(make_view(state, 0)) if a.kind == "play" and a.discard_mask == run]
        self.assertTrue(whole)
        self.assertTrue(all(a.draw in ("stock", "open") for a in whole))
        before = state.to_json()
        with self.assertRaises(ValueError):
            apply_action(state, Action("play", run, 2, "none"))
        self.assertEqual(state.to_json(), before)  # a rejected move changes nothing
        top_of_stock = state.stock[-1]
        apply_action(state, Action("play", run, 2, "stock"))
        self.assertFalse(state.terminal)
        self.assertEqual(state.hands[0], 1 << top_of_stock)
        self.assertEqual(state.turn, 1)
        self.assertTrue(conserved(state))

    def test_every_throw_draws_and_hands_never_empty(self):
        rng = random.Random(8)
        for game in range(60):
            state = new_round([0, 0], game % 2, rng)
            while not state.terminal and state.ply < 48:
                actions = legal_actions(make_view(state, state.turn))
                for action in actions:
                    if action.kind == "play":
                        self.assertIn(action.draw, ("stock", "open"))
                apply_action(state, rng.choice(actions))
                if not state.terminal:
                    self.assertGreater(state.hands[0], 0)
                    self.assertGreater(state.hands[1], 0)
                    self.assertTrue(conserved(state))
            if state.terminal:
                self.assertIn(state.result.reason, ("yaniv", "showdown"))

    def test_draw_open_tracks_known_card_and_top(self):
        clubs_king, diamonds_king, hearts_king = 12, 12 + 13, 12 + 26
        five, four, two = 4, 3, 1
        hand = (1 << clubs_king) | (1 << diamonds_king) | (1 << five) | (1 << four) | (1 << two)
        opp = (1 << 8) | (1 << 7) | (1 << 6) | (1 << 18) | (1 << 19)
        used = hand | opp | (1 << hearts_king)
        stock = [c for c in range(52) if not (used & (1 << c))]
        from .engine import State

        state = State(
            hands=[hand, opp],
            known=[0, 0],
            stock=stock,
            discard=[hearts_king],
            scores=[0, 0],
            turn=0,
            ply=2,
        )
        self.assertTrue(conserved(state))
        apply_action(state, Action("play", 1 << five, five, "open"))
        self.assertEqual(state.hands[0] & (1 << hearts_king), 1 << hearts_king)
        self.assertEqual(state.known[0], 1 << hearts_king)
        self.assertEqual(state.discard[-1], five)
        self.assertNotIn(hearts_king, state.discard)
        self.assertTrue(conserved(state))
        self.assertEqual(state.turn, 1)

    def test_empty_stock_forbids_blind_draw(self):
        state = new_round([0, 0], 0, random.Random(4))
        state.stock = []
        view = make_view(state, 0)
        actions = legal_actions(view)
        self.assertTrue(actions)
        self.assertTrue(all(action.draw != "stock" for action in actions))

    def test_random_games_conserve_cards(self):
        rng = random.Random(9)
        for _ in range(30):
            state = new_round([rng.randint(0, 40), rng.randint(0, 40)], rng.randrange(2), rng)
            from .engine import resolve_showdown

            while not state.terminal:
                self.assertTrue(conserved(state))
                if state.ply >= 48:
                    resolve_showdown(state)
                    break
                view = make_view(state, state.turn)
                actions = legal_actions(view)
                self.assertTrue(actions)
                apply_action(state, actions[rng.randrange(len(actions))])
            self.assertTrue(state.terminal)
            self.assertIsNotNone(state.result)

    def test_feature_layout_anchors(self):
        view = View(
            my_hand=1 << 0,
            my_known=0,
            opp_known=1 << 12,
            opp_size=2,
            stock_size=20,
            discard=(5, 9),
            my_score=10,
            opp_score=20,
            ply=4,
            seat=0,
        )
        vec = state_features(view)
        self.assertEqual(vec.shape, (STATE_DIM,))
        self.assertEqual(float(vec[0]), 1.0)
        self.assertEqual(float(vec[104 + 9]), 1.0)
        self.assertEqual(float(vec[156 + 12]), 1.0)
        self.assertAlmostEqual(float(vec[238]), 1 / 40, places=6)
        self.assertAlmostEqual(float(vec[247]), 1.0, places=6)
        self.assertAlmostEqual(float(vec[255]), 1.0, places=6)
        call = action_features(view, Action("yaniv"))
        self.assertEqual(call.shape, (ACTION_DIM,))
        self.assertEqual(float(call[106]), 1.0)
        self.assertEqual(float(call.sum()), 1.0)
        thrown = action_features(view, Action("play", 1 << 0, 0, "open"))
        self.assertEqual(float(thrown[0]), 1.0)
        self.assertEqual(float(thrown[52]), 1.0)
        self.assertEqual(float(thrown[105]), 1.0)
        self.assertEqual(float(thrown[107]), 0.0)

    def test_heuristic_takes_the_third_king(self):
        clubs, diamonds, hearts = 12, 12 + 13, 12 + 26
        five, four, two = 4, 3, 1
        hand = (1 << clubs) | (1 << diamonds) | (1 << five) | (1 << four) | (1 << two)
        opp = 0
        for card in (8, 7, 6, 18, 19):
            opp |= 1 << card
        used = hand | opp | (1 << hearts)
        stock = [c for c in range(52) if not (used & (1 << c))]
        from .engine import State

        state = State(
            hands=[hand, opp],
            known=[0, 0],
            stock=stock,
            discard=[hearts],
            scores=[0, 0],
            turn=0,
            ply=3,
        )
        view = make_view(state, 0)
        action = choose_heuristic(view, legal_actions(view))
        self.assertEqual(action.kind, "play")
        self.assertEqual(action.draw, "open")
        self.assertNotIn(action.top, (clubs, diamonds))

    def test_guaranteed_call_and_risky_call(self):
        from .engine import State

        # A A A 2 = 5, not one combination. Opponent still holds five cards (minimum 6).
        caller = (1 << 0) | (1 << 13) | (1 << 26) | (1 << 1)
        opp = 0
        for card in (8, 9, 10, 11, 12):
            opp |= 1 << card
        used = caller | opp
        rest = [c for c in range(52) if not (used & (1 << c))]
        state = State(
            hands=[caller, opp],
            known=[0, 0],
            stock=rest[1:],
            discard=[rest[0]],
            scores=[0, 0],
            turn=0,
            ply=6,
        )
        view = make_view(state, 0)
        action = choose_heuristic(view, legal_actions(view))
        self.assertEqual(action.kind, "yaniv")

        # A♠ 4♦ = 5, and the opponent is known to hold an ace plus one hidden card.
        mine = (1 << 0) | (1 << 16)  # A♣ + 4♦
        known_low = 1 << 14  # A♦
        hidden = 3  # 4♣
        used = mine | known_low | hidden
        rest = [c for c in range(52) if not (used & (1 << c))]
        state = State(
            hands=[mine, known_low | (1 << hidden)],
            known=[0, known_low],
            stock=rest[1:],
            discard=[rest[0]],
            scores=[0, 0],
            turn=0,
            ply=8,
        )
        view = make_view(state, 0)
        action = choose_heuristic(view, legal_actions(view))
        self.assertEqual(action.kind, "play")

    def test_bench_beats_greedy_and_random(self):
        rng = random.Random(11)
        wins = losses = 0
        for i in range(80):
            if i % 2 == 0:
                state = play_round(choose_heuristic, choose_greedy, rng, [0, 0], i % 2)
                mine, theirs = state.result.added
            else:
                state = play_round(choose_greedy, choose_heuristic, rng, [0, 0], i % 2)
                theirs, mine = state.result.added
            if mine < theirs:
                wins += 1
            elif theirs < mine:
                losses += 1
        self.assertGreater(wins, losses + 15)

        from .heuristic import choose_random

        wins = losses = 0
        for i in range(40):
            if i % 2 == 0:
                state = play_round(choose_heuristic, choose_random, rng, [0, 0], 0)
                mine, theirs = state.result.added
            else:
                state = play_round(choose_random, choose_heuristic, rng, [0, 0], 1)
                theirs, mine = state.result.added
            if mine < theirs:
                wins += 1
            elif theirs < mine:
                losses += 1
        self.assertGreater(wins, 30)


class GradTest(unittest.TestCase):
    def test_mlp_gradient_matches_finite_difference(self):
        net = MLP([6, 5, 1], seed=3)
        rng = np.random.default_rng(1)
        x = rng.standard_normal((5, 6)).astype(np.float32)
        target = rng.standard_normal((5, 1)).astype(np.float32)
        pred = net.forward(x, cache=True)
        dy = pred - target
        net.zero_grad()
        net.backward_acc(dy)
        analytic = float(net.gW[0][2, 1])
        eps = np.float32(1e-2)
        original = float(net.W[0][2, 1])
        net.W[0][2, 1] = original + eps
        plus = net.forward(x)
        net.W[0][2, 1] = original - eps
        minus = net.forward(x)
        net.W[0][2, 1] = original
        numerical = float(0.5 * np.sum((plus - target) ** 2) - 0.5 * np.sum((minus - target) ** 2)) / (2 * float(eps))
        self.assertAlmostEqual(analytic, numerical, delta=0.05)


if __name__ == "__main__":
    unittest.main()
