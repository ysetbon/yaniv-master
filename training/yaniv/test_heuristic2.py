"""counter2 (heuristic2.py): always legal, the same move for the same view, and blind to hidden state."""

from __future__ import annotations

import random
import unittest
from dataclasses import replace

from training.yaniv.engine import (HAND_SIZE, MAX_PLIES, YANIV_LIMIT, State, View, apply_action, legal_actions,
                                   make_view, mask_points, new_round, play_match, play_round, popcount)
from training.yaniv.heuristic import choose_heuristic, choose_random
from training.yaniv.heuristic2 import CAP, beliefs, choose_counter2, evaluate, opp_total_cdf, _read
from training.yaniv.parallel import bot_spec, chooser


def positions(rounds: int, seed: int, opponents=(choose_heuristic, choose_random, choose_counter2)):
    """(state, view, actions) for every decision counter2 meets in `rounds` rounds against mixed opponents."""
    rng = random.Random(seed)
    for i in range(rounds):
        state = new_round([rng.randrange(100), rng.randrange(100)], i % 2, rng)
        opponent = opponents[i % len(opponents)]
        me = (i // 2) % 2
        while not state.terminal and state.ply < MAX_PLIES:
            view = make_view(state, state.turn)
            actions = legal_actions(view)
            if not actions:
                break
            if state.turn == me:
                yield state, view, actions
                action = choose_counter2(view, actions, rng)
            else:
                action = opponent(view, actions, rng)
            apply_action(state, action)


class Counter2Test(unittest.TestCase):
    def test_registered_as_a_bot(self) -> None:
        self.assertIs(chooser(bot_spec("counter2")), choose_counter2)
        self.assertIs(chooser(bot_spec("counter")), choose_heuristic)

    def test_always_a_legal_action(self) -> None:
        seen = 0
        for _, view, actions in positions(150, 11):
            self.assertIn(choose_counter2(view, actions, None), actions)
            seen += 1
        self.assertGreater(seen, 500)

    def test_same_view_same_move_without_random_numbers(self) -> None:
        class NoRandom:
            def __getattr__(self, name):
                raise AssertionError(f"counter2 asked the random generator for {name}")

        for _, view, actions in positions(60, 12):
            first = choose_counter2(view, actions, NoRandom())
            again = choose_counter2(replace(view), list(actions), random.Random(99))
            self.assertEqual(first, again)

    def test_never_reads_hidden_state(self) -> None:
        """Re-deal every card counter2 cannot see (the opponent's hidden cards and the deck): same view, same move.
        And it is handed only a View, never the State."""
        rng = random.Random(13)
        checked = 0
        for state, view, actions in positions(60, 13):
            self.assertIsInstance(view, View)
            me = state.turn
            opp_hidden = state.hands[1 - me] & ~state.known[1 - me]
            pool = [c for c in range(54) if opp_hidden >> c & 1] + list(state.stock)
            rng.shuffle(pool)
            hidden_n = popcount(opp_hidden)
            hands = list(state.hands)
            hands[1 - me] = state.known[1 - me]
            for card in pool[:hidden_n]:
                hands[1 - me] |= 1 << card
            other = State(hands=hands, known=list(state.known), stock=pool[hidden_n:], discard=list(state.discard),
                          scores=list(state.scores), turn=state.turn, ply=state.ply)
            other_view = make_view(other, me)
            self.assertEqual(other_view, view)
            self.assertEqual(choose_counter2(other_view, legal_actions(other_view), None),
                             choose_counter2(view, actions, None))
            checked += 1
        self.assertGreater(checked, 200)

    def test_the_view_is_not_changed(self) -> None:
        for _, view, actions in positions(20, 14):
            before = (replace(view), list(actions))
            choose_counter2(view, actions, None)
            self.assertEqual((view, actions), before)

    def test_calls_only_when_legal_and_always_when_it_cannot_be_caught(self) -> None:
        from training.yaniv.features import opp_min_points

        sure = 0
        for _, view, actions in positions(300, 15):
            action = choose_counter2(view, actions, None)
            mine = mask_points(view.my_hand)
            if action.kind == "yaniv":
                self.assertLessEqual(mine, YANIV_LIMIT)
            if mine <= YANIV_LIMIT and mine < opp_min_points(view):
                self.assertEqual(action.kind, "yaniv")
                sure += 1
        self.assertGreater(sure, 5)

    def test_opponent_total_is_a_distribution_and_exact_when_all_is_known(self) -> None:
        for state, view, _ in positions(40, 16):
            cdf, _, _ = opp_total_cdf(view, _read(view)[3])
            self.assertEqual(len(cdf), CAP + 1)
            self.assertTrue(all(-1e-9 <= a <= b + 1e-9 <= 1.0 + 2e-9 for a, b in zip(cdf, cdf[1:])))
            theirs = mask_points(state.hands[1 - state.turn])
            if theirs <= CAP:                              # the true total is never given no chance
                self.assertGreater(cdf[theirs], 0.0)
            if view.opp_size == popcount(view.opp_known):  # every card known: no doubt left
                self.assertEqual(cdf, [1.0 if t >= theirs else 0.0 for t in range(CAP + 1)])
            self.assertTrue(0.0 <= beliefs(view)["hazard"] <= 1.0)

    def test_plays_whole_matches_and_handles_an_empty_deck(self) -> None:
        rng = random.Random(17)
        for _ in range(3):
            scores = play_match(choose_counter2, choose_heuristic, rng)
            self.assertEqual(len(scores), 2)
        state = new_round([0, 0], 0, rng)
        state.stock = []                                   # only the open card can be drawn
        view = make_view(state, 0)
        actions = legal_actions(view)
        self.assertEqual(choose_counter2(view, actions, None).draw, "open")
        self.assertIsNotNone(evaluate(view)[1])

    def test_beats_the_old_counter(self) -> None:
        """A coarse check that the rules are wired up: 300 mirrored rounds (the real margin is in results/counter2.md)."""
        from training.yaniv.train import round_duel

        result = round_duel(choose_counter2, choose_heuristic, 300, 2024)
        self.assertGreater(result["villain_avg"] - result["hero_avg"], 0.0 if HAND_SIZE == 5 else -1.0)
        self.assertTrue(play_round(choose_counter2, choose_counter2, random.Random(1), [0, 0], 0).terminal)


if __name__ == "__main__":
    unittest.main()
