"""The teacher test (teacher_test.py): same deals for every player, paired comparisons."""

from __future__ import annotations

import unittest

from .heuristic import choose_heuristic
from .match_value import win_table
from .network import default_net
from .parallel import Workers, chooser, net_spec, search_spec
from .teacher_test import play, scored_rounds, summarize
from .value import value_net


class TeacherTest(unittest.TestCase):
    def test_the_same_player_gets_the_same_deals_and_scores(self):
        table = win_table([(0, 10), (10, 0), (5, 5)])
        a = scored_rounds(choose_heuristic, choose_heuristic, 8, 3, table)
        b = scored_rounds(choose_heuristic, choose_heuristic, 8, 3, table)
        self.assertEqual(a, b)
        # Mirrored pairs: the same player in both seats of a deal takes the two seats' points.
        for (h1, v1, _), (h2, v2, _) in zip(a[0::2], a[1::2]):
            self.assertEqual((h1, v1), (v2, h2))

    def test_against_itself_the_gain_is_exactly_zero(self):
        table = win_table([(0, 10), (10, 0)])
        rows = scored_rounds(choose_heuristic, choose_heuristic, 6, 1, table)
        s = summarize(rows, rows)
        self.assertEqual((s["margin_gain"], s["chance_gain"], s["margin_se"]), (0.0, 0.0, 0.0))

    def test_search_players_travel_to_workers(self):
        net, table = default_net(), win_table([(0, 10), (10, 0)])
        pool = Workers(1)
        for spec in (net_spec(net), search_spec(net, None, 2, table=table),
                     search_spec(net, value_net(), 2, depth=1, table=table)):
            self.assertTrue(callable(chooser(spec)))
            rows = play(pool, spec, 2, 5, table)
            self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
