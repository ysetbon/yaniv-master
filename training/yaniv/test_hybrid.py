"""The switch player (hybrid.py): the main net, then a smaller game's net once both hands are small."""

from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from .engine import Action, View, legal_actions, make_view, new_round, play_round, popcount
from .hybrid import compare, handover_share, small_enough, switch_chooser
from .network import default_net, save_net
from .parallel import chooser, net_spec, switch_spec


def _labelled(name: str, calls: list):
    def choose(view: View, actions: list[Action], rng):
        calls.append((name, popcount(view.my_hand), view.opp_size))
        # Throw the biggest set, so hands shrink and the hand-over comes.
        return max((a for a in actions if a.kind == "play"), key=lambda a: popcount(a.discard_mask))

    return choose


class Switch(unittest.TestCase):
    def test_it_hands_over_once_both_hands_are_small_and_never_back(self):
        calls: list = []
        player = switch_chooser(_labelled("main", calls), _labelled("end", calls), 4)
        for seed in range(30):
            play_round(player, player, random.Random(seed), [0, 0], seed % 2)
        self.assertTrue(any(name == "main" for name, *_ in calls))
        self.assertTrue(any(name == "end" for name, *_ in calls))
        for name, mine, theirs in calls:
            self.assertEqual(name == "end", mine <= 4 and theirs <= 4)

    def test_a_hand_never_grows(self):
        state = new_round([0, 0], 0, random.Random(3))
        sizes = [popcount(state.hands[0]), popcount(state.hands[1])]
        rng = random.Random(4)
        while not state.terminal and state.ply < 40:
            view = make_view(state, state.turn)
            plays = [a for a in legal_actions(view) if a.kind == "play"]
            if not plays:
                break
            from .engine import apply_action

            apply_action(state, rng.choice(plays))
            now = [popcount(state.hands[0]), popcount(state.hands[1])]
            self.assertTrue(all(b <= a for a, b in zip(sizes, now)))
            sizes = now

    def test_switch_specs_travel_to_workers(self):
        spec = switch_spec(net_spec(default_net(1)), net_spec(default_net(2)), 4)
        player = chooser(spec)
        view = make_view(new_round([0, 0], 0, random.Random(1)), 0)
        self.assertIn(player(view, legal_actions(view), random.Random(0)), legal_actions(view))
        self.assertFalse(small_enough(view, 4))

    def test_compare_reports_both_players_and_the_head_to_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            main, end = Path(tmp) / "main.json", Path(tmp) / "end.json"
            save_net(default_net(1), main)
            save_net(default_net(2), end, {"rules": {"hand": 4, "yaniv_at": 7}})
            out = compare(main, end, rounds=4, matches=2, workers=1, log=lambda _: None)
        self.assertEqual(set(out), {"handover", "main alone", "switch", "head to head"})
        self.assertGreater(out["handover"]["rounds_reaching"], 0)

    def test_handover_share_is_a_share(self):
        from .train import net_chooser

        share = handover_share(net_chooser(default_net(), 0.0), 20, 1)
        self.assertTrue(0 <= share["decisions_after"] <= share["rounds_reaching"] <= 1)


if __name__ == "__main__":
    unittest.main()
