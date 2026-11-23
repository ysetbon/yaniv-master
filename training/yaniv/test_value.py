"""The value net (value.py), the search that asks it, and the smaller game (--hand, --yaniv-at)."""

from __future__ import annotations

import dataclasses
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .engine import Action, apply_action, legal_actions, make_view, new_round
from .features import STATE_DIM
from .network import default_net
from .search import playout, round_value, sample_values
from .train import Observer, TrainConfig, net_chooser, run_training
from .value import VALUE_DIM, collect_value, explained, predict, train_value, value_features, value_net

ROOT = Path(__file__).resolve().parents[2]


class ValueNet(unittest.TestCase):
    def test_features_see_both_hands_and_whose_turn(self):
        state = new_round([0, 0], 0, random.Random(1))
        mine, theirs = value_features(state, 0), value_features(state, 1)
        self.assertEqual(mine.shape, (VALUE_DIM,))
        self.assertEqual((mine[-1], theirs[-1]), (1.0, 0.0))
        # From the other side the two views swap places.
        np.testing.assert_array_equal(mine[:STATE_DIM], theirs[STATE_DIM:2 * STATE_DIM])

    def test_self_play_data_holds_how_rounds_ended(self):
        x, y = collect_value(default_net(), 12, 3, match_scores=False, explore=0.0)
        self.assertEqual(x.shape[1], VALUE_DIM)
        self.assertEqual(len(x), len(y))
        self.assertEqual(len(y) % 2, 0)
        # Without a table a target is the point margin / 100, and the two players' rows are opposite.
        np.testing.assert_allclose(y[0::2], -y[1::2], atol=1e-6)

    def test_with_a_match_table_targets_are_win_chances(self):
        from .match_value import win_table

        table = win_table([(0, 10), (10, 0), (5, 5)])
        _, y = collect_value(default_net(), 8, 4, table=table, explore=0.0)
        self.assertTrue(np.all((y >= 0) & (y <= 1)))

    def test_it_learns_and_says_how_well(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((2000, VALUE_DIM)).astype(np.float32)
        y = (0.3 * x[:, 0] - 0.2 * x[:, 5]).astype(np.float32)
        net = value_net()
        before = explained(net, x, y)
        train_value(net, x, y, epochs=8, lr=2e-3)
        self.assertGreater(explained(net, x, y), max(0.8, before))


class ValueSearch(unittest.TestCase):
    def test_a_playout_can_stop_after_a_few_moves(self):
        state = new_round([0, 0], 0, random.Random(2))
        playout(state, net_chooser(default_net(), 0.0), random.Random(0), limit=3)
        self.assertTrue(state.terminal or state.ply == 3)

    def test_the_search_asks_the_value_net_and_scores_endings_exactly(self):
        rng = random.Random(5)
        state = new_round([0, 0], 0, rng)
        view = make_view(state, 0)
        plays = [a for a in legal_actions(view) if a.kind == "play"][:2]
        vnet = value_net()
        policy = net_chooser(default_net(), 0.0)
        values = sample_values(view, plays, policy, 3, random.Random(9), value=vnet, depth=0)
        # Depth 0: each value is the value net's opinion right after the move, on the same imagined deals.
        again = random.Random(9)
        from .search import _copy, deal_from_view

        expected = []
        for _ in range(3):
            deal = deal_from_view(view, again)
            row = []
            for action in plays:
                s = _copy(deal)
                apply_action(s, action)
                row.append(predict(vnet, [s], 0)[0])
            expected.append(row)
        np.testing.assert_allclose(values, np.array(expected), rtol=1e-5, atol=1e-4)
        # A Yaniv call ends the round, so it is scored exactly, not by the value net.
        from .heuristic import choose_heuristic

        low = None
        for seed in range(50):
            game = new_round([0, 0], 0, random.Random(seed))
            while not game.terminal and low is None:
                seen = make_view(game, game.turn)
                actions = legal_actions(seen)
                if any(a.kind == "yaniv" for a in actions):
                    low = seen
                    break
                move = choose_heuristic(seen, actions)
                apply_action(game, move if move.kind == "play" else next(a for a in actions if a.kind == "play"))
            if low is not None:
                break
        call = Action("yaniv")
        exact = _copy(deal_from_view(low, random.Random(1)))
        apply_action(exact, call)
        got = sample_values(low, [call], policy, 1, random.Random(1), value=vnet, depth=2)[0, 0]
        self.assertEqual(got, round_value(exact, low.seat))

    def test_a_value_net_run_trains_it_and_records_it(self):
        cfg = dataclasses.replace(TrainConfig.search(quick=True), workers=1, cache=False, generations=2,
                                  value_net=True, value_rounds=6, search_rounds=2, search_samples=4, eval_rounds=4,
                                  eval_matches=2, gate_rounds=4, gate_matches=2, final_rounds=4, final_matches=2)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, results = Path(tmp) / "20990101-000001", Path(tmp) / "results"
            report = run_training(cfg, Observer(), model_path=None, report_path=None, run_dir=run_dir,
                                  results_dir=results)
            log = (run_dir / "log.txt").read_text()
            summary = (run_dir / "summary.md").read_text()
            self.assertTrue((run_dir / "value-01.json").exists())
            self.assertIn("Value net:", log)
            self.assertIn("then the value net judges", summary)
            self.assertIn("Value net fit", summary)
        searched = [h for h in report["history"] if h["phase"] == "search"]
        self.assertEqual(len(searched), 2)


class SmallGame(unittest.TestCase):
    def _run(self, code: str, **env) -> dict:
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True,
                             env={**os.environ, **env})
        return json.loads(out.stdout.strip().splitlines()[-1])

    def test_four_cards_and_yaniv_at_nine(self):
        code = (
            "import json, random\n"
            "from training.yaniv.engine import *\n"
            "from training.yaniv import train\n"
            "s = new_round([0, 0], 0, random.Random(1))\n"
            "v = make_view(s, 0)\n"
            "nine = make_view(s, 0).__class__(**{**v.__dict__, 'my_hand': (1 << 8)})\n"
            "print(json.dumps({'hands': [popcount(h) for h in s.hands], 'stock': len(s.stock), "
            "'nine': any(a.kind == 'yaniv' for a in legal_actions(nine)), 'suffix': RULES_SUFFIX, "
            "'model': train.MODEL_PATH.name}))\n"
        )
        got = self._run(code, YANIV_HAND_SIZE="4", YANIV_CALL_AT="9")
        self.assertEqual(got, {"hands": [4, 4], "stock": 54 - 9, "nine": True, "suffix": "-4cards-yaniv9",
                               "model": "yaniv-net-4cards-yaniv9.json"})
        standard = self._run(code)
        self.assertEqual((standard["hands"], standard["nine"], standard["model"]),
                         ([5, 5], False, "yaniv-net.json"))

    def test_main_hands_the_rules_to_the_engine(self):
        code = (
            "import json, os, sys\n"
            "sys.argv = ['main.py', '--hand', '4', '--yaniv-at', '9', '--quick']\n"
            "import main\n"
            "from training.yaniv import engine\n"
            "print(json.dumps([engine.HAND_SIZE, engine.YANIV_LIMIT, os.environ['YANIV_HAND_SIZE']]))\n"
        )
        env = {k: v for k, v in os.environ.items() if k not in ("YANIV_HAND_SIZE", "YANIV_CALL_AT")}
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True, env=env)
        self.assertEqual(json.loads(out.stdout.strip().splitlines()[-1]), [4, 9, "4"])


if __name__ == "__main__":
    unittest.main()
