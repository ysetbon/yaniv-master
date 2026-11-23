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


class Confidence(unittest.TestCase):
    def test_keeps_the_default_unless_a_rival_clearly_wins(self):
        import numpy as np

        from .search import pick_move

        rng = np.random.default_rng(0)
        base = rng.normal(0, 10, (48, 1))
        noise = rng.normal(0, 3, (48, 3))
        close = np.hstack([base, base + 0.3 + noise[:, :1], base - 1 + noise[:, 1:2]])
        self.assertEqual(pick_move(close, 0, 1.5), 0)  # a 0.3-point edge is luck
        clear = np.hstack([base, base + 4 + noise[:, :1], base + 1 + noise[:, 1:2]])
        self.assertEqual(pick_move(clear, 0, 1.5), 1)
        edge = np.hstack([base, base + 0.3, base - 1])  # a steady 0.3-point edge, no noise
        self.assertEqual(pick_move(edge, 0, 1.5), 1)


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

    def test_start_from_a_saved_net_skips_the_copy(self):
        import tempfile
        from pathlib import Path

        import numpy as np

        from .network import save_net

        start = default_net(9)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "start.json"
            save_net(start, path, {})
            cfg = dataclasses.replace(TrainConfig.search(quick=True), workers=1, cache=False, generations=0,
                                      start_from=str(path), eval_rounds=4, eval_matches=2, final_rounds=4,
                                      final_matches=2)

            class Keep(Observer):
                def __init__(self):
                    self.gens, self.nets = [], []

                def generation(self, gen, net):
                    self.gens.append(gen)
                    self.nets.append(net)

            obs = Keep()
            report = run_training(cfg, obs, model_path=None, report_path=None)
        self.assertFalse([h for h in report["history"] if h["phase"] == "imitation"])
        self.assertEqual(obs.gens[0]["label"], "Started from start.json")
        for a, b in zip(obs.nets[0].W, start.W):
            np.testing.assert_allclose(a, b, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()


class FastFeatures(unittest.TestCase):
    """The one-pass feature builders must give exactly the rows the net was trained on."""

    def test_action_matrix_equals_action_features(self):
        import numpy as np

        from .features import action_features, action_matrix

        for seed in range(30):
            state = new_round([seed * 3 % 95, seed * 7 % 95], seed % 2, random.Random(seed))
            while not state.terminal and state.ply < 48:
                view = make_view(state, state.turn)
                actions = legal_actions(view)
                if not actions:
                    break
                expected = np.stack([action_features(view, a) for a in actions])
                np.testing.assert_array_equal(action_matrix(view, actions), expected)
                apply_action(state, choose_heuristic(view, actions))

    def test_state_features_match_the_parity_fixtures(self):
        import json
        from pathlib import Path

        import numpy as np

        from .engine import View
        from .features import action_matrix, net_actions, state_features

        path = Path(__file__).resolve().parents[2] / "src" / "lib" / "yaniv" / "fixtures.json"
        for case in json.loads(path.read_text())["features"]:
            raw = case["view"]
            view = View(my_hand=int(raw["my_hand"]), my_known=int(raw["my_known"]), opp_known=int(raw["opp_known"]),
                        opp_size=raw["opp_size"], stock_size=raw["stock_size"], discard=tuple(raw["discard"]),
                        my_score=raw["my_score"], opp_score=raw["opp_score"], ply=raw["ply"], seat=raw["seat"])
            np.testing.assert_allclose(state_features(view), case["state"], atol=1e-6)
            rows = action_matrix(view, net_actions(legal_actions(view))[0][:len(case["actions"])])
            np.testing.assert_allclose(rows, [a["features"] for a in case["actions"]], atol=1e-6)


class Effort(unittest.TestCase):
    def test_share_of_cores(self):
        import os

        from .parallel import default_workers

        cores = os.cpu_count() or 2
        self.assertEqual(default_workers(1.0), cores)
        self.assertEqual(default_workers(0.7), max(1, round(cores * 0.7)))
        self.assertEqual(default_workers(0.01), 1)


class AssafChance(unittest.TestCase):
    def test_exact_chance_matches_counting_every_hand(self):
        import itertools
        from math import comb

        from .engine import View
        from .features import assaf_chance, pool_mask

        rng = random.Random(4)
        for _ in range(40):
            deck = list(range(DECK_SIZE))
            rng.shuffle(deck)
            hand = sum(1 << c for c in deck[:rng.randint(1, 3)])
            known = sum(1 << c for c in deck[3:3 + rng.randint(0, 1)])
            discard = tuple(deck[5:5 + rng.randint(1, 12)])
            opp_size = rng.randint(1, 4)
            view = View(my_hand=hand, my_known=0, opp_known=known, opp_size=opp_size, stock_size=20, discard=discard,
                        my_score=0, opp_score=0, ply=8, seat=0)
            mine = sum(POINTS_OF[c] for c in cards_from_mask(hand))
            known_pts = sum(POINTS_OF[c] for c in cards_from_mask(known))
            pool = cards_from_mask(pool_mask(view))
            need = opp_size - bin(known).count("1")
            if need <= 0:
                expected = 1.0 if known_pts <= mine else 0.0
            else:
                caught = sum(1 for combo in itertools.combinations(pool, need)
                             if known_pts + sum(POINTS_OF[c] for c in combo) <= mine)
                expected = caught / comb(len(pool), need)
            self.assertAlmostEqual(assaf_chance(view), expected, places=12)


class CopyFrom(unittest.TestCase):
    def test_copy_an_older_389_input_net(self):
        import tempfile
        from pathlib import Path

        from . import features_v1
        from .network import MLP, save_net

        old = MLP([features_v1.INPUT_DIM, 8, 8, 1], seed=4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old-best.json"
            save_net(old, path, {})
            cfg = dataclasses.replace(TrainConfig.search(quick=True), workers=1, cache=False, generations=0,
                                      copy_from=str(path), expert_rounds=4, holdout_rounds=2, epochs=1,
                                      eval_rounds=4, eval_matches=2, final_rounds=4, final_matches=2)

            class Keep(Observer):
                def __init__(self):
                    self.gens = []

                def generation(self, gen, net):
                    self.gens.append((gen, net))

            obs = Keep()
            report = run_training(cfg, obs, model_path=None, report_path=None)
        gen, net = obs.gens[0]
        self.assertEqual(gen["label"], "Copied old-best.json")
        from .features import INPUT_DIM

        self.assertEqual(net.dims[0], INPUT_DIM)  # the copy uses the suit-free inputs
        self.assertTrue([h for h in report["history"] if h["phase"] == "imitation"])


class RunRecord(unittest.TestCase):
    def test_a_run_writes_its_log_and_summary(self):
        import tempfile
        from pathlib import Path

        cfg = dataclasses.replace(TrainConfig.search(quick=True), workers=1, cache=False, generations=1,
                                  search_rounds=2, search_samples=4, eval_rounds=4, eval_matches=2, gate_rounds=4,
                                  gate_matches=2, final_rounds=4, final_matches=2)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, results = Path(tmp) / "20990101-000000", Path(tmp) / "results"
            run_training(cfg, Observer(), model_path=None, report_path=None, run_dir=run_dir, results_dir=results)
            log = (run_dir / "log.txt").read_text()
            summary = (run_dir / "summary.md").read_text()
            self.assertIn("Search 1:", log)
            self.assertTrue(log.splitlines()[0].startswith("0:00:0"))
            self.assertIn("## Generations", summary)
            self.assertIn("| 1 | Search pass 1 |", summary)
            self.assertIn("ships.", summary)
            self.assertEqual((results / "20990101-000000.md").read_text(), summary)


class Experts(unittest.TestCase):
    def test_mixture_learns_and_runs_in_the_search_recipe(self):
        import tempfile
        from pathlib import Path

        import numpy as np

        from .moe import MoE
        from .network import load_net, save_net

        start = default_net(9)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "start.json"
            save_net(start, path, {})
            cfg = dataclasses.replace(TrainConfig.search(quick=True), workers=2, cache=False, generations=2, experts=3,
                                      start_from=str(path), search_rounds=2, search_samples=4, eval_rounds=4,
                                      eval_matches=2, gate_rounds=4, gate_matches=2, final_rounds=4, final_matches=2,
                                      expert_rounds=20, holdout_rounds=4)
            run_dir = Path(tmp) / "run"
            report = run_training(cfg, Observer(), model_path=None, report_path=None, run_dir=run_dir, results_dir=None)
            gen1 = load_net(run_dir / "gen-01.json")
        self.assertIsInstance(gen1, MoE)
        self.assertEqual(len(gen1.experts), 3)
        self.assertEqual(report["experts"] in (1, 3), True)  # Gen 0 stays a plain net and may still ship
        x = np.random.default_rng(1).random((5, start.dims[0])).astype(np.float32)
        self.assertEqual(gen1.forward(x).shape, (5, 1))
