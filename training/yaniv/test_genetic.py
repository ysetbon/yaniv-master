"""The genetic recipe: operators, breeding and a tiny end-to-end run."""

from __future__ import annotations

import dataclasses
import random
import unittest

import numpy as np

from .genetic import Member, breed, crossover, fitness, mutate, seed_population
from .network import MLP, default_net
from .parallel import Workers, bot_spec, net_spec
from .train import Observer, TrainConfig, run_training


class Operators(unittest.TestCase):
    def test_zero_mutation_is_a_copy_and_parents_are_untouched(self):
        base = default_net(3)
        before = [w.copy() for w in base.W]
        same = mutate(base, 0.0, np.random.default_rng(0))
        moved = mutate(base, 0.2, np.random.default_rng(0))
        for w, b0, s in zip(base.W, before, same.W):
            np.testing.assert_array_equal(w, b0)
            np.testing.assert_array_equal(s, b0)
        self.assertTrue(any(not np.allclose(m, b0) for m, b0 in zip(moved.W, before)))

    def test_crossover_takes_whole_neurons_from_one_parent(self):
        a, b = default_net(1), default_net(2)
        child = crossover(a, b, np.random.default_rng(5))
        for layer in range(len(child.W)):
            for unit in range(child.W[layer].shape[1]):
                col = child.W[layer][:, unit]
                from_a = np.array_equal(col, a.W[layer][:, unit]) and child.b[layer][unit] == a.b[layer][unit]
                from_b = np.array_equal(col, b.W[layer][:, unit]) and child.b[layer][unit] == b.b[layer][unit]
                self.assertTrue(from_a or from_b)

    def test_breed_keeps_elites_and_fills_the_population(self):
        rng, pick = np.random.default_rng(0), random.Random(0)
        ranked = seed_population(default_net(3), 8, 0.1, rng)
        for i, m in enumerate(ranked):
            m.fitness = 1.0 - i / 10
        nxt = breed(ranked, 8, 2, 3, 0.5, 0.1, 2, rng, pick)
        self.assertEqual(len(nxt), 8)
        self.assertEqual([m.op for m in nxt[:2]], ["elite", "elite"])
        self.assertIs(nxt[0].net, ranked[0].net)
        self.assertTrue(all(m.op in ("cross", "mutant") for m in nxt[2:]))
        self.assertTrue(all(m.id.startswith("2.") for m in nxt))

    def test_fitness_prefers_more_wins_then_fewer_points(self):
        a = {"win_rate": 0.5, "hero_avg": 5.0, "villain_avg": 5.0}
        b = {"win_rate": 0.5, "hero_avg": 4.0, "villain_avg": 5.0}
        c = {"win_rate": 0.55, "hero_avg": 8.0, "villain_avg": 4.0}
        self.assertGreater(fitness(b), fitness(a))
        self.assertGreater(fitness(c), fitness(b))


class SameDeals(unittest.TestCase):
    def test_every_member_sees_the_same_deals(self):
        net = default_net(4)
        spec = net_spec(net)
        one = Workers(1).rounds_many([spec, spec], bot_spec("counter"), 12, 9)
        self.assertEqual(one[0], one[1])
        pool = Workers(2)
        try:
            two = pool.rounds_many([spec, spec, spec], bot_spec("counter"), 12, 9)
        finally:
            pool.close()
        self.assertEqual(two[0], one[0])
        self.assertEqual(two[0], two[2])


class Recording(Observer):
    def __init__(self):
        self.pops, self.gens = [], []

    def log(self, message: str) -> None:
        pass

    def population(self, summary, nets):
        self.pops.append(summary)
        self.nets = nets

    def generation(self, gen, net):
        self.gens.append(gen)


class EndToEnd(unittest.TestCase):
    def test_tiny_genetic_run(self):
        cfg = dataclasses.replace(TrainConfig.genetic(quick=True), workers=1, cache=False, generations=2, population=4,
                                  fitness_rounds=6, eval_rounds=6, eval_matches=2, final_rounds=6, final_matches=2,
                                  gate_finalists=2, gate_rounds=6, gate_matches=2)
        obs = Recording()
        report = run_training(cfg, obs, model_path=None, report_path=None)
        self.assertEqual(report["recipe"], "genetic")
        self.assertEqual(len(obs.pops), 2)
        self.assertEqual(len(obs.pops[0]["members"]), 4)
        self.assertEqual([g["index"] for g in obs.gens], [0, 1, 2])
        self.assertTrue(all(isinstance(n, MLP) for n in obs.nets))
        rates = [m["win_rate"] for m in obs.pops[0]["members"]]
        self.assertEqual(rates, sorted(rates, reverse=True))
        self.assertIn(report["best_generation"], (0, 1, 2))
        replay = report["gate_replay"]
        self.assertEqual(len(replay["finalists"]), 2)
        self.assertIn(replay["chosen"], replay["finalists"])
        self.assertEqual(sorted(replay["results"]), sorted(str(g) for g in replay["finalists"]))
        self.assertIn(report["best_generation"], (0, replay["chosen"]))


if __name__ == "__main__":
    unittest.main()
