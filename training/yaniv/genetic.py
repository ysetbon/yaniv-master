"""Neuroevolution for the Yaniv net: a population of networks, bred by fitness.

Every member has the same shape as the copied net (377 -> 192 -> 192 -> 1)
and descends from it, so neuron i means roughly the same thing in every
member and crossover can swap whole neurons between parents.

One generation:
1. every member plays the same deals against the card counter (fitness),
2. the best `elites` members carry over unchanged,
3. the rest of the next generation are children: parents are picked by
   tournament, about half are crossed neuron by neuron, and every child gets
   Gaussian noise on its weights, scaled to each layer's own spread.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from .network import MLP


@dataclass
class Member:
    id: str  # "<generation>.<slot>", e.g. "4.7"
    net: MLP
    parents: list[str] = field(default_factory=list)
    op: str = "mutant"  # "copy", "elite", "cross" or "mutant"
    fitness: float = 0.0
    record: dict | None = None

    def info(self) -> dict:
        r = self.record or {}
        return {
            "id": self.id,
            "parents": self.parents,
            "op": self.op,
            "fitness": round(self.fitness, 5),
            "win_rate": r.get("win_rate"),
            "wins": r.get("wins"),
            "losses": r.get("losses"),
            "ties": r.get("ties"),
            "hero_avg": r.get("hero_avg"),
            "villain_avg": r.get("villain_avg"),
        }


def fitness(record: dict) -> float:
    """Rounds won against the counter, with the points margin as a small tie-breaker."""
    return record["win_rate"] + 0.005 * (record["villain_avg"] - record["hero_avg"])


def mutate(net: MLP, sigma: float, rng: np.random.Generator) -> MLP:
    """A copy with Gaussian noise of `sigma` times each layer's weight spread."""
    child = net.clone()
    for i in range(len(child.W)):
        spread = float(child.W[i].std()) or 1e-3
        child.W[i] += rng.normal(0.0, sigma * spread, child.W[i].shape).astype(np.float32)
        child.b[i] += rng.normal(0.0, sigma * spread, child.b[i].shape).astype(np.float32)
    return child


def crossover(a: MLP, b: MLP, rng: np.random.Generator) -> MLP:
    """Each neuron, with its incoming weights and bias, comes from one parent or the other."""
    child = a.clone()
    for i in range(len(child.W)):
        take = rng.random(child.W[i].shape[1]) < 0.5
        child.W[i][:, take] = b.W[i][:, take]
        child.b[i][take] = b.b[i][take]
    return child


def tournament(ranked: list[Member], k: int, pick: random.Random) -> Member:
    """The fittest of k members drawn at random."""
    return max(pick.sample(ranked, min(k, len(ranked))), key=lambda m: m.fitness)


def seed_population(base: MLP, size: int, sigma: float, rng: np.random.Generator) -> list[Member]:
    """Generation 1: the copied net itself plus mutants of it."""
    members = [Member("1.0", base.clone(), [], "copy")]
    for slot in range(1, size):
        members.append(Member(f"1.{slot}", mutate(base, sigma, rng), ["copy"], "mutant"))
    return members


def breed(ranked: list[Member], size: int, elites: int, tournament_k: int, crossover_rate: float,
          sigma: float, generation: int, rng: np.random.Generator, pick: random.Random) -> list[Member]:
    """The next generation from members sorted best first."""
    out = [Member(f"{generation}.{slot}", m.net, [m.id], "elite") for slot, m in enumerate(ranked[:elites])]
    while len(out) < size:
        slot = len(out)
        first = tournament(ranked, tournament_k, pick)
        if pick.random() < crossover_rate and len(ranked) > 1:
            second = tournament([m for m in ranked if m is not first], tournament_k, pick)
            child = mutate(crossover(first.net, second.net, rng), sigma, rng)
            out.append(Member(f"{generation}.{slot}", child, [first.id, second.id], "cross"))
        else:
            out.append(Member(f"{generation}.{slot}", mutate(first.net, sigma, rng), [first.id], "mutant"))
    return out
