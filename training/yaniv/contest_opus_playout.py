"""Paired play-outs for team opus: is move B better than move A in the positions where a rule says so?

A test, not a fit: nothing here writes or searches weights. Rounds from 0-0 in the real game (5 cards, Yaniv at
7; leave YANIV_HAND_SIZE / YANIV_CALL_AT unset). The hero plays `hero` against `opp`. At every hero decision,
`alt(view, actions, a)` may return another move B (or None). When B differs from the hero's move A, the round is
copied twice, A is applied to one copy and B to the other, and both are played to the end with the hero's own
policy (and the same opponent) on the same hidden cards (the stock order is fixed). The record keeps
gain = (opponent's points - hero's points | B) - (same | A), so gain > 0 means B was better for the hero.
Then the round continues with A as played.

    from training.yaniv.contest_opus_playout import playout, summary
    recs = playout("public/model/yaniv-net-5cards-handmade.json", "counter", "my_module:my_alt",
                   rounds=4000, seed=990000)
    print(summary(recs))

`alt` is given as "module:function" so the worker processes can import it. The function receives
(view, actions, a, ctx) where ctx = {"hero": chooser, "opp_hand": mask (truth, for logging only - a rule must not
read it), "state": the State (truth: same caution)} and returns an Action, or None, or (Action, tag) to group
records by a tag.

Players: "counter", "trained", "hm5" (handmade5), "hm4" (the 4-card hand-written net), or a net JSON path.
"""

from __future__ import annotations

import importlib
import math
import random
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAMED = {"trained": ROOT / "public/model/yaniv-net.json",
         "hm5": ROOT / "public/model/yaniv-net-5cards-handmade.json",
         "hm4": ROOT / "public/model/yaniv-net-4cards-yaniv7-handmade.json",
         "opus": ROOT / "public/model/yaniv-net-contest-opus.json"}


def player(name: str):
    """A chooser (view, actions, rng) -> Action for a name or a net path."""
    if name == "counter":
        from .heuristic import choose_heuristic
        return choose_heuristic
    from .network import load_net
    from .train import net_chooser
    return net_chooser(load_net(Path(NAMED.get(name, name))), 0.0)


def _fn(spec: str):
    mod, fn = spec.split(":")
    return getattr(importlib.import_module(mod), fn)


def fork(st):
    from .engine import State
    return State(list(st.hands), list(st.known), list(st.stock), list(st.discard), list(st.scores), st.turn, st.ply)


def finish(st, hero, opp, hs: int) -> int:
    """Play to the end; the hero's margin (opponent's points added minus the hero's)."""
    from .engine import MAX_PLIES, apply_action, legal_actions, make_view, resolve_showdown
    while not st.terminal:
        if st.ply >= MAX_PLIES:
            resolve_showdown(st)
            break
        v = make_view(st, st.turn)
        acts = legal_actions(v)
        if not acts:
            resolve_showdown(st)
            break
        apply_action(st, (hero if st.turn == hs else opp)(v, acts, None))
    return st.result.added[1 - hs] - st.result.added[hs]


def _job(job):
    hero_name, opp_name, alt_spec, rounds, seed, scores = job
    from .engine import apply_action, cards_from_mask, legal_actions, make_view, new_round, POINTS_OF, MAX_PLIES, \
        resolve_showdown
    from .features import action_key
    hero, opp, alt = player(hero_name), player(opp_name), _fn(alt_spec)
    rng = random.Random(seed)
    recs, margins = [], []
    for i in range(rounds):
        st = new_round(list(scores), i % 2, rng)
        hs = (i // 2) % 2
        while not st.terminal:
            if st.ply >= MAX_PLIES:
                resolve_showdown(st)
                break
            v = make_view(st, st.turn)
            acts = legal_actions(v)
            if not acts:
                resolve_showdown(st)
                break
            if st.turn != hs:
                apply_action(st, opp(v, acts, None))
                continue
            a = hero(v, acts, None)
            got = alt(v, acts, a, {"hero": hero, "opp_hand": st.hands[1 - hs], "state": st})
            tag = ""
            if isinstance(got, tuple):
                got, tag = got
            if got is not None and action_key(got) != action_key(a):
                fa, fb = fork(st), fork(st)
                apply_action(fa, a)
                apply_action(fb, got)
                va = fa.result.added[1 - hs] - fa.result.added[hs] if fa.terminal else finish(fa, hero, opp, hs)
                vb = fb.result.added[1 - hs] - fb.result.added[hs] if fb.terminal else finish(fb, hero, opp, hs)
                recs.append({"round": i, "seed": seed, "ply": v.ply, "tag": tag, "gain": vb - va,
                             "hand": sorted(POINTS_OF[c] for c in cards_from_mask(v.my_hand)),
                             "opp_n": v.opp_size, "a": _txt(a), "b": _txt(got)})
            apply_action(st, a)
        margins.append(st.result.added[1 - hs] - st.result.added[hs])
    return recs, margins


def _txt(act) -> str:
    from .engine import card_name, cards_from_mask
    if act.kind == "yaniv":
        return "YANIV"
    return "+".join(card_name(c) for c in cards_from_mask(act.discard_mask)) + "/" + act.draw


def playout(hero: str, opp: str, alt: str, rounds: int = 4000, seed: int = 990000, processes: int = 4,
            scores=(0, 0)):
    """Records of every decision where `alt` disagreed with the hero, plus rounds played (for a per-round rate)."""
    chunks = [(hero, opp, alt, rounds // processes + (1 if k < rounds % processes else 0), seed * 1000 + k, scores)
              for k in range(processes)]
    with Pool(processes) as pool:
        parts = pool.map(_job, chunks)
    recs = [r for p, _ in parts for r in p]
    margins = [m for _, ms in parts for m in ms]
    return recs, len(margins), sum(margins) / max(1, len(margins))


def summary(recs, rounds: int, by: str = "tag") -> str:
    """Gain per occurrence (mean, standard error), occurrences per round, and gain per round, grouped."""
    groups: dict = {}
    for r in recs:
        groups.setdefault(r[by] if by else "", []).append(r["gain"])
    lines = [f"| {by or 'all'} | n | per round | gain each (+-se) | gain a round |", "|---|---|---|---|---|"]
    for k in sorted(groups, key=lambda k: -len(groups[k])):
        g = groups[k]
        n = len(g)
        mean = sum(g) / n
        sd = math.sqrt(sum((x - mean) ** 2 for x in g) / max(1, n - 1))
        lines.append(f"| {k} | {n} | {n / rounds:.3f} | {mean:+.2f} (+-{sd / math.sqrt(n):.2f}) | "
                     f"{mean * n / rounds:+.3f} |")
    return "\n".join(lines)
