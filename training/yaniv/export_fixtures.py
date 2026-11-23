"""Write rule and feature fixtures the TypeScript port must match."""

from __future__ import annotations

import json
import random
from pathlib import Path

from training.yaniv.engine import (
    DECK_SIZE,
    Action,
    View,
    apply_action,
    best_keep_points,
    legal_actions,
    make_view,
    new_round,
    resolve_showdown,
)
from training.yaniv.features import action_features, net_actions, state_features

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "src" / "lib" / "yaniv" / "fixtures.json"
# Card masks need 54 bits, more than a JSON number holds exactly in
# JavaScript, so every mask is written as a decimal string.
MASK_KEYS = {"hands", "known", "discard_mask", "mask", "my_hand", "my_known", "opp_known"}


def masks_as_strings(value, key=None):
    if isinstance(value, dict):
        return {k: masks_as_strings(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [masks_as_strings(v, key) for v in value]
    if key in MASK_KEYS and isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return value


def main() -> None:
    rng = random.Random(21)
    keeps = []
    for _ in range(250):
        deck = list(range(DECK_SIZE))
        rng.shuffle(deck)
        mask = 0
        for card in deck[: rng.randint(0, 5)]:
            mask |= 1 << card
        keeps.append({"mask": mask, "keep": best_keep_points(mask)})

    steps = []
    near_cuts = [[45, 92], [48, 96], [40, 90], [99, 49]]
    for game in range(32):
        scores = near_cuts[game % 4] if game >= 24 else [game % 7, (game * 3) % 11]
        state = new_round(list(scores), game % 2, rng)
        guard = 0
        while not state.terminal and guard < 18:
            guard += 1
            if state.ply >= 48:
                before = state.to_json()
                resolve_showdown(state)
                steps.append({"before": before, "action": None, "after": state.to_json()})
                break
            view = make_view(state, state.turn)
            actions = legal_actions(view)
            if not actions:
                before = state.to_json()
                resolve_showdown(state)
                steps.append({"before": before, "action": None, "after": state.to_json()})
                break
            action = actions[(game + guard) % len(actions)]
            before = state.to_json()
            apply_action(state, action)
            steps.append({"before": before, "action": action.to_json(), "after": state.to_json()})

    # Round endings that land exactly on 50 or 100, so the TypeScript port
    # must cut those scores the same way.
    ten, ace, seven, joker = 9, 0, 6, 52
    for scores, hands, turn, yaniv in (
        ([40, 0], [1 << ten, 1 << joker], 0, False),  # showdown: 40 + 10 = 50 -> 0
        ([90, 30], [1 << ten, 1 << ace], 1, True),  # Yaniv: 90 + 10 = 100 -> 50
        ([63, 20], [1 << seven, 1 << joker], 0, True),  # Assaf: 63 + 7 + 30 = 100 -> 50
        ([45, 95], [(1 << 52) | (1 << 53), 1 << 4], 0, True),  # two jokers call: 95 + 5 = 100 -> 50
    ):
        state = new_round(scores, turn, rng)
        state.hands = list(hands)
        state.known = [0, 0]
        before = state.to_json()
        if yaniv:
            apply_action(state, Action("yaniv"))
            action = Action("yaniv").to_json()
        else:
            resolve_showdown(state)
            action = None
        steps.append({"before": before, "action": action, "after": state.to_json()})

    anchor = View(
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
    features = [feature_case(anchor)]
    # Low hands where Yaniv is legal, with jokers, so the exact Assaf chance
    # and the leaning estimate of the opponent's hand are checked too.
    for hand, known, opp_size, discard, ply in (
        ((0, 1), 0, 2, (9, 22, 35, 48, 12, 25), 10),  # ace and two against two hidden cards
        ((52, 3), 1 << 13, 3, (8, 21, 34), 6),  # a joker and a four; the opponent shows an ace
        ((53,), 0, 1, (10, 23, 36, 49, 11, 24, 37, 50), 20),  # a lone black joker against one card
        ((0, 13, 2), 1 << 52, 4, (5, 18), 4),  # two aces and a three; the opponent took the red joker
    ):
        mask = 0
        for card in hand:
            mask |= 1 << card
        seen = mask | known
        for card in discard:
            seen |= 1 << card
        stock = 54 - bin(seen).count("1") - (opp_size - bin(known).count("1"))
        features.append(feature_case(View(my_hand=mask, my_known=0, opp_known=known, opp_size=opp_size,
                                          stock_size=stock, discard=discard, my_score=30, opp_score=48,
                                          ply=ply, seat=0)))
    probe = random.Random(8)
    for i in range(6):
        state = new_round([i, 3], 0, probe)
        features.append(feature_case(make_view(state, 0)))
        if not state.terminal:
            apply_action(state, legal_actions(make_view(state, state.turn))[0])
            features.append(feature_case(make_view(state, state.turn)))

    payload = masks_as_strings({"keeps": keeps, "steps": steps, "features": features, "moe": moe_case(features)})
    OUT.write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {OUT} keeps={len(keeps)} steps={len(steps)} features={len(features)}")


def moe_case(features: list[dict]) -> dict:
    """A small decider with three experts, and the scores Python gives some real feature rows."""
    import numpy as np

    from training.yaniv.features import INPUT_DIM, STATE_DIM
    from training.yaniv.moe import MoE
    from training.yaniv.network import MLP, net_payload

    moe = MoE(MLP([STATE_DIM, 6, 3], seed=31), [MLP([INPUT_DIM, 8, 1], seed=40 + k) for k in range(3)])
    for expert in moe.experts:
        expert.W[-1] *= 60.0  # wake up the small last layer, so the experts disagree
    moe.gate.W[-1] *= 60.0
    cases = []
    for case in features[:6]:
        state = np.asarray(case["state"], dtype=np.float32)
        rows = np.asarray([a["features"] for a in case["actions"]], dtype=np.float32)
        packed = np.concatenate([np.repeat(state[None, :], len(rows), axis=0), rows], axis=1)
        cases.append({"state": case["state"], "actions": [a["features"] for a in case["actions"]],
                      "scores": moe.forward(packed)[:, 0].tolist()})
    return {"net": net_payload(moe), "cases": cases}


def feature_case(view: View) -> dict:
    actions = net_actions(legal_actions(view))[0]
    return {
        "view": {
            "my_hand": view.my_hand,
            "my_known": view.my_known,
            "opp_known": view.opp_known,
            "opp_size": view.opp_size,
            "stock_size": view.stock_size,
            "discard": list(view.discard),
            "my_score": view.my_score,
            "opp_score": view.opp_score,
            "ply": view.ply,
            "seat": view.seat,
        },
        "state": state_features(view).tolist(),
        "actions": [
            {"action": action.to_json(), "features": action_features(view, action).tolist()}
            for action in actions[:12]
        ],
    }


if __name__ == "__main__":
    main()
