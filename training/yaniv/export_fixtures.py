"""Write rule and feature fixtures the TypeScript port must match."""

from __future__ import annotations

import json
import random
from pathlib import Path

from training.yaniv.engine import (
    Action,
    View,
    apply_action,
    best_keep_points,
    legal_actions,
    make_view,
    new_round,
    resolve_showdown,
)
from training.yaniv.features import action_features, state_features

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "src" / "lib" / "yaniv" / "fixtures.json"


def main() -> None:
    rng = random.Random(21)
    keeps = []
    for _ in range(250):
        deck = list(range(52))
        rng.shuffle(deck)
        mask = 0
        for card in deck[: rng.randint(0, 5)]:
            mask |= 1 << card
        keeps.append({"mask": mask, "keep": best_keep_points(mask)})

    steps = []
    for game in range(24):
        state = new_round([game % 7, (game * 3) % 11], game % 2, rng)
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
    probe = random.Random(8)
    for i in range(6):
        state = new_round([i, 3], 0, probe)
        features.append(feature_case(make_view(state, 0)))
        if not state.terminal:
            apply_action(state, legal_actions(make_view(state, state.turn))[0])
            features.append(feature_case(make_view(state, state.turn)))

    payload = {"keeps": keeps, "steps": steps, "features": features}
    OUT.write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {OUT} keeps={len(keeps)} steps={len(steps)} features={len(features)}")


def feature_case(view: View) -> dict:
    actions = legal_actions(view)
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
