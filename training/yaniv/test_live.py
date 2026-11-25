"""The dashboard hub: everything it sends must be JSON a browser can parse."""

from __future__ import annotations

import json
import math
import unittest

from .live.hub import Hub, _clean


class BrowserJson(unittest.TestCase):
    def test_nan_and_infinity_become_null(self):
        value = {"fit": float("nan"), "rows": [1.0, float("inf"), {"x": -math.inf}], "ok": 0.5, "n": 3}
        self.assertEqual(_clean(value), {"fit": None, "rows": [1.0, None, {"x": None}], "ok": 0.5, "n": 3})

    def test_published_generations_and_snapshots_parse_strictly(self):
        hub = Hub({}, "cpu", "", False)
        q = hub.subscribe()
        hub.append("generation", "generations", {"index": 1, "value_fresh_fit": float("nan")})
        message = json.loads(q.get_nowait(), parse_constant=lambda c: self.fail(f"{c} in a message"))
        self.assertIsNone(message["data"]["value_fresh_fit"])
        state = json.loads(json.dumps(hub.snapshot(), allow_nan=False))
        self.assertEqual(state["generations"][0]["index"], 1)


if __name__ == "__main__":
    unittest.main()
