"""Thread `match`: check that a design moves as its rules say. Plays hm5 v hm5 matches; at every decision of seat 0
compares the design's move with hm5's and classifies the differences by the rule predicates (Kit.trace values)."""
import json
import random
import sys
from collections import Counter

import numpy as np

from . import contest_fable_lab as lab
from . import contest_fable_match as m
from .engine import MATCH_LIMIT, POINTS_OF, apply_action, legal_actions, make_view, mask_points, new_round
from .features import action_key, action_matrix, net_actions, state_features
from .train import net_chooser

kw = json.loads(sys.argv[1])
n = int(sys.argv[2])
net, kit = m.build(**kw)
new, old = net_chooser(net, 0.0), lab.player("hm5")
rng = random.Random(5)
c = Counter()
shown = 0
for i in range(n):
    scores, leader, played = [0, 0], i % 2, 0
    while max(scores) < MATCH_LIMIT and played < 40:
        st = new_round(scores, leader, rng)
        while not st.terminal and st.ply < 48:
            v = make_view(st, st.turn)
            acts = legal_actions(v)
            if not acts:
                break
            a = old(v, acts, None)
            if st.turn == 0:
                b = new(v, acts, None)
                c["decisions"] += 1
                c["low score decisions"] += v.my_score + mask_points(v.my_hand) < 30 and v.opp_score < 30
                f = state_features(v)
                s, t, est = v.my_score, v.opp_score, float(f[94]) * 40
                hidden = v.opp_size - round(float(f[84]) * 5)
                d = min((t + est - 50, t + est - 100), key=abs)
                deny = "DENY" in kw and hidden <= 2 and (abs(d) < 0.5 or (kw["DENY"][0] and hidden >= 1 and abs(d) < kw["DENY"][0]))
                out = s + mask_points(v.my_hand) > 100
                if a.kind == "yaniv":
                    c["hm5 calls"] += 1
                    c["hm5 calls, deny predicate"] += bool(deny)
                if action_key(a) != action_key(b):
                    c["differ"] += 1
                    if v.my_score + mask_points(v.my_hand) < 30 and v.opp_score < 30:
                        c["differ at low scores (deny only possible)"] += 1
                    if a.kind == "yaniv" and b.kind != "yaniv":
                        c["call -> throw"] += 1
                        c["call -> throw, deny predicate true"] += bool(deny)
                    elif b.kind == "yaniv":
                        c["throw -> call"] += 1
                        c["throw -> call, out-if-lose and assaf < BUSTCALL"] += bool(out and float(f[95]) < kw.get("BUSTCALL", 0))
                    else:
                        c["throw -> other throw"] += 1
                        kept = mask_points(v.my_hand & ~b.discard_mask)
                        opn = POINTS_OF[v.discard[-1]]
                        if b.draw == "open" and s + kept + opn in (50, 100):
                            c["  new move: open take lands on cut"] += 1
                        elif b.draw == "stock" and (50 - s - kept in range(0, 11) or 100 - s - kept in range(0, 11)):
                            c["  new move: deck draw that can land on cut"] += 1
                        else:
                            c["  new move: other"] += 1
                            if shown < 6:
                                shown += 1
                                rows = np.concatenate([np.repeat(f[None, :], 2, 0), action_matrix(v, [a, b])], axis=1)
                                ta, tb = kit.trace(rows[0]), kit.trace(rows[1])
                                diff = {k: (round(ta[k], 2), round(tb[k], 2)) for k in ta if ("cut" in k or k == "OUT") and abs(ta[k] - tb[k]) > 1e-6 and "_lo" not in k and "_mid" not in k and "_hi" not in k}
                                print("other:", "s", s, "t", t, "est", round(est, 1), lab.describe(a), "->", lab.describe(b), diff)
                elif a.kind == "yaniv" and deny:
                    c["deny predicate true but still calls"] += 1
            apply_action(st, a)
        lab.finish(st, (old, old))
        scores, leader, played = st.scores, 1 - leader, played + 1
for k, x in c.items():
    print(f"{k:55s} {x}")
