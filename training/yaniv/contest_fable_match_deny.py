"""Thread `match`: report of probe_deny records (the hero called Yaniv; alternative = its best throw)."""
import pickle
import sys

from .contest_fable_match_probe import band, report

recs, stats, n = pickle.load(open(sys.argv[1], "rb"))
print(sys.argv[1], "matches", n, stats, "records", len(recs))
D = (-8, -5, -3, -2, -1, 1, 2, 3, 5, 8)
print(report(recs, stats, n, lambda r: "on cut" if r["on"] else "off", title="-- hindsight"))
print(report(recs, stats, n, lambda r: ("need", min(r["need"], 3)), title="-- hidden opponent cards"))
print(report(recs, stats, n, lambda r: ("need", min(r["need"], 3), "on" if r["on"] else "off"), title="-- hidden x hindsight"))
ex = lambda r: r["need"] == 0 and r["t"] + r["known_pts"] in (50, 100)
print(report(recs, stats, n, lambda r: "all known and exactly on cut" if ex(r) else "other", title="-- exact", least=5))
print(report(recs, stats, n, lambda r: ("t+est-cut", band(r["d"], D)), title="-- signed distance of opp score + est from the cut"))
print(report(recs, stats, n, lambda r: ("need", min(r["need"], 3), "t+est-cut", band(r["d"], (-3, -1, 1, 3))), title="-- hidden x distance", least=20))
print(report(recs, stats, n, lambda r: ("sure" if r["sure"] else "unsure", "t+est-cut", band(r["d"], (-3, -1, 1, 3))), title="-- sure call x distance", least=20))
print(report(recs, stats, n, lambda r: ("cut", 50 if abs(r["t"] + r["est"] - 50) < abs(r["t"] + r["est"] - 100) else 100, "t+est-cut", band(r["d"], (-3, -1, 1, 3))), title="-- which cut x distance", least=20))
import collections
c = collections.Counter(); tot = collections.Counter()
for r in recs:
    k = (min(r["need"], 3), band(r["d"], (-3, -1, 1, 3))); tot[k] += 1; c[k] += r["on"]
print("P(on cut) by (hidden, distance):", {k: (tot[k], round(c[k] / tot[k], 2)) for k in sorted(tot)})
