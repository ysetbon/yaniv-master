"""Thread `match`: print the grouped equity report of a paired-records pickle (see contest_fable_match_probe)."""
import pickle
import sys

from .contest_fable_match_probe import band, eq_gain, report

recs, stats, n = pickle.load(open(sys.argv[1], "rb"))
print(sys.argv[1], "matches", n, stats)
lab = lambda r: r["label"]
print(report(recs, stats, n, lab, title="-- by label"))
call = [r for r in recs if r["label"].startswith("call")]
print(report(call, stats, n, lambda r: (r["label"], r["ab"]), title="-- call by assaf band"))
print(report(call, stats, n, lambda r: (r["label"], "s" + r["sb"]), title="-- call by my score band"))
print(report(call, stats, n, lambda r: (r["label"], "s" + r["sb"], r["ab"]), title="-- call by my score x assaf", least=25))
hi = lambda x: "hi" if x >= 80 else "mid" if x >= 50 else "lo"
print(report(call, stats, n, lambda r: (r["label"], "s" + r["sb"], "t" + hi(r["t"]), r["ab"]), title="-- call by s x t(lo<50,mid,hi>=80) x assaf", least=25))
print(report(call, stats, n, lambda r: (r["label"], "bust" if r["s"] + r["pts"] > 100 else "s+p+30=cut" if r["s"]+r["pts"]+30 in (50,100) else "caughtfatal" if r["s"] + r["pts"] + 30 > 100 else "safe", r["ab"]), title="-- call by fatal class x assaf", least=25))
nb = [r for r in recs if r["label"] == "nobust"]
print(report(nb, stats, n, lambda r: ("nobust", "oc", r["oc"]), title="-- nobust by opp cards", least=20))
print(report(nb, stats, n, lambda r: ("nobust", "est", band(r["est"], (9, 12, 15, 20))), title="-- nobust by opp est", least=20))
print(report(nb, stats, n, lambda r: ("nobust", "t" + hi(r["t"])), title="-- nobust by opp score", least=20))
mh = [r for r in recs if r["label"] == "minhold"]
print(report(mh, stats, n, lambda r: ("minhold", "s" + r["sb"], "chbust" if r["ch_bust"] else "", "altbust" if r["alt_bust"] else ""), title="-- minhold", least=20))
print(report(mh, stats, n, lambda r: ("minhold", "s" + r["sb"], "oc", min(r["oc"], 5) if r["oc"] > 2 else 2), title="-- minhold by s, oc", least=20))
ct = [r for r in recs if r["label"].startswith("cut")]
print(report(ct, stats, n, lambda r: (r["label"], r["thr"]), title="-- cut alternatives by opp est", least=15))
print(report(ct, stats, n, lambda r: (r["label"], "oc", r["oc"]), title="-- cut alternatives by opp cards", least=15))
