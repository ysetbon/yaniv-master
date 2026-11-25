"""Team fable, thread `yaniv`: tables over the records of contest_fable_yaniv_probe (grouping only).

    python -m training.yaniv.contest_fable_yaniv_tab OPP SEED [section ...]
"""

from __future__ import annotations

import math
import pickle
import sys

from .contest_fable_yaniv_probe import SCRATCH


def load(opp, seed, tag="hm5"):
    with open(SCRATCH / f"calls_{tag}_{opp}_{seed}.pkl", "rb") as fh:
        return pickle.load(fh)


def band(x, cuts):
    for i, c in enumerate(cuts):
        if x < c:
            return f"{i}:<{c:g}"
    return f"{len(cuts)}:>={cuts[-1]:g}"


def aband(r):
    if r["sure"]:
        return "0:sure"
    return band(r["assaf"], [1e-9, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2])


def table(recs, rounds, key, val=lambda r: r["gain"], least=30, title=""):
    groups: dict = {}
    for r in recs:
        groups.setdefault(key(r), []).append(val(r))
    lines = [title] if title else []
    for k in sorted(groups, key=str):
        g = groups[k]
        if len(g) < least:
            continue
        m = sum(g) / len(g)
        se = math.sqrt(max(0.0, sum(x * x for x in g) / len(g) - m * m) / len(g))
        lines.append(f"  {str(k):<44} n {len(g):6d} ({len(g) / rounds:.4f}/rd) each {m:+7.3f} +-{se:.3f}  /rd {sum(g) / rounds:+.4f}")
    return "\n".join(lines)


def rate(recs, key, least=30, title=""):
    """True catch rate of a call (share with opp_true <= pts) against the mean assaf input."""
    groups: dict = {}
    for r in recs:
        groups.setdefault(key(r), []).append(r)
    lines = [title] if title else []
    for k in sorted(groups, key=str):
        g = groups[k]
        if len(g) < least:
            continue
        p = sum(r["caught"] for r in g) / len(g)
        a = sum(r["assaf"] for r in g) / len(g)
        se = math.sqrt(max(p * (1 - p), 1e-9) / len(g))
        lines.append(f"  {str(k):<44} n {len(g):6d}  caught {p:6.1%} +-{se:5.1%}   mean assaf {a:6.1%}   ratio {p / a if a > 0 else float('nan'):5.2f}")
    return "\n".join(lines)


def main(opp, seed, sections):
    recs, stats = load(opp, seed)
    R = stats["rounds"]
    dec = [r for r in recs if r["label"] in ("call", "net_throw")]       # one per Yaniv-legal decision
    called = [r for r in dec if r["did_call"]]
    held = [r for r in dec if not r["did_call"]]
    print(f"== {opp} seed {seed}: {R} rounds, margin {stats['margin'] / R:+.3f}; Yaniv legal {len(dec) / R:.3f}/rd, "
          f"calls {len(called) / R:.3f}/rd (caught {sum(r['caught'] for r in called) / max(1, len(called)):.1%}), "
          f"not called {len(held) / R:.3f}/rd")
    S = lambda name: not sections or name in sections
    if S("rate"):
        risky = [r for r in dec if not r["sure"]]
        print(rate(dec, aband, title="-- catch rate by assaf band (all Yaniv-legal decisions)"))
        print(rate(risky, lambda r: band(r["ply"], [4, 8, 12, 16, 20, 28]), title="-- not sure: by ply"))
        print(rate(risky, lambda r: r["oc"], title="-- not sure: by opp cards"))
        print(rate(risky, lambda r: r["open"], title="-- not sure: by open card points"))
        print(rate(risky, lambda r: r["pts"], title="-- not sure: by my points"))
        print(rate(risky, lambda r: r["oc"] - r["known_n"], title="-- not sure: by hidden cards"))
        print(rate(risky, lambda r: band(r["est"] - r["pts"], [0, 2, 4, 6, 9, 12, 16]), title="-- not sure: by est - pts"))
        print(rate(risky, lambda r: (aband(r), band(r["open"], [3, 6, 9])), title="-- not sure: assaf x open"))
        print(rate(risky, lambda r: (aband(r), band(r["ply"], [8, 14, 20])), title="-- not sure: assaf x ply"))
        print(rate(risky, lambda r: (aband(r), r["oc"]), title="-- not sure: assaf x opp cards"))
        print(rate(risky, lambda r: (aband(r), r["pts"]), title="-- not sure: assaf x my points"))
    if S("called"):
        for lab in ("net_throw", "hi_open", "hi_deck", "single_open"):
            rs = [r for r in recs if r["label"] == lab]
            print(table(rs, R, lambda r: "all", title=f"-- hero calls; alternative {lab}"))
            print(table(rs, R, aband, title="   by assaf band"))
            if lab in ("hi_open", "single_open"):
                low = [r for r in rs if r["open"] < r["thrown"]]
                print(table(low, R, lambda r: "open < thrown", title="   open card lower than the throw"))
                print(table(low, R, aband, title="   open lower: by assaf band"))
                nz = [r for r in low if not r["sure"]]
                print(table(nz, R, lambda r: band(r["thrown"] - r["open"], [2, 3, 4, 6]), title="   open lower, not sure: by points saved"))
                print(table(nz, R, lambda r: r["after"], title="   open lower, not sure: by hand after"))
                print(table(nz, R, lambda r: r["oc"], title="   open lower, not sure: by opp cards"))
                print(table(nz, R, lambda r: r["pts"], title="   open lower, not sure: by my points"))
                print(table(nz, R, lambda r: band(r["ply"], [8, 14, 20]), title="   open lower, not sure: by ply"))
                print(table(nz, R, lambda r: r["open"], title="   open lower, not sure: by open points"))
                print(table(nz, R, lambda r: band(r["est"] - r["pts"], [2, 4, 6, 9, 12]), title="   open lower, not sure: by est - pts"))
                print(table(nz, R, lambda r: r["known_n"], title="   open lower, not sure: by known cards"))
                print(table(nz, R, lambda r: (aband(r), band(r["thrown"] - r["open"], [2, 3, 4, 6])), title="   open lower, not sure: assaf x saved"))
            if lab == "hi_deck":
                nz = [r for r in rs if not r["sure"]]
                print(table(nz, R, lambda r: r["oc"], title="   not sure: by opp cards"))
                print(table(nz, R, lambda r: r["pts"], title="   not sure: by my points"))
                print(table(nz, R, lambda r: band(r["ply"], [8, 14, 20]), title="   not sure: by ply"))
                print(table(nz, R, lambda r: (aband(r), r["oc"]), title="   not sure: assaf x opp cards"))
    if S("held"):
        print(table(held, R, lambda r: "all", title="-- hero plays on; alternative: call"))
        print(table(held, R, aband, title="   by assaf band"))
        print(table(held, R, lambda r: r["pts"], title="   by my points"))
        print(table(held, R, lambda r: r["oc"], title="   by opp cards"))
        print(table(held, R, lambda r: band(r["ply"], [4, 8, 12, 16, 20, 28]), title="   by ply"))
        print(table(held, R, lambda r: band(r["est"] - r["pts"], [0, 2, 4, 6, 9, 12, 16]), title="   by est - pts"))
        print(table(held, R, lambda r: r["known_n"], title="   by known cards"))
        print(table(held, R, lambda r: band(r["open"], [3, 6, 9]), title="   by open points"))
        print(table(held, R, lambda r: (aband(r), r["oc"]), title="   assaf x opp cards"))
        print(table(held, R, lambda r: (aband(r), band(r["ply"], [8, 14, 20])), title="   assaf x ply"))
        print(table(held, R, lambda r: (aband(r), band(r["open"], [3, 6, 9])), title="   assaf x open"))
        print(table(held, R, lambda r: (aband(r), r["pts"]), title="   assaf x my points"))
        print(table(held, R, lambda r: (r["draw"], band(r["after"], [0, 3, 5, 8])), title="   by what the hero did instead (draw, hand after)"))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3:])


# ---------------------------------------------------------------- unified: value of calling minus the net's throw
def unified(opp, seed, tag="hm5"):
    """One row per Yaniv-legal decision: v = points margin of CALLING minus that of the hero net's best throw."""
    recs, stats = load(opp, seed, tag)
    out = []
    for r in recs:
        if r["label"] == "call":
            out.append({**r, "v": r["gain"]})
        elif r["label"] == "net_throw":
            out.append({**r, "v": -r["gain"]})
    return out, stats["rounds"]


def fine(r):
    if r["sure"]:
        return "0:sure"
    return band(r["assaf"], [0.0025, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3])


def grid(opp, seed, keyname="oc"):
    rows, R = unified(opp, seed)
    print(f"== {opp}: value of calling minus the net's best throw, by ({keyname}, assaf band); 'called' = share hm5 calls")
    groups: dict = {}
    for r in rows:
        k = r[keyname] if isinstance(keyname, str) else keyname(r)
        groups.setdefault((k, fine(r)), []).append(r)
    for k in sorted(groups, key=str):
        g = groups[k]
        if len(g) < 40:
            continue
        v = [r["v"] for r in g]
        m = sum(v) / len(v)
        se = math.sqrt(max(0.0, sum(x * x for x in v) / len(v) - m * m) / len(v))
        print(f"  {str(k):<26} n {len(g):6d} ({len(g) / R:.4f}/rd) called {sum(r['did_call'] for r in g) / len(g):4.0%} "
              f"caught {sum(r['caught'] for r in g) / len(g):5.1%}  call-throw {m:+7.2f} +-{se:.2f}")


if __name__ == "__main__" and len(sys.argv) > 3 and sys.argv[3] == "grid":
    pass


# ---------------------------------------------------------------- a candidate rule against hm5's, first order
def flips(rule, opps=("hm5", "trained", "counter"), seed=7101, name="", detail=None):
    """`rule(r) -> bool` (call?) from the position record. For each opponent: the decisions where it differs from
    hm5's choice, and the first-order gain a round (the other move played out, hm5's policy afterwards)."""
    for opp in opps:
        try:
            rows, R = unified(opp, seed)
        except FileNotFoundError:
            continue
        to_call = [r["v"] for r in rows if rule(r) and not r["did_call"]]
        to_throw = [-r["v"] for r in rows if not rule(r) and r["did_call"]]
        allv = to_call + to_throw
        se = math.sqrt(sum(x * x for x in allv)) / R
        f = lambda v: f"n {len(v):5d} ({len(v) / R:.4f}/rd) each {sum(v) / max(1, len(v)):+6.2f} = {sum(v) / R:+.4f}/rd"
        print(f"  {name:<14} vs {opp:<8} now calls: {f(to_call)} | now throws: {f(to_throw)} | total {sum(allv) / R:+.4f} +-{se:.4f}")
        if detail:
            groups: dict = {}
            for r in rows:
                if rule(r) != r["did_call"]:
                    groups.setdefault((("call" if rule(r) else "throw"), detail(r)), []).append(r["v"] if rule(r) else -r["v"])
            for k in sorted(groups, key=str):
                g = groups[k]
                if len(g) >= 40:
                    m = sum(g) / len(g)
                    s = math.sqrt(max(0.0, sum(x * x for x in g) / len(g) - m * m) / len(g))
                    print(f"      {str(k):<34} n {len(g):5d} each {m:+6.2f} +-{s:.2f}")
