"""Thread `survey`: systematic scan. Every decision is put in a situation class written before looking
(hand shape by group points order, how the open card relates to the hand, opponent's card count) and every
alternative gets a canonical label (which group, how many of it, which draw). Prints the cells where the
alternative beats the net's move.   python -m ...survey_scan [--fine] name=FILE ..."""
import math, sys
from .contest_fable_survey_rules import PTS, load, pb


def groups(P):
    return sorted(P.cnt, key=lambda r: (-PTS[r] * P.cnt[r], P.unseen[r], r))


def mlabel(P, g, m):
    i = g.index(m[0])
    return f"g{i + 1}:{m[1]}/{P.cnt[m[0]]}:{'D' if m[2] == 'stock' else 'O'}"


def situation(P, g, fine):
    shape = ",".join(str(P.cnt[r]) for r in g)
    if P.open in P.cnt:
        o = f"open=g{g.index(P.open) + 1}"
    else:
        p = PTS[P.open]
        o = "open" + ("<=2" if p <= 2 else "3" if p == 3 else "4" if p == 4 else "5+")
    oc = "opp5" if P.oc == 5 else "opp4" if P.oc == 4 else "opp1-3"
    s = f"[{shape}] {o} {oc}"
    if fine:
        s += " " + "/".join(pb(PTS[r]) for r in g)
    return s


def scan(sets, fine=False, least=60):
    table = {}
    for opp, (pos, rounds) in sets.items():
        for P in pos:
            g = groups(P)
            sit = situation(P, g, fine) + " net " + mlabel(P, g, P.ch)
            for m, gain in P.g.items():
                table.setdefault((sit, mlabel(P, g, m)), {}).setdefault(opp, []).append(gain)
    rows = []
    for key, per in table.items():
        cells, ok, tot = [], True, 0.0
        for opp, (_, rounds) in sets.items():
            x = per.get(opp, [])
            if len(x) < least:
                ok = False
                break
            mean = sum(x) / len(x)
            se = math.sqrt(max(1e-9, sum(v * v for v in x) / len(x) - mean * mean) / len(x))
            if mean < 2.0 * se or mean * len(x) / rounds < 0.004:
                ok = False
                break
            tot += mean * len(x) / rounds
            cells.append(f"{opp}: n {len(x):5d} each {mean:+6.2f}+-{se:.2f} rd {mean * len(x) / rounds:+.4f}")
        if ok:
            rows.append((tot, f"{key[0]:<50} -> {key[1]:<10} | " + " | ".join(cells)))
    for _, line in sorted(rows, reverse=True):
        print(line)
    print(f"({len(table)} cells looked at; shown: mean >= 2 se and >= 0.004 a round against every opponent given)")


if __name__ == "__main__":
    fine = "--fine" in sys.argv
    sets = {a.split("=")[0]: load(a.split("=")[1]) for a in sys.argv[1:] if "=" in a}
    scan(sets, fine)
