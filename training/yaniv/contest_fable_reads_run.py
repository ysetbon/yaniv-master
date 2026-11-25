"""Thread `reads`: run `paired` with the probe that offers hm5's move where the hero (a reads design) differs.

    python -m training.yaniv.contest_fable_reads_run 'py:contest_fable_reads.build' hm5 20000 71 [out.pkl]

gain = margin with hm5's move minus margin with the design's move: NEGATIVE means the design's move is better.
"""
import pickle
import sys

from .contest_fable_lab import paired, summarize

if __name__ == "__main__":
    hero, opp, n, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    records, stats = paired(hero, opp, "contest_fable_reads:probe_hm5", n, seed, workers=2)
    rounds = stats["rounds"]
    print(f"{hero} vs {opp}: {rounds} rounds seed {seed}, margin {stats['margin'] / rounds:+.3f}, "
          f"{len(records)} decisions where hm5 would move differently ({len(records) / rounds:.4f} a round)")
    print(" all:"); print(summarize(records, rounds, key=lambda r: "all"))
    print(" by regime:"); print(summarize(records, rounds, key=lambda r: "opp cards " + r["reg"]))
    print(" by my cards after my move:"); print(summarize(records, rounds, key=lambda r: f"k_hero {r['k_hero']}"))
    print(" by open low:"); print(summarize(records, rounds, key=lambda r: f"open_low {r['open_low']}"))
    print(" by kind of change:")
    def kind(r):
        h, a = r["hero_move"], r["alt_move"]
        if h["draw"] != a["draw"] and (h["rank"], h["n"]) == (a["rank"], a["n"]):
            return f"same throw, hero draws {h['draw']}"
        if h["n"] != a["n"]:
            return f"hero throws {h['n']} cards, hm5 {a['n']}"
        return "other throw: hero " + ("higher" if h["pts"] > a["pts"] else "lower" if h["pts"] < a["pts"] else "equal") + " points"
    print(summarize(records, rounds, key=kind))
    if len(sys.argv) > 5:
        with open(sys.argv[5], "wb") as fh:
            pickle.dump((records, stats), fh)
