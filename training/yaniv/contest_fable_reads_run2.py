"""Thread `reads`: hm5 against an opponent; at each throw the alternative is hm5's own second choice.
gain > 0: the second choice was better. Records pickled for grouping.

    python -m training.yaniv.contest_fable_reads_run2 hm5 10000 81 out.pkl
"""
import pickle
import sys

from .contest_fable_lab import paired, summarize

if __name__ == "__main__":
    opp, n, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    records, stats = paired("hm5", opp, "contest_fable_reads:probe_second", n, seed, workers=2)
    rounds = stats["rounds"]
    print(f"hm5 vs {opp}: {rounds} rounds seed {seed}, margin {stats['margin'] / rounds:+.3f}, {len(records)} records")
    print(summarize(records, rounds, key=lambda r: "gap " + ("<1" if r["gap"] < 1 else "1-3" if r["gap"] < 3 else "3+")))
    with open(sys.argv[4], "wb") as fh:
        pickle.dump((records, stats), fh)
