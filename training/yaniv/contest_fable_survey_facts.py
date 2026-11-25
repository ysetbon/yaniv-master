"""Thread `survey`: base facts by plain play (no fitting): what each player takes from the open pile, round length,
showdowns.   python -m training.yaniv.contest_fable_survey_facts A B ROUNDS SEED"""
import random, sys, collections
from .contest_fable_lab import player
from .engine import (POINTS_OF, apply_action, cards_from_mask, legal_actions, make_view, new_round, rank_of,
                     resolve_showdown, mask_points, popcount)


def run(a, b, n, seed):
    rng = random.Random(seed)
    ch = (player(a), player(b))
    take = collections.Counter(); seen = collections.Counter()
    ends = collections.Counter(); plies = []; empties = 0
    for i in range(n):
        st = new_round([0, 0], i % 2, rng)
        emptied = False
        while not st.terminal and st.ply < 48:
            v = make_view(st, st.turn)
            acts = legal_actions(v)
            act = ch[st.turn](v, acts, None)
            if act.kind == "play" and st.turn == 1:      # player B's draws
                o = v.discard[-1]
                kept = [rank_of(c) for c in cards_from_mask(v.my_hand & ~act.discard_mask)]
                inhand = [rank_of(c) for c in cards_from_mask(v.my_hand)]
                k = (POINTS_OF[o], "pairs" if rank_of(o) in inhand else "no", "first" if st.ply == 0 or (st.ply == 1) else "")
                seen[k] += 1
                take[k] += act.draw == "open"
            apply_action(st, act)
            if not st.stock:
                emptied = True
        if not st.terminal:
            resolve_showdown(st)
        ends[st.result.reason] += 1
        plies.append(st.ply); empties += emptied
    print(f"{a} vs {b}: {n} rounds seed {seed}; ends {dict(ends)}; stock emptied in {empties}; "
          f"ply mean {sum(plies)/n:.1f}, share ply>=20 {sum(p>=20 for p in plies)/n:.3f}, >=30 {sum(p>=30 for p in plies)/n:.3f}, >=40 {sum(p>=40 for p in plies)/n:.4f}")
    print(f"what {b} takes from the open pile (points, holds that rank):")
    for k in sorted(seen):
        if seen[k] >= 20:
            print(f"   {k}: {take[k]}/{seen[k]} = {take[k]/seen[k]:.2f}")


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
