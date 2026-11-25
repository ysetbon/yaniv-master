"""Thread `survey`: where does a design's move differ from hm5's? (plain play of hm5 against hm5, both nets asked)
    python -m training.yaniv.contest_fable_survey_check 'py:contest_fable_survey_design.build' ROUNDS SEED [show]"""
import collections, random, sys
from .contest_fable_lab import player
from .engine import POINTS_OF, apply_action, cards_from_mask, legal_actions, make_view, new_round, rank_of
from .features import action_key

N = "A23456789TJQK*"


def main(design, rounds, seed, show=6):
    base, new = player("hm5"), player(design)
    rng = random.Random(seed)
    diff = collections.Counter(); total = 0; shown = 0
    for i in range(rounds):
        st = new_round([0, 0], i % 2, rng)
        while not st.terminal and st.ply < 48:
            v = make_view(st, st.turn)
            acts = legal_actions(v)
            a, b = base(v, acts, None), new(v, acts, None)
            total += 1
            if action_key(a) != action_key(b):
                hand = "".join(N[rank_of(c)] for c in sorted(cards_from_mask(v.my_hand), key=rank_of))
                o = N[rank_of(v.discard[-1])]
                pair_of_open = hand.count(o)
                kind = f"holds {pair_of_open} of the open rank, opp {v.opp_size}: hm5 {action_key(a)[1:] } -> {action_key(b)[1:]}"
                diff[(pair_of_open, v.opp_size, a.kind, b.kind, a.draw, b.draw)] += 1
                if shown < show:
                    shown += 1
                    print(f"  hand {hand} open {o} opp {v.opp_size} ply {v.ply}: hm5 {action_key(a)} -> design {action_key(b)}")
            apply_action(st, a)
    print(f"{design}: {sum(diff.values())} of {total} decisions differ from hm5 ({rounds} rounds, seed {seed})")
    for k, n in diff.most_common(12):
        print(f"   (holds n of open rank, opp cards, hm5 kind, design kind, hm5 draw, design draw) = {k}: {n}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
