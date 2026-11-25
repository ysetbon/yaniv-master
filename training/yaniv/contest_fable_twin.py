"""Thread `twin` (team fable): the 'keep the high single whose twin the opponent will throw' idea.

Probes for contest_fable_lab.paired, the mechanism count (`mech`), and the candidate designs (`build`).
Nothing here fits or searches a weight: it plays, counts, and builds nets from stated constants.

    python -m training.yaniv.contest_fable_twin which HERO OPP --rounds N --seed S   # which single to throw
    python -m training.yaniv.contest_fable_twin take  HERO OPP --rounds N --seed S   # open twin: take or deck
    python -m training.yaniv.contest_fable_twin mech  HERO OPP --rounds N --seed S   # how often the twin comes
"""

from __future__ import annotations

import argparse
import math
import pickle
import random
from collections import Counter
from pathlib import Path

from .engine import POINTS_OF, apply_action, cards_from_mask, legal_actions, make_view, new_round, rank_of

LOGS = Path(__file__).resolve().parents[2] / "results" / "contest-fable-logs"
HIGH = 5            # a 'high single' is worth at least this (classes split further by points)


def rank_counts(view):
    """(mine, gone incl. the open card, known to the opponent, unseen) copies per rank."""
    mine, gone, known = Counter(), Counter(), Counter()
    for c in cards_from_mask(view.my_hand):
        mine[rank_of(c)] += 1
    for c in view.discard:
        gone[rank_of(c)] += 1
    for c in cards_from_mask(view.opp_known):
        known[rank_of(c)] += 1
    unseen = {r: 4 - mine[r] - gone[r] - known[r] for r in range(13)}
    return mine, gone, known, unseen


def _pts(rank: int) -> int:
    return POINTS_OF[rank]          # card id r (0..12) has rank r


# ---------------------------------------------------------------- probe 1: which high single to throw
def probe_which(view, actions, chosen, state):
    """The hero throws a single of a rank it holds once (worth >= HIGH) against 4-5 cards, and holds another
    such single: alternative = throw the other one, same draw. Described a priori by points and unseen copies."""
    if chosen.kind != "play" or view.opp_size not in (4, 5):
        return []
    thrown = cards_from_mask(chosen.discard_mask)
    if len(thrown) != 1:
        return []
    mine, gone, known, unseen = rank_counts(view)
    c = rank_of(thrown[0])
    if c >= 13 or mine[c] != 1 or _pts(c) < HIGH:
        return []
    open_rank = rank_of(view.discard[-1])
    singles = [r for r in range(13) if mine[r] == 1 and _pts(r) >= HIGH]
    out = []
    for a in singles:
        if a == c:
            continue
        alt = next((x for x in actions if x.kind == "play" and x.draw == chosen.draw
                    and len(cards_from_mask(x.discard_mask)) == 1 and rank_of(cards_from_mask(x.discard_mask)[0]) == a),
                   None)
        if alt is None:
            continue
        info = {"pc": _pts(c), "pa": _pts(a), "uc": unseen[c], "ua": unseen[a], "kc": known[c], "ka": known[a],
                "gc": gone[c], "ga": gone[a], "oc": view.opp_size, "ply": view.ply, "draw": chosen.draw,
                "open_c": open_rank == c, "open_a": open_rank == a, "nhs": len(singles),
                "open_pairs": chosen.draw == "open" and mine[open_rank] > 0,
                "mc": sum(mine.values()), "paired": any(v >= 2 for v in mine.values())}
        out.append(("which", alt, info))
    return out


# ---------------------------------------------------------------- probe 2: the open card is my single's twin
def probe_take(view, actions, chosen, state):
    """The open card has the rank of a single I keep this turn (I throw something else): alternative = the same
    throw with the other draw. gain is always (other draw) - (played); `took` says which one was played."""
    if chosen.kind != "play" or view.opp_size not in (4, 5):
        return []
    mine, gone, known, unseen = rank_counts(view)
    open_rank = rank_of(view.discard[-1])
    if open_rank >= 13 or mine[open_rank] != 1:
        return []
    thrown = cards_from_mask(chosen.discard_mask)
    if rank_of(thrown[0]) == open_rank:
        return []
    other = "stock" if chosen.draw == "open" else "open"
    alt = next((x for x in actions if x.kind == "play" and x.discard_mask == chosen.discard_mask
                and x.top == chosen.top and x.draw == other), None)
    if alt is None:
        return []
    kept = [c for c in cards_from_mask(view.my_hand) if not (chosen.discard_mask >> c) & 1]
    info = {"took": chosen.draw == "open", "p": _pts(open_rank), "u": unseen[open_rank], "oc": view.opp_size,
            "ply": view.ply, "nthrown": len(thrown), "pthrown": POINTS_OF[thrown[0]],
            "kept_pts": sum(POINTS_OF[c] for c in kept), "kept_n": len(kept)}
    return [("take", alt, info)]


# ---------------------------------------------------------------- mechanism: how often does the twin come?
def _mech_job(job):
    from .contest_fable_lab import player

    hero_name, opp_name, n, seed = job
    hero, opp = player(hero_name), player(opp_name)
    rng = random.Random(seed)
    recs = []
    for i in range(n):
        hs = i % 2
        st = new_round([0, 0], (i // 2) % 2, rng)
        turns = []          # per hero turn: dict rank -> record
        pending = None      # records of the hero's last turn waiting for the opponent's reply
        last_took = None    # (rank, record) if the hero just took a twin
        init_rank = rank_of(st.discard[-1])
        thrown_by = (Counter(), Counter())      # ranks thrown so far by (hero, opponent)
        while not st.terminal and st.ply < 48:
            v = make_view(st, st.turn)
            acts = legal_actions(v)
            if not acts:
                break
            if st.turn != hs:
                a = opp(v, acts, None)
                if pending is not None:
                    for r, rec in pending.items():
                        if a.kind == "yaniv":
                            rec["end"] = 1
                        else:
                            rec["opp1"] = int(rank_of(cards_from_mask(a.discard_mask)[0]) == r)
                            rec["opp_n"] = len(cards_from_mask(a.discard_mask))
                if a.kind == "play":
                    thrown_by[1][rank_of(cards_from_mask(a.discard_mask)[0])] += len(cards_from_mask(a.discard_mask))
                apply_action(st, a)
                continue
            a = hero(v, acts, None)
            thrown = cards_from_mask(a.discard_mask) if a.kind == "play" else []
            tr = rank_of(thrown[0]) if thrown else -1
            if last_took is not None:                       # did I throw the pair I made last turn?
                r, rec = last_took
                rec["pair_thrown"] = int(tr == r and len(thrown) >= 2)
                rec["called"] = int(a.kind == "yaniv")
                last_took = None
            if pending is not None and a.kind == "play":    # did I take the twin the opponent just threw?
                open_rank = rank_of(v.discard[-1])
                for r, rec in pending.items():
                    if rec.get("opp1"):
                        rec["took"] = int(a.draw == "open" and tr != r and open_rank == r)
                        rec["threw_it"] = int(tr == r)
                        if rec["took"]:
                            last_took = (r, rec)
            pending = None
            if a.kind != "play":
                apply_action(st, a)
                continue
            thrown_by[0][tr] += len(thrown)
            apply_action(st, a)
            if st.terminal:
                continue
            v2 = make_view(st, hs)                          # what I know after my move
            if v2.opp_size not in (4, 5):
                turns.append({})
                continue
            mine, gone, known, unseen = rank_counts(v2)
            singles = [r for r in range(13) if mine[r] == 1 and _pts(r) >= HIGH]
            pending = {}
            for r in singles:
                rec = {"p": _pts(r), "u": unseen[r], "k": known[r], "oc": v2.opp_size, "ply": v2.ply,
                       "nhs": len(singles), "g": gone[r], "g_me": thrown_by[0][r], "g_opp": thrown_by[1][r],
                       "g_init": int(init_rank == r), "mc": sum(mine.values()), "opp1": 0, "end": 0, "turn": len(turns),
                       "thrown_p": POINTS_OF[thrown[0]], "thrown_same_p": int(POINTS_OF[thrown[0]] == _pts(r))}
                pending[r] = rec
            turns.append(pending)
        # two-turn window: the opponent throws the rank on its next turn or the one after (still my single then)
        for t, per in enumerate(turns):
            for r, rec in per.items():
                nxt = turns[t + 1].get(r) if t + 1 < len(turns) else None
                rec["opp2"] = int(rec["opp1"] or bool(nxt and nxt["opp1"]))
                recs.append(rec)
    return recs, n


def mech(hero: str, opp: str, rounds: int, seed: int, workers: int = 2):
    from .contest_fable_lab import _pool

    size = max(1, rounds // (workers * 4))
    jobs = [(hero, opp, min(size, rounds - at), seed * 7919 + i) for i, at in enumerate(range(0, rounds, size))]
    with _pool(workers) as pool:
        parts = list(pool.map(_mech_job, jobs))
    return [r for p, _ in parts for r in p], sum(n for _, n in parts)


def rate_table(recs, rounds, key, fields=("opp1", "opp2", "end"), least=50) -> str:
    groups: dict = {}
    for r in recs:
        groups.setdefault(key(r), []).append(r)
    lines = []
    for k in sorted(groups, key=str):
        g = groups[k]
        if len(g) < least:
            continue
        cells = []
        for f in fields:
            p = sum(x.get(f, 0) for x in g) / len(g)
            cells.append(f"{f} {p:6.1%} +-{math.sqrt(p * (1 - p) / len(g)):.1%}")
        lines.append(f"  {str(k):<44} n {len(g):6d} ({len(g) / rounds:.3f} a round)  " + "  ".join(cells))
    return "\n".join(lines)


# ---------------------------------------------------------------- command line
def pclass(p: int) -> str:
    return "10" if p == 10 else "8-9" if p >= 8 else "5-7"


def which_report(recs, rounds) -> str:
    from .contest_fable_lab import summarize

    clean = [r for r in recs if not r["open_c"] and not r["open_a"]]
    out = [f"records {len(recs)}, clean (open card is neither rank) {len(clean)}"]

    def sgn(x):
        return "+" if x > 0 else "-" if x < 0 else "0"

    def by(title, key, rs=clean, least=30):
        out.append(title)
        out.append(summarize(rs, rounds, key=key, least=least))

    by("A. all alternatives, by opp cards", lambda r: f"oc{r['oc']}")
    by("B. by (opp cards, points alt - points played, unseen alt - unseen played)  [alt throws a, keeps c]",
       lambda r: f"oc{r['oc']} dp{sgn(r['pa'] - r['pc'])} du{r['ua'] - r['uc']:+d}")
    by("C. equal points only (two ten-cards etc.): unseen alt - unseen played, opp cards",
       lambda r: f"oc{r['oc']} du{r['ua'] - r['uc']:+d}", [r for r in clean if r["pa"] == r["pc"]])
    by("D. THE IDEA: alt throws a LOWER single and keeps the played (higher) one, by exact (dp, uc, ua), oc 5",
       lambda r: f"dp{r['pa'] - r['pc']:+d} keep u{r['uc']} throw u{r['ua']}",
       [r for r in clean if r["pa"] < r["pc"] and r["oc"] == 5])
    by("D4. same, oc 4", lambda r: f"dp{r['pa'] - r['pc']:+d} keep u{r['uc']} throw u{r['ua']}",
       [r for r in clean if r["pa"] < r["pc"] and r["oc"] == 4])
    by("E. idea, alt keeps the HIGHER and MORE LIVE single (pa < pc, uc > ua), by kept points class, dp, ply",
       lambda r: f"oc{r['oc']} keep {pclass(r['pc'])} dp{max(r['pa'] - r['pc'], -3):+d} ply{'<4' if r['ply'] < 4 else '4-7' if r['ply'] < 8 else '8+'}",
       [r for r in clean if r["pa"] < r["pc"] and r["uc"] > r["ua"]])
    by("F. alt keeps the higher single, by unseen copies of the KEPT rank only (uc), oc, ply<6 / 6+",
       lambda r: f"oc{r['oc']} keep u{r['uc']} {'early' if r['ply'] < 6 else 'late'}",
       [r for r in clean if r["pa"] < r["pc"]])
    by("G. known: the opponent is KNOWN to hold the played rank (kc>0) or the alt rank (ka>0)",
       lambda r: f"oc{r['oc']} kc{min(r['kc'], 1)} ka{min(r['ka'], 1)} dp{sgn(r['pa'] - r['pc'])}",
       [r for r in clean if r["kc"] or r["ka"]])
    by("H. gone: copies gone of the played / alt rank, equal points",
       lambda r: f"oc{r['oc']} gc{r['gc']} ga{r['ga']}", [r for r in clean if r["pa"] == r["pc"]])
    by("I. open card IS the alt's rank and it lies open (alt throws the open card's twin), by draw",
       lambda r: f"oc{r['oc']} draw {r['draw']} dp{sgn(r['pa'] - r['pc'])}", [r for r in recs if r["open_a"]])
    by("J. open card is the PLAYED rank", lambda r: f"oc{r['oc']} draw {r['draw']} dp{sgn(r['pa'] - r['pc'])}",
       [r for r in recs if r["open_c"]])
    return "\n".join(out)


def take_report(recs, rounds) -> str:
    from .contest_fable_lab import summarize

    out = [f"records {len(recs)}; gain = (the other draw) - (the played draw)"]
    out.append("by (opp cards, played draw, twin's points class)")
    out.append(summarize(recs, rounds, key=lambda r: f"oc{r['oc']} played {'TAKE' if r['took'] else 'deck'} twin {pclass(r['p']) if r['p'] >= 5 else '<5'}"))
    out.append("by exact points of the twin, played draw (oc 4-5)")
    out.append(summarize(recs, rounds, key=lambda r: f"played {'TAKE' if r['took'] else 'deck'} twin {r['p']:2d}"))
    out.append("high twins (>= 8), by what I throw this turn (single / set) and ply")
    out.append(summarize([r for r in recs if r["p"] >= 8], rounds,
                         key=lambda r: f"played {'TAKE' if r['took'] else 'deck'} throw n{r['nthrown']} ply{'<4' if r['ply'] < 4 else '4-7' if r['ply'] < 8 else '8+'}"))
    return "\n".join(out)


def mech_report(recs, rounds) -> str:
    out = [f"single-turn records {len(recs)} over {rounds} rounds ({len(recs) / rounds:.2f} held high singles a round, "
           "counted once per hero turn)",
           "opp1 = the opponent's very next throw is of that rank; opp2 = next throw or the one after (I still hold "
           "it as a single); end = it called Yaniv instead"]
    out.append("by points class, opp cards")
    out.append(rate_table(recs, rounds, lambda r: f"oc{r['oc']} {pclass(r['p'])}"))
    out.append("by exact points (oc 5)")
    out.append(rate_table([r for r in recs if r["oc"] == 5], rounds, lambda r: f"p{r['p']:2d}"))
    out.append("by points class and unseen copies (known = 0)")
    out.append(rate_table([r for r in recs if not r["k"]], rounds, lambda r: f"oc{r['oc']} {pclass(r['p'])} u{r['u']}"))
    out.append("opponent KNOWN to hold the rank")
    out.append(rate_table([r for r in recs if r["k"]], rounds, lambda r: f"oc{r['oc']} {pclass(r['p'])} known"))
    out.append("by ply (ten-cards, oc 5)")
    out.append(rate_table([r for r in recs if r["oc"] == 5 and r["p"] == 10], rounds,
                          lambda r: f"ply {min(r['ply'], 10) // 2 * 2:2d}+ u{r['u']}"))
    out.append("only turns where I hold >= 2 high singles (the choice exists), ten-cards and 8-9")
    out.append(rate_table([r for r in recs if r["nhs"] >= 2 and r["p"] >= 8], rounds,
                          lambda r: f"oc{r['oc']} {pclass(r['p'])} u{r['u']} k{min(r['k'], 1)}"))
    hit = [r for r in recs if r["opp1"]]
    out.append(f"when the twin came (opp1, n {len(hit)}, {len(hit) / rounds:.3f} a round): what I did")
    out.append(rate_table(hit, rounds, lambda r: f"oc{r['oc']} {pclass(r['p'])}", fields=("took", "threw_it")))
    took = [r for r in hit if r.get("took")]
    out.append(f"after I took it (n {len(took)}, {len(took) / rounds:.3f} a round): pair thrown on my next turn / I called")
    out.append(rate_table(took, rounds, lambda r: f"oc{r['oc']} {pclass(r['p'])}", fields=("pair_thrown", "called")))
    return "\n".join(out)


def main() -> None:
    from . import contest_fable_lab as lab

    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["which", "take", "mech"])
    parser.add_argument("hero")
    parser.add_argument("opp")
    parser.add_argument("--rounds", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--save", default="")
    args = parser.parse_args()
    head = f"## {args.cmd}: hero {args.hero} vs {args.opp}, {args.rounds} rounds, seed {args.seed}"
    if args.cmd == "mech":
        recs, n = mech(args.hero, args.opp, args.rounds, args.seed, args.workers)
        text = mech_report(recs, n)
    else:
        recs, stats = lab.paired(args.hero, args.opp, f"contest_fable_twin:probe_{args.cmd}", args.rounds, args.seed,
                                 workers=args.workers)
        n = stats["rounds"]
        text = (which_report if args.cmd == "which" else take_report)(recs, n)
        text = f"hero margin {stats['margin'] / n:+.3f} a round\n" + text
    if args.save:
        with open(LOGS / args.save, "wb") as fh:
            pickle.dump(recs, fh)
    print(head + "\n" + text, flush=True)


if __name__ == "__main__":
    main()
