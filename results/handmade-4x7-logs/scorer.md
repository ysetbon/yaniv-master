# scorer: variants log
Design file: training/yaniv/handmade_designs/scorer.py (base = add_throw_units + add_yaniv_units; score = add_score_units).
"official" = handmade_eval default (2,000 rounds seed 41000, 800 matches seed 41001, --workers 2).
"diag" = SCRATCH/scorer/diag.py: mirrored rounds vs the counter started from score pairs sampled from 500
counter-vs-counter matches (0-0 excluded), valued by a match-win table (match_value.win_table on 6,000
counter-vs-counter rounds); paired against the first net; "pp a round" = change in match-win chance.
Rough match prediction = pp a round x 11.7 non-0-0 rounds a match.

Analysis before building (no weights from it): table of W(m,o); break-even Assaf chance for a call
vs "status quo" G/(G+A) is ~0.22-0.28 almost everywhere, rising to 0.3-0.49 only at my score >= 90 —
i.e. the call threshold barely needs to depend on the score. Frequencies (counter v counter, per match,
both players): Yaniv-legal turns 18.9; Assaf would land on 50/100: 0.34 (counter calls 2/3 of them);
an open-card take landing my score+hand on 50/100 available 7.5 (within 2 badness points of the
counter's choice: 1.7, within 5: 3.7); "desperate" (>=80 and >=15 behind) Yaniv-legal 1.6.

Base development (agreement with the counter's move on 3,446 positions, not games):
- B0 throws = -(points after draw) + 0.78*shed(next throw) - top penalty: agreement 62.7% (took the open card too often)
- B1 + deck_up = 0.6*relu(9.5 - high) on deck draws (a deck card may become the highest): 82.7% (now too few open takes)
- B2 deck_up = 0.5*relu(9 - high): 84.7%; Yaniv decisions agree 95.6%. Kept.

Game-measured variants:
1. scorer_base (B2 alone) — the reference.
   official: margin -0.12 a round, rounds won 1002/2000 (50.1%), Yaniv calls 993 of which caught 75 (7.6%); matches 380-420 (47.5%, 95% range 44.1%-51.0%)
2. scorer v1 = base + open_cut 8 flat (both 50 and 100) + assaf_cut (+400) + desperate (thr +0.10). Reasoning: landing on
   the cut when the opponent calls saves 50; a caught call that lands on 50/100 is no loss; far behind near 100, variance helps.
   official: margin -0.12 a round, rounds won 1002/2000 (50.1%), Yaniv calls 993 of which caught 75 (7.6%); matches 426-374 (53.2%, 95% range 49.8%-56.7%)
   diag (6000): +0.660 +- 0.119 pp a round (~+7.7 pp a match), margin -1.1 a round worse, 15% of rounds changed
3. ablations (diag 6000, vs base): only open_cut 8: +0.649 +- 0.119; only assaf_cut: +0.020 +- 0.009; only desperate: +0.011 +- 0.014.
   -> the open-card cut is the whole effect; desperate is not measurable.
4. Measured P(opponent calls on its next turn | I land on the cut) by opp_est: <=9: 40%, 10-14: 14%, >14: 3%.
   oc_threat: open_cut = (flat 1.5 + 18 x threat) at 50, (1 + 9 x threat) at 100 [= 50 x P(call), half at 100 where a point is
   worth more already]: diag +0.717 +- 0.096 (margin cost halved: -0.5 a round instead of -1.0)
   oc_6_4: flat 6 at 50, 4 at 100 (no threat): diag +0.507 +- 0.097. -> threat scaling kept.
5. oc_deck = oc_threat + deck_cut (20, 10) x P(deck card worth exactly the needed value, unseen/34) x threat:
   diag vs oc_threat +0.178 +- 0.053. Kept.
6. oc_bust = oc_threat + penalty (1 + 4 x threat) when my score + held >= 100: diag -0.150 +- 0.053 — BUG: ">= 100" also
   penalised exactly 100, which is a cut. Fixed to > 100 (oc_bust101): +0.011 +- 0.027 -> no effect, dropped.
7. FINAL scorer = base + open_cut threat + deck_cut + assaf_cut (desperate, bust off).
   official: margin -0.12 a round, rounds won 1002/2000 (50.1%), Yaniv calls 993 of which caught 75 (7.6%); matches 463-337 (57.9%, 95% range 54.4%-61.3%)
   diag (10000, seed 11): +0.842 +- 0.079 pp a round (~+9.8 pp a match predicted; official shows +10.4 pp)
   matches only, other seed (--seed-matches 777): base 376-424 (47.0%) -> scorer 457-343 (57.1%)
8. Plug-in test: planner.py (as on disk at the time) with add_score_units added at build (SCRATCH/scorer/plug.py, no edit to planner.py):
   planner plain official: margin +0.79 a round, rounds won 1058/2000 (52.9%), Yaniv calls 1043 of which caught 73 (7.0%); matches 451-349 (56.4%, 95% range 52.9%-59.8%)
   planner + score official: margin +0.79 a round, rounds won 1058/2000 (52.9%), Yaniv calls 1043 of which caught 73 (7.0%); matches 506-294 (63.2%, 95% range 59.9%-66.5%)
   diag: +0.743 +- 0.087 pp a round
