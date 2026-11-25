# merge_small: variant log
File: training/yaniv/handmade_designs/merge_small.py (build(**overrides); variants built by SCRATCH/merge_small/mk.py).
"dev" = handmade_eval --rounds 12000 --matches 3000 --seed-rounds 52000 --seed-matches 52001 --workers 2
("rounds only" = same with --matches 0, used for changes that cannot touch match-score play).

v0 = reader.py rebuilt in the new structure (Yaniv row now -1000/+1000 through one saturating "reasons" unit).
   Same move as reader.json on 20,000 probe positions except 50 exact ties (float tie-breaking).
   dev: margin +0.88 a round, rounds won 6171/12000 (51.4%), Yaniv calls 5804 of which caught 269 (4.6%); matches 1724-1276 (57.5%, 95% range 55.7%-59.2%)
v1c = v0 with clone's Yaniv rule (sure, or assaf <= 0.5 thr(pts), or assaf <= 1.75 thr(pts) and gap >= 3). Why: compare rules on the same throws.
   Offline on 26k probe positions: calls 61% of legal non-sure spots (reader 44%, trained 43%), true caught 10.2% (reader 7.2%).
   dev rounds only: margin +0.77 a round, rounds won 6459/12000 (53.8%), Yaniv calls 6396 of which caught 443 (6.9%)  -> worse margin (more rounds won, more Assafs)
v2c = v0 with caller's C1 (opp_est - pts > 0.4 pts + 1.3 + 10 assaf, or sure). Why: same comparison.
   Offline: calls 49%, true caught 7.6%; extra calls mostly at 1-3 points and late plies.
   dev rounds only: margin +0.89 a round, rounds won 6296/12000 (52.5%), Yaniv calls 6015 of which caught 310 (5.2%)  -> same margin as reader, more rounds won
v4catch (previous agent, exact catch steps not recorded before the container restart): v0 + catch bonus (deck/open: P(hand after draw <= 5) x opp_est-stepped weights).
   dev rounds only: margin +0.92 a round, rounds won 6197/12000 (51.6%), Yaniv calls 5828 of which caught 271 (4.6%)
(container restart here; resumed by a new agent. v3g2/v3rc1/v5pp/v6est/v8info nets were built but never measured; not counted.)
(variants below built by mk.py with the overrides shown; all on v0's Yaniv rule unless said)
v9catch {'catch':[(17,0.9),(14,1.2),(11,3.0),(8,3.6),(5,7.5)]}: v0 + clone's catch bonus 30 x pi(opp_est) x P(hand after draw <= 5), pi = 0.55/0.30/0.18/0.08/0.04 as cumulative steps. Why: a low hand turns the opponent's call into an Assaf.
   dev rounds only: margin +0.91 a round, rounds won 6200/12000 (51.7%), Yaniv calls 5834 of which caught 268 (4.6%)  -> +0.03, within noise (reader already weights held points at 1, so it already sits low)
v10feed {'feed':(1.0,0.5)}: v0 + feeding penalty (1 + 0.5 pts) per relu(throw_r + 4 known_r - 1). Why: clone's oracle rollouts: leaving a known-held rank face up costs ~2.5.
   dev rounds only: margin +0.91 a round, rounds won 6203/12000 (51.7%), Yaniv calls 5836 of which caught 260 (4.5%)  -> +0.03, within noise
v11plan {'plan':1.0}: v0 + one-step plan: + (1 - w(opp_est)) x G, G = points of my best rank group after the move (tournament max over 13 ranks; open card taken counts in its group), + E[relu(drawn card - G)] for a deck draw; w = clone's P(call next) + 0.05 by opp_est (<5/8/11/14/17). Why: -(w held + (1-w)(held - G)): unless the opponent calls first, my best group leaves next turn; reader had no notion of which card goes next (clone and planner both have it).
   dev rounds only: margin +1.01 a round, rounds won 6298/12000 (52.5%), Yaniv calls 5891 of which caught 253 (4.3%)  -> +0.13, kept
v12h5 {'plan':1.0,'soon':'est'}: v11 with heavy charge 5 x (1 - w(opp_est)) instead of 5 (3 at opp 1-2 cards). Why: merge_exact's form; the future matters less when a call is likely.
   dev rounds only: margin +1.01 a round, rounds won 6305/12000 (52.5%), Yaniv calls 5896 of which caught 253 (4.3%)  -> neutral, not kept (simpler card-count relief stays)
v13h3 {'plan':1.0,'soon':'est','card':3.0}: v12 with charge 3 (merge_exact's value; the plan now credits a kept high single, so maybe less charge needed).
   dev rounds only: margin +0.94 a round, rounds won 6286/12000 (52.4%), Yaniv calls 5893 of which caught 262 (4.4%)  -> worse, reverted to 5
(from here plan=1.0 is the default)
v14catch {'catch':[(17,0.9),(14,1.2),(11,3.0),(8,3.6),(5,7.5)]}: v11 + the catch bonus of v9. Why: retry now that the plan lowers the weight of held points (as in clone, where it was +0.28).
   dev rounds only: margin +1.12 a round, rounds won 6340/12000 (52.8%), Yaniv calls 5946 of which caught 248 (4.2%)  -> +0.11, kept
v15feed {'feed':(1.0,0.5)}: v11 + feeding penalty of v10.
   dev rounds only: margin +1.02 a round, rounds won 6323/12000 (52.7%), Yaniv calls 5913 of which caught 244 (4.1%)  -> +0.01, neutral
(from here catch=CATCH_STEPS is the default; default build checked identical to v14catch)
v16c1 {'rule':'caller'}: v14 with caller's C1 Yaniv rule (opp_est - pts > 0.4 pts + 1.3 + 10 assaf, or sure). Why: re-compare rules on the new throws (v2c on v0 was +0.89 vs +0.88).
   dev rounds only: margin +1.14 a round, rounds won 6472/12000 (53.9%), Yaniv calls 6174 of which caught 297 (4.8%)  -> +0.02, more rounds won
v17or {'rule':'reader_or_c1'}: v14 with Yaniv = sure OR reader's (assaf < T(ply, gap, pts)) OR caller's C1. Why: the blend: reader's rule takes calls whose uniform risk is tiny whatever the gap, C1 takes calls where the opponent looks clearly higher than me whatever assaf (caller found opp_est - pts separates good and bad calls better); each misses the other's safe calls.
   dev rounds only: margin +1.15 a round, rounds won 6492/12000 (54.1%), Yaniv calls 6213 of which caught 308 (5.0%)  -> +0.03, kept (best so far; rounds won now = trained net's 54.3%)
(from here rule='reader_or_c1' is the default; default build checked identical to v17or)
v17or full dev: margin +1.15 a round, rounds won 6492/12000 (54.1%), Yaniv calls 6213 of which caught 308 (5.0%); matches 1748-1252 (58.3%, 95% range 56.5%-60.0%)
v18score {'score':True}: v17 + scorer.add_score_units (copied, same weights; "assaf on cut" is a call REASON fed to the saturating Yaniv unit, so it always gives +YBIG = 1000 and nothing else can outvote it). Why: aim to hold the points that land my score on 50/100 (cut), and call when even a caught call lands on the cut.
   dev: margin +1.16 a round, rounds won 6501/12000 (54.2%), Yaniv calls 6224 of which caught 308 (4.9%); matches 1869-1131 (62.3%, 95% range 60.6%-64.0%)  -> matches +4 points (trained net on dev: 60.5%), kept
(from here score=True is the default; default build checked identical to v18score)
diagnostic (SCRATCH/merge_small/diag.py, 25k recorded positions): v18 takes an open card that pairs a kept card 98-99% at every value (trained net 84-93%, reader 69-100%): the plan's G already credits the pair (it goes out next turn), so PAIR_OPEN=9 now double-counts.
v19po5 {'pair_open':5.0}: PAIR_OPEN 9 -> 5 = CARD: what is left of the bonus is merge_exact's exemption (a card taken onto its twin leaves with it next turn, so it is not a lasting heavy card); the set value itself is in the plan. Take rate of pairing cards becomes 93-97%.
   dev rounds only: margin +1.28 a round, rounds won 6513/12000 (54.3%), Yaniv calls 6222 of which caught 311 (5.0%)  -> +0.12, kept
(from here PAIR_OPEN = 5 is the default)
v20tr0 {'triple':0.0}: drop reader's triple bonus (+5 for an open card that makes a triple): the plan's G now counts the third card of the group.
   dev rounds only: margin +1.28 a round, rounds won 6506/12000 (54.2%), Yaniv calls 6217 of which caught 313 (5.0%)  -> neutral; adopted for simplicity (14 units gone; rebuilt net differs from v20tr0 in 9/10,000 positions, float ties)
v21feed {'feed':(1.0,0.5)}: v20 + feeding penalty (1 + 0.5 pts) for relu(throw_r + 4 known_r - 1) (retry on the new base).
   dev rounds only: margin +1.32 a round, rounds won 6534/12000 (54.4%), Yaniv calls 6238 of which caught 305 (4.9%)  -> +0.04 (third small non-negative reading: +0.03 on v0, +0.01 on v11), kept
(from here feed=FEED is the default; default build identical to v21feed)
v22 = v21 full dev: margin +1.32 a round, rounds won 6534/12000 (54.4%), Yaniv calls 6238 of which caught 305 (4.9%); matches 1932-1068 (64.4%, 95% range 62.7%-66.1%)
FINAL (clean rewrite of v22, no variant switches; variant-capable copy kept at SCRATCH/merge_small/merge_small_variants.py, used by mk.py):
   same units except: catch bonus computes P(hand <= 5) once and gates it by opp_est steps (12 fewer units), the est cuts are "< t" for both plan and catch (v22's catch used "<= t"),
   and fold_output sums the output terms into a running total (pos/neg pair per layer) instead of carrying each term: dims [125,197,84,42,30,28,48,24,7,1], 460 hidden units (v22: 1000).
   Same move as v22 on 19,952/20,000 probe positions (Yaniv decision identical everywhere).
   dev: margin +1.32 a round, rounds won 6532/12000 (54.4%), Yaniv calls 6237 of which caught 309 (5.0%); matches 1931-1069 (64.4%, 95% range 62.6%-66.1%)
   official: margin +1.44 a round, rounds won 1104/2000 (55.2%), Yaniv calls 1049 of which caught 52 (5.0%); matches 521-279 (65.1%, 95% range 61.8%-68.3%)
   (trained net: dev +1.39, 54.3%, matches 60.5%; official +1.52, 55.4%, matches 61.3%)
