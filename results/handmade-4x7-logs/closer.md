# closer: log
Task: diagnose why scorer.add_score_units pays ~2x more on merge_exact than merge_small, then write closer.py.

## Diagnosis
Tools: SCRATCH/closer/diag.py (plays mirrored match pairs vs the counter, seed 60001 - not the dev seed - and counts cut
landings by how the round ended + decision events), SCRATCH/closer/mk_noscore.py (both finalists without score units:
closer/nets/me_noscore.json, ms_noscore.json). Implementation check: merge_small's and merge_exact's add_score_units are
weight-for-weight identical (same tents, threat, deck cut, assaf-on-cut); merge_small feeds assaf-on-cut as a call reason
into its saturating unit (+1000), merge_exact adds +400 to a -200/+200 row: both force the call. So the difference is
not the implementation.
Diagnostic 1 (closer/diag1.txt, 3,000 matches each, seed 60001; per match; ms = merge_small, me = merge_exact, 0 = no score units):
                                    ms     ms0     me     me0
  matches won                      0.641  0.598  0.695  0.611   (score units: ms +4.3, me +8.4 points)
  rounds / match                   12.97  12.61  13.66  12.68
  my decisions / match             72.9   71.0   77.9   72.6
  open-card cut take available     3.98   3.89   4.13   4.05   (same opportunities)
  open-card cut take CHOSEN        0.72   0.22   1.55   0.25   <- the difference: me takes 2.2x as many
  opponent called while I held cut 0.25   0.10   0.48   0.11
  my cut landings at 50 / 100      .25/.06 .08/.04 .46/.10 .08/.03
  assaf-on-cut available/called    .22/.22 .21/.13 .25/.25 .23/.13  (same)
  held cut, Yaniv legal / called   .16/.10         .18/.08  (both call ~half the time they hold the cut)
  my calls / caught                6.75/.35        6.69/.27
  opponent cut landings            0.14            0.15     (my calls hand the counter a cut ~0.11/match in both)
Diagnostic 2 (closer/gap.py: 2,339 positions from ms games where an open-card cut take exists; both nets score them):
  take rate by opp_est  <9: ms .52 me .87 | 9-12: ms .30 me .76 | 12-15: ms .10 me .34 | 15+: ms .07 me .13
  take rate by open card (opp_est<12): 3-5 pts: ms .60-.75 me .78-.89 | 8: .29/.83 | 9: .21/.77 | 10: ms .10 me .78
  merge_small's output gap (best move - cut take), mean by term family where it declines: held points 6.5-7.2,
  OPEN_TELL 7.2-8.3 (1.5 per point above 3 of an unpaired open card: 10.5 for a 10), heavy charge 1.5-3.5,
  vs the cut bonus 5-15. merge_exact has no open-card tell (its h3 heavy charge is 3x(1-w), ~0.6 for a 10).
  => CAUSE: merge_small's OPEN_TELL charge (+ its larger flat CARD charge) outweighs the cut bonus for every
  open card worth 6+, so it takes half as many cut landings. The Yaniv rule matters little (opportunities 3.98
  vs 4.13 a match, +4%; rounds/match 12.97 vs 13.66 partly BECAUSE of the cuts). Implementations are identical.
Match-win table check (scorer/table.npy, counter-v-counter W(my, opp)): value of a cut in points-equivalent =
  dW(cut) / local dW per point: 50-cut ~51/44/43/30/24 points at opp score 10/30/50/70/90; 100-cut ~55-57 at
  every opp score. So scorer's "half weight at 100" is not supported: the 100-cut is worth at least the 50-cut.
v1 TELL_ON_CUT (OPEN_TELL given back on a cut-landing open take). Why: diagnosis. dev matches only (rounds unchanged: 0-0 starts never land on a cut): matches 2010-990 (67.0%, 95% range 65.3%-68.7%)  (base merge_small 64.4%)
  check (closer/gap2.py, same 2,339 positions, Yaniv-best rows dropped): cut-take rate v1 vs me by opp_est <9 .89/.91,
  9-12 .71/.82, 12-15 .32/.37, 15+ .10/.13 -> parity with merge_exact restored up to the small heavy-charge difference.
  diag (seed 60001): cut takes 1.29/match (ms 0.72, me 1.55), landings at 50/100 .37/.10, matches 66.6%.
v2 = v1 + 100-cut at the 50-cut's weight: OPEN_CUT 100 (1,9)->(1.5,18), DECK_CUT 100 10->20. Why: the match-win table
  says a 100-cut is worth ~55 points-equivalent at every opponent score (more than the 50-cut), not half.
  dev matches only: matches 2064-936 (68.8%, 95% range 67.1%-70.4%)  -> +1.8 (noise-level, positive, kept)
v3 = v2 + cut veto: on the Yaniv row, if my score + my points is 50/100 (lands_on_cut) and opp_est < 10, -3000 (no call,
  even a sure one). Why: if the opponent calls next turn (~40% at opp_est <= 9) my score is cut, ~+45 net of the round
  (=> ~+18 expected), vs ~3-5 points of round value lost by not ending a round I am winning.
  dev matches only: matches 2065-935 (68.8%, 95% range 67.2%-70.5%)  -> neutral (event is rare: ~0.17 a match)
  second seed (diag.py, 3,000 matches seed 60001): v1 66.6% -> v2 68.6%, v3 68.5%; v3's veto changes only 0.015 calls
  a match (holding the cut with Yaniv legal AND opp_est < 10 is rare) -> v3 DROPPED (no measurable effect, extra units).
v4 = v2 + opponent-cut veto: no call when every opponent card is known and opp_score + known points is 50/100 (my win
  would add its hand and cut its score; ~45 points to it). diag seed 60001: matches 68.7% (v2 68.6%), opponent cut
  landings from my calls 0.113 -> 0.108 a match: the all-known case is too rare -> DROPPED. (A partial-knowledge version
  would be ~break-even: P(last hidden card is the exact value) ~10% x 45 vs ~4 points of round value lost by waiting.)
v5 = v2 + C1_BASE 1.3 -> 2.3 (merge_exact's tighter C1: calls with < 1 point of C1 margin are caught ~14% and playing on is worth as much).
  dev: margin +1.33 a round, rounds won 6433/12000 (53.6%), Yaniv calls 6055 of which caught 259 (4.3%); matches 2080-920 (69.3%, 95% range 67.7%-71.0%)
  v2 full dev: margin +1.32 a round, rounds won 6532/12000 (54.4%), Yaniv calls 6237 of which caught 309 (5.0%); matches 2064-936 (68.8%, 95% range 67.1%-70.4%)
  -> neutral (+0.01 margin, +0.5 matches, fewer rounds won): not adopted, merge_small's call rule kept.
FINAL = v2 (clean file: veto code removed).
FINAL closer.py (python -m training.yaniv.handmade_designs.closer -> nets/closer.json, dims [125,197,86,42,30,28,48,24,7,1],
462 hidden units; byte-identical net to closer_v2.json):
  dev: margin +1.32 a round, rounds won 6532/12000 (54.4%), Yaniv calls 6237 of which caught 309 (5.0%); matches 2064-936 (68.8%, 95% range 67.1%-70.4%)
  official: margin +1.44 a round, rounds won 1104/2000 (55.2%), Yaniv calls 1049 of which caught 52 (5.0%); matches 549-251 (68.6%, 95% range 65.3%-71.7%)
  (merge_small official 521-279 65.1%; merge_exact 562-238 70.2%; trained 490-310 61.3%. Rounds identical to merge_small.)
