# clone: variant log (official test unless noted: 2000 rounds seed 41000, 800 matches seed 41001)

1. v1 clone (exact rebuild of the counter: E over draw of points-(1-w)*maxgroup, exact 1/n via one-hot of n, top penalty, call = sure OR exact assaf <= thr(pts)). Diagnostic: 100% of throw decisions equal the counter's up to exact ties in its (badness, points) key; all Yaniv disagreements are the counter's 36-sample assaf noise.
   official: margin +0.27 a round, rounds won 1011/2000 (50.5%), Yaniv calls 999 of which caught 69 (6.9%); matches 423-377 (52.9%, 95% range 49.4%-56.3%)
2. wshift: w (weight of current points) shifted one card count up (opp 4 -> 0.40, 3 -> 0.62, <=2 -> 0.75), reasoning that in the 4-card game an opponent with N cards is as close to calling as one with N+1 in the 5-card game the counter was tuned for.
   official: margin -0.53 a round, rounds won 950/2000 (47.5%), Yaniv calls 944 of which caught 67 (7.1%); matches 376-424 (47.0%, 95% range 43.6%-50.5%)  -> wrong direction, speed matters more.
3. thr15: call thresholds x1.5 (break-even true risk for a call is ~25%: a clean call wins ~12, a caught one costs pts+30).
   official: margin -0.03 a round, rounds won 1019/2000 (50.9%), Yaniv calls 1039 of which caught 95 (9.1%); matches 412-388 (51.5%, 95% range 48.0%-54.9%)
4. thr06: call thresholds x0.6 (opponents keep low cards so true risk > uniform assaf).
   official: margin +0.28 a round, rounds won 979/2000 (48.9%), Yaniv calls 937 of which caught 48 (5.1%); matches 404-396 (50.5%, 95% range 47.0%-54.0%)
   Diagnostic after 3/4 (true catch rate measured from the real hidden hand at every legal-Yaniv turn): true risk ~ 3-4x uniform assaf (assaf <1%: 3.6%, 1-2%: 7.6%, 2-4%: 11%, 4-7%: 20%, 7-12%: 28%); the counter's thresholds already sit at true risk ~20%, near break-even, so threshold scale is flat.
5. callB2: +2 x P(hand after draw <= 7, i.e. may call next turn), exact (deck: one-hot on kept points and on n). Reason: a callable hand can end the round next turn.
   official: margin +0.25 a round, rounds won 1021/2000 (51.0%), Yaniv calls 1014 of which caught 76 (7.5%); matches 454-346 (56.8%, 95% range 53.3%-60.1%)  -> no margin change.
6. wlow: w lowered (opp <=2 -> 0.50, 3 -> 0.30, 4 -> 0.12), the opposite of 2: shedding speed (keep) dominates.
   official: margin +0.42 a round, rounds won 1036/2000 (51.8%), Yaniv calls 1032 of which caught 68 (6.6%); matches 454-346 (56.8%, 95% range 53.3%-60.1%)
7. wlow_call2: wlow + new call rule: call if sure, or assaf <= 0.5*thr(pts), or (assaf <= 1.75*thr(pts) and opp_est - my_pts >= 3). Reason: measured true risk at fixed assaf is ~2x lower when the lean-low estimate of the opponent's total is 3+ above mine (assaf 2-5%: 12% vs 24%).
   official: margin +0.49 a round, rounds won 1040/2000 (52.0%), Yaniv calls 1034 of which caught 67 (6.5%); matches 446-354 (55.8%, 95% range 52.3%-59.2%)
   Diagnostic (oracle): 1000 positions, every move rolled out 80x (counter vs counter after it, hidden cards resampled leaning low). Mostly noise-level regrets; one real pattern: leaving face up a rank the opponent is KNOWN to hold costs ~2.5 extra (n=10). Diagnostic (oppcall): P(opponent calls on its next turn) depends mainly on opp_est: est<5: 40-70%, 5-8: ~29%, 8-11: ~15-23%, 11-14: ~8%, 14-17: ~5%, 17+: <3%; its calling total averages ~5.
8. estw: w chosen by opp_est bucket (cuts 5/8/11/14/17 points) instead of by card count, w = P(opponent calls next) + 0.05 = 0.60/0.35/0.23/0.13/0.09/0.06. Reason: the weight of current points should be the chance I get caught holding them.
   official: margin +0.75 a round, rounds won 1086/2000 (54.3%), Yaniv calls 1090 of which caught 77 (7.1%); matches 456-344 (57.0%, 95% range 53.5%-60.4%)
9. estw_catch30: + 30 * pi_b * P(my hand after the draw <= m_b) (pi = 0.55/0.30/0.18/0.08/0.04/0.01, m = 3/5/5/6/6/6): if the opponent calls and I am at or under its total, it is Assaf (~+35 swing). Exact P via one-hot of kept points and n.
   official: margin +1.03 a round, rounds won 1103/2000 (55.1%), Yaniv calls 1089 of which caught 76 (7.0%); matches 480-320 (60.0%, 95% range 56.6%-63.3%)
10. c30_feed: 9 + penalty (1 + 0.5 * points) for leaving face up a rank the opponent is known to hold (oracle pattern above; it lets the opponent take a set-maker). unit feed_r = relu(throw_r + 4*known_r - 1).
   official: margin +1.08 a round, rounds won 1106/2000 (55.3%), Yaniv calls 1091 of which caught 77 (7.1%); matches 472-328 (59.0%, 95% range 55.6%-62.4%)
11. c30_feed_call2: 10 + the call rule of 7 (sure, or assaf <= 0.5 thr, or assaf <= 1.75 thr and opp_est - my_pts >= 3).
   official: margin +1.18 a round, rounds won 1103/2000 (55.1%), Yaniv calls 1079 of which caught 72 (6.7%); matches 482-318 (60.2%, 95% range 56.8%-63.6%)
12. best_stocklean: 11 + deck card worth 0.6 * (h*avg - hidden estimate)/(n-3) more than the unseen average (the stock excludes the opponent's low hidden cards; 0.6 because a high drawn card is often the next throw).
   official: margin +1.18 a round, rounds won 1102/2000 (55.1%), Yaniv calls 1077 of which caught 73 (6.8%); matches 479-321 (59.9%, 95% range 56.4%-63.2%)  -> no effect, dropped.
13. pi2: 11 with pi (and w = pi + 0.05) per (opp_est bucket x opponent card count), from the oppcall table (e.g. est 8-11: 2 cards 0.23, 3 cards 0.15, 4 cards 0.08).
   official: margin +1.12 a round, rounds won 1100/2000 (55.0%), Yaniv calls 1078 of which caught 74 (6.9%); matches 483-317 (60.4%, 95% range 56.9%-63.7%)  -> no gain over 11, dropped.
14. best_callB2: 11 + 2 x P(hand after draw <= 7) (retry of 5 now that w is lower: own-call value of points may be missing).
   official: margin +1.00 a round, rounds won 1086/2000 (54.3%), Yaniv calls 1072 of which caught 76 (7.1%); matches 472-328 (59.0%, 95% range 55.6%-62.4%)  -> worse, dropped.

FINAL: build() = variant 11 (BEST in clone.py), saved as nets/clone.json (re-run official: identical +1.18 line). Pure clone saved as nets/clone_pure.json.
