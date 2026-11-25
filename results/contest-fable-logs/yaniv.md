# yaniv: when to call Yaniv and when to play on (rounds from 0-0)

Thread `yaniv` of team fable. Files: `training/yaniv/contest_fable_yaniv.py` (the design: `build`, `rule`,
`check`), `contest_fable_yaniv_probe.py` (the paired probe), `contest_fable_yaniv_tab.py` (tables),
`contest_fable_yaniv_rules.py` (candidate rules as functions of a position record, for the first-order check).
Scratch (pickles of records, ad hoc table scripts `zone.py two.py twoh.py thr.py swap.py swap2.py r1..r5.py`):
`/tmp/claude-0/-home-user-yaniv-master/ca258bc3-d453-5237-b2d9-a66c415800ab/scratchpad`.
Nothing here fits or searches weights: every number is a count over played rounds, and the rule's constants
are hand-picked from the tables below (every variant tried is listed in "Every measurement").

## Method

`python -m training.yaniv.contest_fable_yaniv_probe run OPP 60000 7101` = `paired("hm5", OPP,
"contest_fable_yaniv_probe:calls", 60000, 7101, workers=2)`. At every decision of the hero (hm5) where Yaniv is
legal:
- hm5 calls: the alternatives `net_throw` (hm5's own best throw row: what the net does if the Yaniv row is
  switched off), `hi_open` (throw the highest-points group, take the open card), `hi_deck` (same throw, deck),
  `single_open`;
- hm5 plays on: the alternative `call`.
Both moves are played to the end of the round on the same hidden cards, hm5's policy afterwards. `v` below =
points margin of CALLING minus that of hm5's best throw (one number per Yaniv-legal decision, either side).
Every record also has the true hands, so "caught" = the opponent's total <= mine (what a call would meet).

Yaniv is legal 0.87 times a round; hm5 calls 0.50 times a round (caught 6.8% by hm5, 7.4% by trained).

## 1. The true catch rate of a call against the inputs (seed 7101, 60,000 rounds per opponent)

All Yaniv-legal decisions that are not sure (`sure_yaniv` = 0), whoever would have called. Cell = share of them
where the opponent's total <= mine, against hm5 / trained / counter. `.` = under 60 cases.

| opp cards | assaf <0.0025 | <0.005 | <0.01 | <0.02 | <0.03 | <0.05 | <0.08 | <0.12 | <0.2 | <0.4 |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 3.4 / 2.9 / 2.1% | . / . / 12.9% | . | . | . | . | . | . | . | . |
| 4 | 3.0 / 3.3 / 2.3% | 8.7 / 9.2 / 8.0% | 15.5 / 14.5 / 12.7% | 19.0 / 18.2 / 15.5% | 26.2 / 23.8 / 18.1% | 26.2 / 28.4 / 21.2% | . | . | . | . |
| 3 | 1.2 / 1.2 / 0.4% | 3.0 / 3.5 / 2.5% | 7.3 / 7.8 / 5.2% | 11.8 / 12.5 / 11.3% | 17.5 / 18.7 / 15.1% | 25.2 / 25.5 / 21.3% | 34.1 / 31.2 / 25.5% | 35.7 / 36.6 / 35.0% | 46.7 / 49.2 / 41.7% | 60.9 / 66.1 / 68.4% |
| 2 | . | . | 7.3 / 6.9 / .% | 4.8 / 6.6 / 1.4% | 7.5 / 6.9 / 5.3% | 10.4 / 14.5 / 9.4% | 13.4 / 21.6 / 12.0% | 23.4 / 27.1 / 23.8% | 40.5 / 38.0 / 36.3% | 67.8 / 59.3 / 59.7% |
| 1 | . | . | . | . | . | . | . | . | . / 14.8 / .% | 34.1 / 30.3 / 31.5% |

Reading: `assaf` (hidden cards uniform) understates the real rate by a factor that depends almost only on the
opponent's card count: about x20-25 against 4-5 cards, x6-8 against 3, x2-2.5 against 2, x1 against 1. The
three opponents agree to within a few points (the counter's hands are a little higher, the trained net's
2-card hands lower). hm5's overall figure (6.8% caught) hides this because most calls are at assaf < 0.005.

Other inputs at fixed assaf (hm5 as the opponent; "not sure" decisions):
- ply: assaf 0.02-0.03 is caught 10% before ply 8, 13% at ply 8-13, 21% at 14-19, 24% from ply 20. Same shape
  in every band.
- my points: at assaf 0.03-0.05, 9-11% at 2-3 points, 18% at 4, 24-26% at 5-7.
- est - pts: 66% under 0, 50% at 0-2, 22% at 2-4, 13% at 4-6, 6% at 6-9, 2% at 9-12 (it is the best single
  predictor, but see part 3: a rule on it alone was worse than one on assaf by card count).

**The open-card hypothesis** (a low open card, just thrown by the opponent, says its other cards are low).
Decisions near the calling thresholds (4 cards assaf 0.005-0.02, 3 cards 0.015-0.05, 2 cards 0.04-0.12):

| open card | hm5 | trained | counter |
|---|---|---|---|
| 0-2 | 18.8% (mean assaf 5.8%, n 1002) | 29.3% (6.2%, 1369) | 19.1% (4.8%, 414) |
| 3-4 | 20.6% (5.2%, 3754) | 24.9% (4.8%, 4201) | 18.4% (4.1%, 3007) |
| 5-6 | 19.4% (3.9%, 2681) | 19.9% (4.1%, 2851) | 17.2% (3.4%, 2611) |
| 7-9 | 18.3% (3.8%, 3421) | 18.5% (4.1%, 3550) | 14.3% (3.1%, 2119) |
| 10 | 16.7% (3.9%, 3891) | 17.0% (4.2%, 4060) | 13.7% (3.2%, 2377) |

True in direction for all three, but weak against hm5 once the mean assaf of each row is allowed for (5.8% vs
3.9%); clear only against the trained net (29% vs 17%). Not used as a risk term; the swap rule below covers
most low-open positions anyway.

Same zone by ply: under 10: 12.2 / 12.6 / 11.4%; 10-13: 16.9 / 16.7 / 14.9%; 14-17: 22.2 / 22.0 / 19.8%;
18-21: 23.1 / 24.5 / 17.0%; 22+: 17.1 / 28.2 / 16.8%. By known cards: 0 known 20.6 / 21.6 / 18.4%; 1 known
14.2 / 19.4 / 13.6%.

## 2. Is hm5's rule wrong anywhere? v = calling minus hm5's best throw (points a round-end, same cards)

By (opponent cards, assaf band); "called" = share of the cell hm5 calls today. hm5 / trained / counter.

| cell | times a round | called | v vs hm5 | v vs trained | v vs counter |
|---|---|---|---|---|---|
| any, sure | 0.11 | 100% | +5.0 to +13.3 | +3.5 to +11.1 | +2.9 to +7.0 |
| 5 cards, < 0.0025 | 0.029 | 100% | +5.66 +-0.36 | +5.84 +-0.32 | +5.42 +-0.26 |
| 4 cards, < 0.0025 | 0.084 | 100% | +6.58 +-0.22 | +7.04 +-0.21 | +5.29 +-0.21 |
| 4 cards, 0.0025-0.005 | 0.031 | 100% | +4.32 +-0.45 | +5.01 +-0.44 | +3.31 +-0.42 |
| 4 cards, 0.005-0.01 | 0.025 | 100% | +1.21 +-0.54 | +1.78 +-0.52 | +0.91 +-0.49 |
| 4 cards, 0.01-0.015 | 0.008 | 100% | +0.29 +-0.96 | +1.38 +-0.85 | -0.16 +-0.81 |
| **4 cards, 0.015-0.02** | 0.004 | 96-98% | **-4.65 +-1.61** | **-3.39 +-1.50** | -1.99 +-1.34 |
| **4 cards, 0.02-0.03** | 0.003 | 65-70% | **-3.86 +-1.83** | -2.73 +-1.57 | -2.52 +-1.37 |
| 3 cards, < 0.005 | 0.037 | 100% | +8.3, +8.8 | +9.2, +8.9 | +6.6, +6.2 |
| 3 cards, 0.005-0.01 | 0.030 | 100% | +5.69 +-0.43 | +6.87 +-0.44 | +4.23 +-0.51 |
| 3 cards, 0.01-0.015 | 0.029 | 100% | +4.39 +-0.48 | +4.60 +-0.48 | +2.08 +-0.57 |
| 3 cards, 0.015-0.02 | 0.029 | 100% | +3.24 +-0.50 | +4.43 +-0.50 | +1.58 +-0.56 |
| 3 cards, 0.02-0.03 | 0.045 | 82-85% | +1.70 +-0.42 | +2.28 +-0.41 | +0.29 +-0.45 |
| 3 cards, 0.03-0.04 | 0.031 | 32-35% | -0.99 +-0.52 | -0.38 +-0.54 | -1.39 +-0.59 |
| 3 cards, 0.04-0.06 | 0.034 | 5-6% | -4.06 +-0.54 | -3.01 +-0.52 | -4.43 +-0.57 |
| 2 cards, 0.02-0.04 | 0.013 | 96-99% | +2.8, +3.8 | +4.5, +3.5 | +0.8, -1.7 |
| 2 cards, 0.04-0.06 | 0.016 | 50-66% | +1.61 +-0.61 | +0.78 +-0.55 | +0.33 +-0.85 |
| **2 cards, 0.06-0.08** | 0.030 | 7-8% | **+0.84 +-0.39** | -0.01 +-0.49 | +1.11 +-0.66 |
| 2 cards, 0.08-0.10 | 0.020 | 0-1% | +0.14 +-0.58 | -0.66 +-0.55 | -1.99 +-0.99 |
| 2 cards, 0.10-0.15 | 0.045 | 0% | -2.93 +-0.44 | -2.61 +-0.38 | -4.86 +-0.67 |
| 1 card, 0.10-0.20 | 0.006 (trained) | 0% | . | +3.1 +-1.4, +1.9 +-1.0 | . |
| 1 card, 0.20-0.30 | 0.004-0.02 | 0% | -4.41 +-1.39 | -0.47 +-0.51 | -7.07 +-2.88 |

So hm5's rule is close to right in this grid. Break-even assaf: about 0.015 against 4 cards, 0.03-0.035
against 3, 0.08-0.09 against 2 (0.07 against the trained net, 0.08 against the counter), about 0.2 against 1.
The break-even TRUE catch rate is about 20% everywhere (a caught call costs ~45 points, but when the opponent
is at or under my total it usually calls next turn itself, so playing on is no rescue).
hm5's errors: it calls too much against 4 cards at assaf 0.015-0.03 (0.007 a round, -3 to -4.6 each), too
little against 2 cards at 0.05-0.09 and against 1 card.

**Inside the marginal zone** (hm5 and the trained net as opponents, pooled; the swap positions of part 2b and
assaf < 0.005 removed) the break-even assaf moves with:
- my points, about proportionally: against 3 cards 0.015-0.02 at 3 points, 0.022 at 4, 0.027 at 5, 0.037 at 6-7;
  against 2 cards 0.065 at 3, 0.085 at 4, 0.11 at 5, 0.13 at 6, 0.15 at 7.
- the ply: against 3 cards over 0.05 before ply 12, 0.033 at 12-17, about 0.02 at 18-23; against 2 cards 0.15 /
  0.10 / 0.075 / 0.06.
- est - pts: larger gap, call more (same information as ply + points + known cards).
- the open card, after the swap positions are removed: small (0.03 against 0.035-0.04 with 3 cards).

### 2b. The 'play on' class: a lower hand is one swap away

hm5 calls, the call is not sure, and my highest-points group (input `best_shed`) is worth >= 2 points more than
the open card. Alternative = hm5's own best throw (the net then throws that group and takes the open card in
~90% of the cases, checked):

| class | hm5 | trained | counter |
|---|---|---|---|
| best_shed - open >= 2, assaf >= 0.005 | 0.0084 a round, **+4.26 +-0.91**, +0.036 a round | 0.0093, **+4.35 +-0.93**, +0.040 | 0.0085, **+3.96 +-0.85**, +0.034 |
| the same, assaf < 0.005 | 0.0042, -0.23 +-1.09 | 0.0046, +0.95 +-1.23 | 0.0063, -0.23 +-0.79 |
| by points saved 2 / 3 / 4 (assaf >= 0.005) | +2.7 / +5.9 / +9.9 | +2.8 / +5.2 / +13.1 | +3.1 / +5.6 / +9.9 |
| against 3 / 4 cards | +3.2 / +6.2 | +3.1 / +6.3 | +4.1 / +4.1 |

A priori alternatives measured directly (hm5 opponent, 60,000 rounds; trained in brackets):
- `hi_open`, open card lower than the throw, not sure, all: +0.52 +-0.43 (+0.53 +-0.43), 0.035 a round. By
  points saved: 1: -0.83 +-0.53 (-1.45 +-0.52); 2: +1.54 +-0.85 (+2.88 +-0.88); 3: +4.46 +-1.58 (+4.89 +-1.47);
  4+: +7.9 +-2.3 (+10.5 +-2.4). Saving ONE point is not worth it. Throwing a SET to take a lower open card:
  -3.86 +-1.00 (-4.05 +-0.98): wrong; the gain is in single throws, +1.67 +-0.47 (+1.82 +-0.47).
- when the call would have been caught the swap gains +37; when not, it loses 4.5 (saved >= 2) or 6.5-7.5 (saved 1).
- `hi_deck` (throw the highest group, draw from the deck) instead of a call: -5.45 +-0.10 over all calls; only
  positive in the cells of part 2 (4 cards, assaf 0.01-0.03: +0.96 +-0.84 and +4.67 +-2.14). No 'deck' class.
- `single_open` when the highest group is a set: -10.9 +-0.27. Never.

## 3. Rules tried on the records (first order: the decisions where the rule differs from hm5's, the other move
played out with hm5's policy afterwards; `contest_fable_yaniv_tab.flips`; seed 7101, in-sample for hm5 and
trained, the counter was not looked at while choosing)

| rule | vs hm5 | vs trained | vs counter |
|---|---|---|---|
| X only: hm5's rule + the swap veto (best_shed - open >= 2 and assaf >= 0.005: play on) | +0.036 +-0.008 | +0.040 +-0.009 | +0.034 +-0.007 |
| A: assaf < T[oc], T = .005/.015/.03/.08/.2 for 5/4/3/2/1 cards | +0.086 +-0.020 | +0.053 +-0.022 | (not run) |
| B: est - pts > G[hidden cards] (3.7 / 3.2 / 5 / 8 for 2 / 3 / 4 / 5; 1 hidden: assaf < 0.1) | +0.025 +-0.019 | -0.026 +-0.020 | |
| B2: est - pts > G[opp cards] (2.8 / 3.5 / 5 / 8) | +0.060 +-0.018 | +0.024 +-0.019 | |
| D: assaf < T0[oc] + S[oc] (est - pts - C[oc]) | +0.090 +-0.019 | +0.046 +-0.021 | |
| D + X | +0.116 +-0.020 | +0.079 +-0.022 | |
| A' + X: T = a[oc] x 5, a = .04/.021/.0055/.0021 for 1/2/3/4 cards, no modifiers | +0.097 +-0.026 | +0.042 +-0.029 | +0.034 +-0.021 |
| pts + X: T = a[oc] x pts | +0.141 +-0.026 | +0.105 +-0.028 | +0.035 +-0.022 |
| ply + X: T = a[oc] x (5 + 0.35 (15 - clip(ply, 8, 20))) | +0.164 +-0.026 | +0.092 +-0.028 | +0.031 +-0.021 |
| F: both in full, T = a[oc] x (pts + 0.35 (15 - clip(ply))) | +0.095 +-0.030 | +0.034 +-0.031 | +0.012 +-0.024 |
| V1: F with the ply term only before ply 15 | +0.131 +-0.029 | +0.101 +-0.030 | +0.006 +-0.024 |
| **V2: both at half weight, T = a[oc] x (5.5 + 0.5 pts - 0.2 clip(ply, 8, 20))** + X | **+0.181 +-0.026** | **+0.127 +-0.027** | **+0.042 +-0.022** |

Reading. The swap veto X is the robust part: the same +0.035 a round against all three opponents. The
threshold part is worth about +0.09-0.14 against the two nets and nothing against the counter (its break-evens
sit a little lower: 3 cards 0.03, 2 cards 0.08). F double-counts (points and ply are correlated), hence the
half weights of V2. V2's sub-classes: against 4 cards it stops calling (0.012-0.014 a round, +3.7 / +2.5 / +2.2
each vs hm5 / trained / counter); against 3 cards it stops calling late or with few points (+1.9 / +0.8 / +1.1)
and calls more early with 6-7 points (+1.4 / +3.4 / -1.2); against 2 cards it calls more (0.044 a round,
+1.9 / +0.9 / +0.2). I stopped after V2 (two variants after F) to keep this hand-picking and not a search.

## 4. The design: `py:contest_fable_yaniv.build` (565 hidden units, contest_fable has 558)

Call Yaniv iff
1. sure_yaniv, or
2. assaf < 0.005, or
3. NOT (best_shed - open card >= 2 points) AND assaf < a[opp cards] x (5.5 + 0.5 x my points - 0.2 x clip(ply, 8, 20)),
   a = 0.04 / 0.021 / 0.0055 / 0.0021 against 1 / 2 / 3 / 4 cards (against 5 cards only reasons 1-2),
4. or contest_fable's score reason (a caught call lands my score on 50 / 100), untouched.
hm5's reader threshold (early, gap, gap_cap, high_call, call, call_sat) and C1 (c1_a, c1_b) are removed.

Units (17 of mine against hm5's 9, +7 in the built net with carries): layer 1 `sure_call`; `ylow_a/b` = step
[assaf < 0.005] on the Yaniv row; `yply_lo/hi` = relu(ply - 8), relu(ply - 20); `yswap_a/b` = step
[40 best_shed - 10 open_pts >= 2]; layer 2 `ycall{1,2,3,4}_a/b` = step on 2000 (T - assaf), gated by the Yaniv row,
the opponent's card count and the swap veto. Reasons are summed and saturated by contest_fable.finish_yaniv
as before (Yaniv row -1000 / +1000).

Check (`python -m training.yaniv.contest_fable_yaniv`, 2,000 rounds against hm5, seed 11): 1,743 Yaniv-legal
decisions, 1,060 calls, **0 mismatches** between the net's move and `rule(view)`; Kit.trace is used to print the
reasons of any mismatch (none). In the 83 positions where the swap condition holds and the net plays on, it
throws its highest group and takes the open card 80 times.

## 5. Duels of the design (rounds from 0-0, mirrored pairs, `contest_fable_lab duel ... --workers 2`)

| seed 31, 40,000 rounds | design | hm5 on the same deals | difference |
|---|---|---|---|
| vs hm5 (head to head) | **+0.133 +-0.028**, rounds won 51.6%, calls caught 6.8% | 0 by symmetry | +0.133 |
| vs trained | +0.150 +-0.058, 52.8%, caught 7.6% | +0.096 +-0.057, 51.4%, caught 7.1% | +0.054 |
| vs counter | +1.773 +-0.068, 57.3%, caught 4.7% | +1.653 +-0.068, 56.3%, caught 4.7% | +0.120 |

(lines in `yaniv-duels.txt`)

## 6. Out-of-sample check with the design as the hero (seed 7202, 60,000 rounds each against hm5 and trained)

`FABLE_YANIV_HERO=py:contest_fable_yaniv.build python -m training.yaniv.contest_fable_yaniv_probe run OPP 60000 7202 v2`
(the alternative to a call is now the design's own best throw, the design's policy afterwards). Round margin of
these runs: +0.115 against hm5, +0.084 against trained; calls 0.52 / 0.54 a round, caught 6.7% / 7.6%.

v = calling minus throwing, by assaf / T (T = the design's threshold; under 1 the design calls), positions
without the swap veto, assaf >= 0.005, not sure. hm5 / trained:

| assaf / T | 2 cards | 3 cards | 4 cards |
|---|---|---|---|
| < 0.4 | +5.41 +-0.67 / +6.66 +-0.63 | +5.48 +-0.38 / +6.46 +-0.38 | +4.55 +-1.15 / +4.04 +-1.26 |
| 0.4-0.6 | +2.33 +-0.69 / +3.06 +-0.63 | +4.09 +-0.42 / +4.15 +-0.44 | +2.14 +-0.80 / +2.86 +-0.74 |
| 0.6-0.8 | +1.39 +-0.63 / +1.58 +-0.57 | +2.22 +-0.49 / +2.05 +-0.49 | +0.57 +-1.00 / +0.33 +-0.94 |
| 0.8-1.0 | +0.89 +-0.62 / +0.61 +-0.56 | +1.23 +-0.56 / +0.88 +-0.55 | -0.56 +-1.14 / -1.57 +-1.13 |
| 1.0-1.25 | -1.38 +-0.53 / -1.18 +-0.55 | -0.69 +-0.61 / -0.17 +-0.60 | +1.02 +-1.12 / -1.05 +-1.21 |
| 1.25-1.6 | -3.82 +-0.56 / -3.94 +-0.50 | -2.72 +-0.62 / -3.11 +-0.60 | -5.99 +-1.59 / -3.46 +-1.52 |
| 1.6-2 | -8.59 +-0.71 / -5.98 +-0.60 | -4.40 +-0.72 / -3.93 +-0.68 | -4.34 +-1.78 / -4.34 +-1.79 |
| 2-3 | -14.7 / -10.3 | -7.1 / -6.9 | -4.5 / -5.4 |

The sign changes at assaf / T = 1 for 2 and 3 cards against both opponents, on rounds not used to choose the
constants. Against 4 cards the 0.8-1.0 band is slightly negative (a = 0.0018 would be a hair better; not
changed, it is 0.005 a round and inside one error). Against 1 card: 1.25-2 T: -1.3 +-1.5 / -2.7 +-0.9 (trained),
-5.6 +-1.4 (hm5): the 0.04 is not too low.

Swap-veto positions (the design plays on): calling instead is worth -14.5 +-0.5 (hm5) / -12.4 +-0.4 (trained)
over all of them (0.042 / 0.057 a round, most would not be called anyway); among those the threshold alone
would have called (assaf < T): 3 cards 0.5-1 T: -4.04 +-1.63 / -4.39 +-1.43; 2 cards: -3.72 +-2.56 / +0.40 +-2.13;
4 cards: -0.54 +-2.47 / -3.99 +-2.07; 3 cards under 0.5 T: +1.17 +-1.70 / -0.40 +-1.65 (a wash: the veto could
be dropped under 0.5 T, not worth units).

Inside 0.6-1.6 T, split by the inputs not in the rule, the design's calls stay >= 0 and its non-calls <= 0 in
every row but two within one error: by open card (called: 0-2: +2.1 / -0.5; 3-4: +0.1 / +0.2; 5-6: +0.8 / +1.0;
7+: +2.1 / +1.9; not called: -1.2 to -2.9), by ply, by my points, by known cards (called with 1 known card:
+0.59 +-0.56 / -0.73 +-0.55). No further class reaches 2 errors and 0.03 a round: nothing more to take here.

## 7. Second seed and matches

`duel py:contest_fable_yaniv.build hm5 --rounds 40000 --matches 6000 --seed 32 --workers 2`:
margin **+0.091 +-0.029** a round, rounds won 51.4%, calls caught 6.7%; **matches 3004-2996 (50.1%, 95% 48.8-51.3%)**.

Head to head against hm5 over both seeds (80,000 rounds): **+0.112 +-0.020 a round**. In 6,000 matches the
gain does not show (50.1% +-1.3): +0.11 points a round is about a tenth of hm5's edge over the counter per
round, which is worth ~20 points of matches there, so the expected match effect is ~+0.5 to +1 point, under
this test's resolution. Honest reading: a real but small round gain; no measurable match gain; no sign of harm.
