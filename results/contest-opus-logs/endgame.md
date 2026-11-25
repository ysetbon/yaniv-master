# endgame: the Yaniv-call decision and play after both hands hold <= 4 cards

Designer topic `endgame`. Code: `training/yaniv/contest_opus_endgame.py` (probes `alt_call`, `alt_swap`,
`alt_trained`; block `add_endgame_units`). Scratch: `/tmp/claude-0/-home-user-yaniv-master/3c7d4d81-ef83-553b-adc8-6a5382f1abf0/scratchpad/endgame/`
(record files `alt_*-{counter,trained}.json`, grouping script `grp.py`).

Hero = hm5 (`yaniv-net-5cards-handmade.json`) in every play-out. Gain = B's margin - A's margin on the same hidden
cards, both continued with hm5's policy; > 0 means the alternative was better. "a round" = sum of gains / rounds,
+- sqrt(sum of squares) / rounds. Rounds from 0-0 (no match-score play).

## Log (every measurement, in order)

### 1. Pricing hm5's call decisions (probe `alt_call`, seed 7101, 24,000 rounds per opponent)

Every hm5 call -> alternative = hm5's own best non-call move; every legal non-call -> alternative = the call.
"eg" = both hands <= 4 cards. hm5's margin in these play-outs: +1.475 vs counter, -0.113 vs trained.

| class | vs counter: per round, each, a round | vs trained: per round, each, a round |
|---|---|---|
| call -> best non-call, all | 0.536, -4.44, -2.38 | 0.516, -5.86, -3.03 |
| ... eg, assaf = 0 | 0.142, -5.29 +- 0.24 | 0.087, -8.54 +- 0.31 |
| ... eg, 0 < assaf < 0.005 | 0.147, -5.69 +- 0.28 | 0.151, -7.44 +- 0.29 |
| ... eg, 0.005-0.01 | 0.049, -2.67 +- 0.52 | 0.054, -5.08 +- 0.51 |
| ... eg, 0.01-0.02 | 0.058, -0.93 +- 0.54 | 0.074, -3.83 +- 0.49 |
| ... eg, 0.02-0.04 | 0.043, -0.20 +- 0.65 | 0.067, -2.26 +- 0.52 |
| ... eg, >= 0.04 | 0.007, +0.76 +- 1.66 | 0.021, +0.06 +- 0.83 |
| ... not eg (a hand holds 5) | 0.092, -6.58 +- 0.30 | 0.063, -7.28 +- 0.39 |
| non-call -> call, all | 0.208, -10.35 | 0.464, -8.85 |
| ... eg, assaf 0.02-0.04 | 0.022, -2.81 +- 1.07 | 0.032, -0.51 +- 0.85 |
| ... eg, assaf >= 0.04 | 0.183, -11.42 +- 0.37 | 0.429, -9.51 +- 0.23 |

Reading: hm5's call threshold is right against both opponents. Its best non-call is usually a deck draw (worse
than calling at every assaf level). The one positive pocket is when the non-call takes an open card that is 2+
points under my highest card at assaf >= 0.01: +5.1 / +7.3 (counter), +6.5 / +17.7 (trained), 0.004-0.005 a round.

### 1b. The swap (probe `alt_swap`, seed 7101, 24,000 rounds per opponent)

Every hm5 call -> throw my highest single, take the open card if it is lower (else deck). sv = my highest card -
the open card. Open-card swaps: 0.042 a round (counter), 0.036 (trained). Cells (eg, opponent 3-4 cards):

| cell | vs counter: n, each | vs trained: n, each |
|---|---|---|
| assaf = 0, sv 1 / 2 | 201 -3.05 / 60 -2.65 | 52 -9.08 / 26 -8.62 |
| 0 < assaf < 0.005, sv 1 / 2 / 3 | 167 -1.10 / 50 -0.42 / 15 +1.73 | 135 -1.87 / 39 +1.46 / 13 -1.77 |
| 0.005-0.01, sv 1 / 2 | 77 -0.75 / 45 +3.20 | 85 +0.99 / 39 +5.74 |
| 0.01-0.02, sv 1 / 2 / 3 | 69 +5.62 / 42 +2.98 / 17 +0.71 | 109 -3.35 / 51 +4.73 / 26 +12.88 |
| 0.02-0.04, sv 1 / 2 / 3 | 75 +1.85 / 26 +9.85 / 15 +15.73 | 101 +1.09 / 27 +11.00 / 14 +26.36 |
| opponent 2 cards, assaf >= 0.01, sv >= 1 | 15 -3.33 | 42 +0.45 |

Candidate aggregates (opponent 3-4): assaf >= 0.005 and sv >= 2: counter 0.0067 a round, +4.73 each, +0.033
+- 0.010 a round; trained 0.0076, +8.68, +0.066 +- 0.014. assaf >= 0.01 and sv >= 1: +0.050 / +0.050.

### 2. hm5 vs the trained net after both hands <= 4 (probe `alt_trained`, seed 7101, 24,000 rounds per opponent)

alt = the trained net's move. Unlike hm4 (handover.md), hm5 has no clear endgame gap to the trained net:

| class | vs counter: per round, each, a round | vs trained: per round, each, a round |
|---|---|---|
| all | 0.428, +0.21, +0.089 +- 0.077 | 0.537, -0.12, -0.065 +- 0.080 |
| hm5 calls, T plays on | 0.069, +0.59, +0.041 +- 0.032 | 0.105, -0.90, -0.095 +- 0.040 |
| ... T takes an open card 2+ under my top, assaf >= 0.01 | 0.0034, +5.11 +- 1.97 | 0.0042, +10.37 +- 2.33 |
| ... T draws from the deck, assaf >= 0.01 | 0.040, +0.01 +- 0.60 | 0.064, -1.28 +- 0.49 |
| hm5 plays on, T calls | 0.014, -0.72 +- 1.26 | 0.019, +0.76 +- 1.01 |
| throws, both play on | 0.345, +0.17 +- 0.20 | 0.413, +0.04 +- 0.16 |
| ... same throw, hm5 takes open, T deck | 0.062, +0.80 +- 0.40 | 0.095, +0.30 +- 0.29 |
| ... same throw, hm5 deck, T takes open | 0.121, +0.15 +- 0.39 | 0.128, -0.41 +- 0.34 |
| ... different single / set vs single | see raw | within +-0.03 a round |
| throws by ply < 9 / 9-15 / 16+ | +0.017 / -0.009 / +0.050 +- 0.028 | +0.027 / -0.020 / +0.009 +- 0.030 |

Only the swap class is positive for both. A hint (cherry-picked from ~15 cells): "hm5 takes an open 4-5 at ply
16+, T draws": counter +0.87 +- 0.68 / +1.54 +- 1.11, trained +1.44 +- 0.58 / +1.34 +- 0.97 -> probed directly below.

### 3. Rule S, replicated on a fresh seed (probe `alt_rule`, seed 7102, 48,000 rounds per opponent)

Rule S: Yaniv legal, opponent holds 3-4 cards, assaf >= 0.005, my highest card - open card >= 2 -> throw that
card, take the open card. The probe offers the swap wherever a looser version fires (assaf >= 0, sv >= 1), so
the record also covers positions where hm5 already plays on.

| cell (hm5 calls) | vs counter: n, each | vs trained: n, each |
|---|---|---|
| assaf 0.005-0.01, sv 2 / 3+ | 69 +4.29 / 28 +4.64 | 85 +6.32 / 33 +6.82 |
| 0.01-0.02, sv 2 / 3+ | 94 +5.21 / 38 +1.24 | 89 +6.56 / 38 +9.76 |
| 0.02-0.04, sv 2 / 3+ | 63 +5.83 / 25 +8.60 | 69 +6.14 / 31 +6.71 |
| sv 1, assaf 0.005-0.01 / 0.01-0.02 / 0.02-0.04 | 128 +3.99 / 177 +4.98 / 150 +1.33 | 156 +2.94 / 214 +0.40 / 183 +1.80 |
| sv 1, assaf < 0.005 (a > 0) / = 0 | 287 -0.23 / 248 -3.35 | 288 -3.25 / 81 -9.69 |
| hm5 already plays on, sv >= 2 (rule S forces the swap) | 14, -2.0 | 9, -0.1 |

Both seeds together (hm5 calls, opponent 3-4):
- **Rule S (assaf >= 0.005, sv >= 2):** counter 0.0067 a round, +5.13 +- 1.37 (seed 7101) and +4.77 +- 1.18
  (7102); trained 0.0074 / 0.0072 a round, +8.89 +- 1.68 and +6.72 +- 1.17. **+0.032 to +0.066 a round.**
- sv = 1, assaf >= 0.005: counter +2.51 +- 1.38 / +3.66 +- 1.02; trained -0.23 +- 1.30 / +1.48 +- 0.91 (0.012 a
  round). Positive against the counter, near zero against the trained net: variant v2 tests it.
- sv >= 3, 0 < assaf < 0.005: +1.9 / +3.2 (counter), +4.3 / +6.1 (trained), 0.001 a round: too rare to bother.

### Block `add_endgame_units` (rule S), 12 hidden units

Six layer-1 units `eg_top_r` (r = 2..7: the row throws one r, takes the open card, no higher card in hand), two
ramp units for thrown - open >= save_min, two for assaf >= assaf_min (ramp 1e-5 wide), one layer-2 AND unit
`eg_swap` (with yaniv_legal and opp_has_3/4), output +2500 on that row only (the Yaniv row is +-1000; the swap
row then wins). Build check (`check.py`, 4,000 rounds vs counter): v1 plays exactly hm5's move or rule S's swap
at all 27,623 decisions (rule fired 48 times); v2 (save_min = 1) 27,669 decisions, fired 118, 0 mismatches.
(First build had a 0.0005-wide assaf ramp: 1 mismatch at assaf 0.00474; ramp narrowed.)

### Other call groupings (alt_call, seed 7101): nothing else to fix

- hm5's calls at assaf >= 0.005 (eg) against its best non-call, by my points 1..7, by opp_min - my points, by
  known opponent cards (0/1/2), by ply (0-5 ... 24+), by opp_est - my points: every group negative or within
  noise for both opponents (counter -1.20 each overall, trained -3.27). Only opp_est - my points < 3 is positive
  (+2.4 n=8, +2.6 n=49): 0.002 a round, not pursued.
- hm5's legal non-calls (eg) against calling, by assaf (< 0.06, 0.06-0.1, 0.1-0.2, higher) and opponent cards:
  every cell negative for both opponents (best: -0.10 +- 1.48 vs trained, opponent 2 cards, assaf < 0.06).
  hm5 should not call more.

### 4. Hint "leave an open 4-5 late" (probe `alt_leave`, seed 7103, 24,000 rounds per opponent)

Where hm5 (eg) takes an open card worth 3-6: the same throw with a deck draw instead.

| cell | vs counter: n, each | vs trained: n, each |
|---|---|---|
| open card pairs a kept card (any value, any ply) | 4,980; -2.5 to -5.0 each in every cell | 4,989; -1.9 to -4.7 (one cell +0.5, n=38) |
| unpaired 3, ply < 9 / 9-15 / 16+ | 550 -2.21 / 1090 -1.65 / 915 -0.09 | 772 -1.60 / 1245 -0.01 / 2251 +0.47 +- 0.32 |
| unpaired 4, ply < 9 / 9-15 / 16+ | 226 -2.05 / 810 -0.57 / 601 +0.93 +- 0.64 | 166 -2.30 / 669 -0.24 / 607 +1.04 +- 0.61 |
| unpaired 5, ply 9-15 / 16+ | 134 +0.27 / 96 +0.88 +- 1.38 | 109 -3.44 / 186 +0.26 +- 0.88 |

Unpaired 4-5 at ply 16+: +0.027 +- 0.017 (counter), +0.028 +- 0.017 (trained) a round. About 1.5 errors each:
the hint replicates in sign but not clearly; not built (the brief asks for groups clearly positive for both).

### 5. Whole-game test, seed 7200 (4,000 rounds + 800 matches, --workers 2), v1 = hm5 + rule S, v2 = rule S with sv >= 1

| net | vs counter: margin, matches | vs trained: margin, matches | vs hm5: margin, matches |
|---|---|---|---|
| hm5 | +1.61, 587-213 (73.4%) | +0.31, 486-314 (60.8%) | 0.00, 400-400 |
| v1 (12 units) | +1.66, 589-211 (73.6%) | +0.35, 491-309 (61.4%) | +0.04, 403-397 (50.4%) |
| v2 (12 units) | +1.65, 590-210 (73.8%) | +0.31, 498-302 (62.3%) | +0.07, 398-402 (49.8%) |

Both are +0.00 to +0.07 a round over hm5 on the same deals, as the play-outs predict (+0.03 to +0.07); far inside
the test's noise (+-0.2, mirrored pairs tighter). Matches: no visible effect either way.

### 6. Variant play-out check, seed 7104 (40,000 rounds per opponent; hero = the variant, alternative = hm5's move where it differs)

gain < 0 means the variant's own move was better than hm5's.

| variant, changed decisions | vs counter: per round, each, a round | vs trained: per round, each, a round |
|---|---|---|
| v1: every change is rule S's swap (sv >= 2) | 0.0066, -5.30 +- 1.21, -0.035 +- 0.008 | 0.0062, -3.49 +- 1.35, -0.022 +- 0.008 |
| v2: the sv >= 2 swaps | 0.0067, -5.38 +- 1.19 | 0.0063, -3.60 +- 1.35 |
| v2: the sv = 1 swaps | 0.0130, -3.22 +- 0.95 | 0.0168, **+0.57 +- 0.83** |

v1's changes pay against both opponents on a third seed: **+5.3 +- 1.2 (counter) and +3.5 +- 1.4 (trained) each,
0.006-0.007 times a round, +0.02 to +0.035 a round.** v2's extra sv = 1 swaps pay against the counter only
(+3.2) and not against the trained net (-0.6 +- 0.8): dropped.

## Conclusion

- hm5's Yaniv decision is already right almost everywhere: calling instead of its non-calls, or playing on
  instead of its calls, loses in every group by my points, assaf, opp_min, opponent card count, known cards,
  ply and estimate, against both opponents. The hm4 endgame gap to the trained net (handover.md) is mostly gone
  in hm5: the trained net's moves after both hands <= 4 are worth +0.09 +- 0.08 (counter) and -0.07 +- 0.08
  (trained) a round to hm5.
- The one class that pays: **rule S** (swap instead of a catchable call when the open card is 2+ points under my
  top card, opponent 3-4 cards, assaf >= 0.005). Three seeds, both opponents, +3.5 to +8.9 each, 0.006-0.0075 a
  round. Kept as `add_endgame_units(kit)` (defaults = rule S), 11 authored units (12 hidden units in the built
  net, one a carry). Built variant: `/tmp/claude-0/-home-user-yaniv-master/3c7d4d81-ef83-553b-adc8-6a5382f1abf0/scratchpad/endgame/v1.json`
  (570 hidden units). Whole game it is worth about +0.03 a round, below what a 4,000-round test can show.
- Not checked: matches near 50/100 (rule S also overrides the score units' "assaf on cut" call reason, which is
  rare); a swap that throws a set rather than a single.
