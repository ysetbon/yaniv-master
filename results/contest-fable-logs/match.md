# Thread `match`: how should the net play differently because of the match scores?

Files: `training/yaniv/contest_fable_match*.py`, `results/contest-fable-logs/match*`. Nothing here is fitted or
searched: tables are read, alternatives are named a priori and played out, units are written by hand.

## 1. What the match-win table says (hand reasoning)

`python -m training.yaniv.contest_fable_match_tab > results/contest-fable-logs/match_tab.txt` (reads wtable.npy and
the 60,000 round outcomes behind it; full tables in match_tab.txt).

Round facts (hm5 v hm5): a round winner adds 0, the loser adds 14.3 on average (10/25/50/75/90th percentile
6/9/13/17/25); 7.3% of round losses are +30 or more (caught calls).

(a) Value of ONE point for me, in points of match-win % (average over the next 5 points, no cut inside):

    my score \ opp score    0     20    40    60    80    90    95
       0-5                 0.59  0.57  0.48  0.32  0.16  0.09  0.06
      20-25                0.59  0.63  0.60  0.48  0.27  0.16  0.12
      40-45                0.62  0.74  0.80  0.75  0.51  0.32  0.25
      45-50 (cut ahead)   -2.45 -2.34 -1.87 -1.01 -0.17  0.09  0.17    <- points are worth MINUS here on average
      55-60                0.48  0.65  0.80  0.88  0.72  0.50  0.42
      65-70                0.50  0.71  0.94  1.15  1.10  0.87  0.78
      75-80                0.31  0.49  0.72  1.01  1.10  0.95  0.89
      85-90                0.34  0.55  0.87  1.40  1.86  1.89  1.88
      95-99                0.59  0.73  0.89  1.06  1.09  0.97  0.91

  A point is worth ~0.6 at 0-0, ~1.0-1.2 when both are at 65-80, ~1.9 when I am at 85-90 and the opponent is high,
  and next to nothing (0.05-0.15) when I am low and the opponent is at 90+. So points matter up to 3x more late in a
  close match and up to 10x less when the opponent is about to bust. The opponent's points mirror this.
  The table is smooth in bands: what is NOT smooth are the cliffs (a) at exactly 50/100, worth 25-50 points of %
  and (b) at 100+: P(win) goes to 0.

(b) A Yaniv call: gain of a successful call (opponent adds a typical losing hand) and cost of a caught one (my 5
points + 30), in points of match-win %, and the gain/cost ratio relative to 0-0 (0.41 = 8.4 / 20.5):

    my score \ opp     0          40         60         80         90         95
       0            8/20 1.0   7/20 0.9   5/15 0.8   3/ 9 0.8   2/ 5 0.8   1/ 4 0.8
      40            7/18 1.0  11/29 0.9  11/32 0.8   8/27 0.7   6/20 0.7   5/17 0.7
      60            5/11 1.2  11/25 1.1  14/35 1.0  14/40 0.8  12/37 0.8  11/35 0.7
      65 (+35=100) the caught call is a GAIN of 13-26 (it lands on 100 and is cut): always call (the existing unit)
      70            4/10 1.0  10/25 1.0  14/40 0.9  17/60 0.7  17/74 0.6  17/79 0.5
      80            3/ 7 1.0   8/17 1.2  14/30 1.1  20/50 1.0  23/65 0.9  24/71 0.8
      90            2/ 4 0.9   6/10 1.3  12/19 1.5  23/35 1.6  34/50 1.6  38/56 1.6
      95            1/ 3 0.9   5/ 8 1.4  11/15 1.7  24/29 2.0  38/44 2.1  43/50 2.1

  Read as a multiplier on the acceptable caught-chance IF not calling were worth the same share of the gain at every
  score. It is mild: 0.5-0.8 when I am at 70-80 and the opponent is high (a caught call ends the match, a lost round
  does not), 1.5-2.1 when I am at 90+ (any lost round ends the match anyway, so being caught costs no more than
  waiting and losing). At 0-0 the ratio means a call breaks even at 29% caught if NOT calling were worth nothing;
  hm5 calls at 1.5-4%: what not calling is really worth must be measured (section 2).

(c) Busting: with my score s, if the opponent wins the round I add my hand h: s + h > 100 loses the match,
  s + h == 100 is cut to 50 (the best thing that can happen), s + h < 100 plays on. So from s >= 80 or so the
  held points have a cliff at 100 - s, not a slope: the 20th point costs everything, the 5 points below it only
  ~1-2 each. And when even the lowest reachable hand is over the cliff, points are worth nothing: only winning
  the round (or landing exactly on 100) matters.

## 2. Paired measurement in match play, valued by match-win chance

Probe: `training/yaniv/contest_fable_match_probe.py:probe` (alternatives named in its docstring, all a priori).
Value of a record = 100 x [equity(wtable, s + a1, t + b1) - equity(wtable, s + a0, t + b0)] = points of match-win %
(`eq_gain`). "per match" = summed gain / matches: what the hero would gain per match if it always played the
alternative in that class. Report script: `contest_fable_match_rep.py` (full output in match_rep_*.txt).

Pilot: `python -m training.yaniv.contest_fable_match_probe hm5 hm5 400 1` (76 s; 16 rounds a match; too small).

### 2.1 hero hm5 v hm5, 6,000 matches, seed 11
`python -m training.yaniv.contest_fable_match_probe hm5 hm5 6000 11` -> match_rec_probe_hm5_hm5_6000_11.pkl
(97,420 rounds, 200,552 records), report in match_rep_hm5_6000_11.txt.

CALL v NOT (equity each +-se, occurrences a match):

    hm5 threw, alternative = call              all scores        caught is fatal (s+p+30>100)   out if I lose (s+p>100)
      assaf 0.02-0.05                          -0.52 +-0.31         -2.48 +-1.04                  +0.78 +-1.77 (0.016)
      assaf 0.05-0.10                          -1.80 +-0.22         -4.36 +-0.78                  +3.68 +-1.93 (0.026)
      assaf 0.10-0.20                          -4.46 +-0.23         -7.43 +-0.74                  +2.32 +-1.76 (0.027)
      assaf 0.20-0.35                         -11.38 +-0.26        -13.2  +-0.88                  -2.89 +-1.59 (0.019)
    hm5 called, alternative = best throw
      assaf < 0.02                             -3.07 +-0.09         -2.97 +-0.27                  -6.04 +-0.91
      assaf 0.02-0.05                          -1.77 +-0.22         -0.05 +-0.77 (0.16 a match)   -6.3  +-2.2

  By my score band (all assaf): calling when hm5 does not is -5.9 (s<40), -7.3 (40-63), -9.1 (63-70), -7.5 (70-80),
  -10.0 (80-86), -12.1 (86-93), -3.7 (93+); not calling when hm5 does is -2.7, -3.4, -3.8, -3.8, -2.4, +0.4 +-0.7
  (86-93), -3.9 (93+). hm5's threshold is right in every band, to within what 6,000 matches can show.
  Cells (my score band x opponent's score <50 / 50-80 / 80+ x assaf band) more than 1.5 se above 0, of ~90:
    call although hm5 does not, s 93+, opp 80+, assaf 0.05-0.1:  +23.9 +-6.8, n 48 (0.008 a match -> +0.19 a match)
    do NOT call although hm5 does, s 86-93, opp < 50, assaf <0.02: +2.1 +-0.54, n 640 (0.107 a match -> +0.22 a match)
    (and three cells of n < 100 at +2.6..+4.4, 1.6-2.6 se).
  => The Yaniv threshold by score band is NOT the lever: everything that could be gained is ~+0.2 a match from
     "call up to assaf ~0.2 when a lost round ends the match anyway" and ~+0.2 from "far behind at 86-93: do not call,
     keep playing for the 100 cut". The "caught call is fatal" veto gains nothing (-0.05 +-0.77 at assaf 0.02-0.05).

SAFER THROWS: `nobust` (get under 100 when the chosen move is over): -0.41 +-0.54 each, 0.38 a match: nothing
  (by opponent estimate / cards / score: no class beyond 1.3 se). `minhold` (fewest points held, s >= 60): negative in
  every score band (-1.8 at 63-70 ... -6.1 at 86-93; -2.2 at 93+), worse with fewer opponent cards. hm5's throws are
  already right about points at high scores; holding fewer points for its own sake costs equity.

CUT-SEEKING (the forced alternative lands / can land exactly on the cut; hm5 has units for these but they are weak
  outside est <= 9..15):

    by opponent's estimate        <9             9-12           12-15          15-20          20+
    cutopen100  (1.51 a match)  +22.6 +-4.8    +5.1 +-2.4     +3.25 +-0.92   +1.49 +-0.49   -1.79 +-0.33
    cutdeck10_100 (1.82)         +5.7 +-1.1    +3.9 +-0.95    +3.53 +-0.65   +1.29 +-0.44   -0.78 +-0.28
    cutopen50   (2.58)           +8.9 +-2.8    +0.4 +-1.1     +1.32 +-0.44   -0.56 +-0.24   -1.77 +-0.16
    cutdeck10_50 (3.11)          +3.9 +-0.5    +1.5 +-0.48    +0.40 +-0.33   -0.25 +-0.22   -1.46 +-0.15
    cutkeep50 / cutkeep100: negative in every band (-1 .. -9): staying near the cut for its own sake loses.

  Per match, the positive bands add up to: cutdeck10_100 +2.4, cutopen100 +1.5, cutdeck10_50 +1.1, cutopen50 +0.4
  (they overlap: the same position can offer two of them). By the opponent's card count the same picture: with 3 or
  fewer cards +4.6 (open 100), +5.3 (deck 100), +1.7 (deck 50); with 5 cards all negative.
  => THIS is the lever: the cut-landing units are too weak, most of all for the 100 cut (est 12-20) and for the
     deck draw that needs a ten.

### 2.2 hero hm5 v trained, 5,000 matches, seed 12
`python -m training.yaniv.contest_fable_match_probe hm5 trained 5000 12` (70,808 rounds), match_rep_trained_5000_12.txt.

  Calls: the same picture. Calling when hm5 does not: -1.4 +-0.34 (assaf 0.02-0.05), -2.9 (0.05-0.1), -5.1, -8.9 ...;
  not calling when it does: -4.1 +-0.09 (assaf <0.02), -0.55 +-0.24 (0.02-0.05). "Out if I lose" class, call
  although hm5 does not: +4.6 +-3.1 / +3.6 +-2.6 / +2.4 +-2.5 at assaf 0.05-0.1 / 0.1-0.2 / 0.2-0.35 (0.054 a match in
  all: +0.19 a match). Caught-is-fatal veto at assaf 0.02-0.05: +0.99 +-0.99 (0.14 a match). The only repeated cell:
  do NOT call at s 86-93, opp < 50, assaf < 0.02: +2.7 +-0.91 (0.052 a match -> +0.14 a match).
  nobust -0.05 +-0.68; minhold negative everywhere.

    by opponent's estimate        <9             9-12           12-15          15-20          20+
    cutopen100  (1.19 a match)   +6.8 +-6.9    -0.1 +-3.4     +1.05 +-1.17   -0.45 +-0.64   -1.49 +-0.41
    cutdeck10_100 (1.42)         +6.3 +-1.3    +5.0 +-1.2     +2.58 +-0.90   +0.19 +-0.58   -1.15 +-0.37
    cutopen50   (2.23)           -0.3 +-2.7    +0.4 +-1.2     +0.30 +-0.45   -0.84 +-0.25   -1.31 +-0.17
    cutdeck10_50 (2.75)          +3.0 +-0.5    +1.15 +-0.49   +0.52 +-0.34   -0.62 +-0.23   -1.56 +-0.16

  The deck-draw-for-a-ten gains hold against the trained net (positive bands: +1.5 a match at 100, +0.9 at 50); the
  open-take gains at est 15-20 do not.

### 2.3 A finding outside the a priori list: the opponent's cut ("denial")
From the same records (call:skip = hm5 called, alternative = its best throw), grouped in hindsight by whether the
opponent's score + its true hand was exactly 50 / 100 (my call then HANDS it the cut):

    v hm5:     on the cut 0.74 a match (9% of hm5's calls!): not calling +18.95 +-0.29 each; else -5.12 +-0.07
    v trained: on the cut 0.13 a match:                      not calling +29.6 +-0.74 each;  else -4.11 +-0.08

hm5 sits on its cut on purpose (its own cut units), the trained net does not. With perfect knowledge that is worth
+14 points of match-win % a match against an hm5-like rival. A priori detector from the inputs, |opp score + opp_est
- cut| (assaf < 0.02):   v hm5: <1: +2.60 +-0.59 (0.31 a match, 46% really on the cut), 1-2: +0.63 +-0.65 (39%),
2-3: -0.40, 3-5: -1.5, 5-8: -2.5, 8+: -4.0.   v trained: <1: -3.9 +-0.6, 1-2: -5.0: it only costs there.
=> worth a dedicated probe with what is KNOWN of the opponent's hand (probe_deny, section 2.5).
