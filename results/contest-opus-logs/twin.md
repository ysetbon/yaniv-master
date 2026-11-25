# twin: keep the high single whose twin the opponent is likely to throw?

Designer `twin`, team opus. Rules: `training/yaniv/contest_opus_twin.py`. Scratch: play-out records and scripts in
the scratchpad `twin/` folder (census.py, run.py, ana.py). Hero = hm5 in every play-out; gain = B's margin − A's
(A = hm5's move), > 0 means the rule's move is better. "occ/rd" = occurrences per round (B ≠ A).

## Situation

Hero holds 5 cards, the opponent ≥ 4, and I hold at least two ranks with exactly one copy worth ≥ HI points
(HI = 8; HI = 7 reported too). In the first version I did NOT exclude positions where the open card pairs one of
my high singles; see variant 0 below - that broke rule (b). Final definition adds: the open card pairs none of my
cards (that is the "pickup" half, measured on its own).

## Variants measured (in order)

### 0. Rule (b) "throw the high single with the fewest unseen copies", first definition (bug) - LOSES
8,000 rounds seed 7101: vs counter −1.74 ± 0.31 each (0.44/rd), vs trained −1.98 ± 0.31 (0.43/rd).
Cause: when the open card is a copy of one of my singles (say open J, I hold J and T), the open J is not
"unseen", so J has the fewest unseen copies and rule (b) threw the J, while hm5 throws the T and takes the J
(528 of 1,271 "10 for 10" cases). That is the pickup position, not the twin question. Fixed by the definition above.

### What hm5 does in the situation (census, 2,000 rounds each, seed 4242, no alternative)
Situation (HI = 8, final definition) occurs 0.56 times a round (HI = 7: 0.74 / 0.73), vs counter and vs trained alike.
hm5's move there, vs counter (vs trained within 1 point):

| hm5 throws | share |
|---|---|
| a top-value high single, unseen counts of my high singles all equal | 65% |
| a set (low pair etc.) or Yaniv | 23% |
| the high single with the FEWEST unseen copies, when counts differ | 9% (8% top value, 1% lower) |
| a high single that is NOT the fewest-unseen | 1% |
| a low single | 1% |

So rule (b) is already hm5's play: the LIVE charge (1 a copy) makes it throw the rank with fewer unseen copies
and keep the more "live" rank, 90% of the time the counts differ.

Pickup half (open card pairs a single worth >= 8 that I hold): with 5 cards hm5 takes it 609 of 610 times
(vs counter) and 606/606 (vs trained); with 4 cards 98% / 97%. hm5 declines mostly with 2-3 cards.

Full plan followed (census2.py, 3,000 rounds each, seed 4243; hm5 throws a high single keeping >= 1 other high
single): 916 / 924 situations (0.31 a round), 1.5 kept high ranks each (98.5% with 3 unseen copies). The twin of a
kept rank turns up as the open card within the opponent's next 2 turns in 28% / 27% of situations; hm5 takes it
248 of 260 / 240 of 250 times and throws the pair the next turn 245 of 248 / 237 of 240. Every kept rank with 2
unseen copies (20 / 19) had its twin turn up: that copy was in the opponent's known hand (it took it), so it was
about to throw the pair.
=> hm5 already plays the whole plan: keep the live rank, take the twin, throw the pair.

### Play-outs, 8,000 rounds seed 7101 each, hero hm5 (records in scratch rec_*.json)

| rule | opp | occ/rd | gain each (+-se) | gain a round |
|---|---|---|---|---|
| (b) fewest-unseen, HI 8 | counter | 0.224 | -1.00 (0.39) | -0.224 |
| (b) fewest-unseen, HI 8 | trained | 0.230 | -0.69 (0.36) | -0.158 |
| (b) fewest-unseen, HI 7 | counter | 0.283 | -1.15 (0.35) | -0.325 |
| (b) fewest-unseen, HI 7 | trained | 0.288 | -0.98 (0.32) | -0.282 |
| (b') MOST-unseen (anti-twin), HI 8 | counter | 0.308 | -1.43 (0.32) | -0.440 |
| (b') MOST-unseen (anti-twin), HI 8 | trained | 0.312 | -0.47 (0.30) | -0.146 |
| (c) keep rank opp known to hold, HI 8 | counter | 0.002 | +6.62 (2.65) | +0.011 |
| (c) keep rank opp known to hold, HI 8 | trained | 0.002 | +7.58 (3.36) | +0.011 |
| (c) HI 7 | counter / trained | 0.002 | +5.73 (2.36) / +6.73 (2.76) | +0.011 / +0.013 |
| (d) always highest single, HI 8 | counter | 0.158 | -2.37 (0.47) | -0.373 |
| (d) always highest single, HI 8 | trained | 0.161 | -1.10 (0.43) | -0.177 |
| (d) HI 7 | counter / trained | 0.220 / 0.223 | -2.19 (0.39) / -1.32 (0.36) | -0.481 / -0.295 |
| pickup: take open twin (>= 8) where hm5 does not | counter | 0.025 | -2.04 (1.24) | -0.050 |
| pickup | trained | 0.031 | -4.32 (0.91) | -0.135 |
| pickup >= 7 | counter / trained | 0.028 / 0.036 | -2.24 (1.15) / -4.20 (0.85) | -0.062 / -0.153 |

Sub-cases (HI 8; "A>B" = points of hm5's single > points of the rule's; d = unseen copies of B's rank - of A's):
- Where hm5 throws a SET and the rule throws a high single instead: (b) -2.08 (0.50) counter / -1.13 (0.48)
  trained, 0.13/rd; (b') -2.97 (0.51) / -1.58 (0.48). hm5's set is right. This is most of every rule's loss.
- (b) single for single: almost all "same value, same unseen count" (d0, 0.07/rd) = an arbitrary tie-break:
  +1.18 (0.69) / +0.40 (0.64): noise, no information. Where (b) disagrees with hm5 on a real count difference
  (0.01/rd) it loses: same value d-1 -6.2 (5.0) / -6.9 (2.6).
- (b') anti-twin, same value, B throws the rank with ONE MORE unseen copy (d+1): -0.87 (1.25) counter,
  -0.78 (1.14) trained, 0.020/rd. So keeping the live rank is worth about +0.8 a time (+0.016 a round), and hm5
  already does it. Same value d0 (tie-break): -0.20 (0.52) / +0.38 (0.48).
- (b) by opponent card count: o5 -0.77 (0.44) / -0.70 (0.43); o4 -1.92 (0.91) / -0.63 (0.64).
- (c) is on-topic in only 4 of 13 cases (the rest are hm5 sets); in those 4 hm5 never threw the known rank (FEED
  already charges it) and the difference is a tie-break between two other 10s. +6.6 is noise on n = 13.
- Pickup: every disagreement is hm5 throwing the single whose twin is open (instead of taking it). By my hand size:
  n=3 -4.15 (1.75) / -5.78 (1.19); n=2 -3.0 / -3.9; n=4 +1.87 (2.98) / -0.30 (2.56); n=5: 8-9 cases. hm5 is right.

### (e) keep the HIGHEST high single, throw a lower one (the opponent throws its high cards first, so the twin of
my 10 should come sooner than the twin of my 8). Only where hm5 throws a top-value single and the lower one has
<= its unseen copies (so LIVE does not disagree). 8,000 rounds each:

| rule | opp | seed | occ/rd | gain each (+-se) | gain a round |
|---|---|---|---|---|---|
| keep-high HI 8 | counter | 7101 | 0.270 | +0.54 (0.35) | +0.145 |
| keep-high HI 8 | trained | 7101 | 0.273 | -0.14 (0.32) | -0.038 |
| keep-high HI 8 | counter | 7202 | 0.278 | -0.18 (0.35) | -0.050 |
| keep-high HI 8 | trained | 7202 | 0.282 | -0.38 (0.33) | -0.106 |
| keep-high HI 7 | counter | 7101 | 0.392 | +0.22 (0.30) | +0.086 |
| keep-high HI 7 | trained | 7101 | 0.393 | -0.23 (0.27) | -0.092 |

Pooled HI 8: counter +0.18 (0.25), trained -0.26 (0.23). Zero within noise; sub-cases (10 kept over 8 / over 9,
by ply, by opponent count) flip sign between seeds. Does not pay.

## Own idea (5-card hands): the draw after an early throw
- `alt_open_mid`: 5 v >= 4 cards, hm5 draws from the deck while the open card is worth 3-6 and pairs nothing kept;
  B takes it (same throw). counter 0.253/rd -1.83 (0.45); trained 0.271/rd -1.93 (0.43). hm5 is right.
- `alt_deck_low`: hm5 takes an unpaired open card worth 0-3; B draws from the deck. counter 0.073/rd -1.11 (0.86);
  trained 0.116/rd -0.36 (0.65). hm5 is right (or it does not matter).

## Conclusion
The owner's idea is real but hm5 already plays it, all three steps:
1. which single: the LIVE charge throws the rank with fewer unseen copies whenever the counts differ (90% of those
   positions); throwing the MORE live rank instead costs about 0.8 a time (-0.87 (1.25) / -0.78 (1.14), 0.02/rd).
2. pickup: hm5 takes the open twin 99-100% of the time with 5 cards (PAIR_OPEN); where it declines (mostly 2-3
   cards left) forcing the pickup loses 2-4 points a time.
3. the pair: having taken the twin, it throws the pair next turn 99% of the time.
No precisely-defined deviation pays against both opponents: (b) -1.0 / -0.7, (c) on-topic only ~0.0005/rd,
(d) -2.4 / -1.1, (e) +0.2 / -0.3 pooled, pickup -2.0 / -4.3. Most losses come from overriding hm5's choice to
throw a SET (low pair) instead of a high single in these positions (-1 to -3 a time): hm5 is right there too.
No net block was written (`add_twin_units` is not offered); no variant built or evaluated.
