# Thread `twin`: keep the high single whose twin the opponent will throw?

The owner's idea: early in a round players throw their large cards, so when I hold two or more HIGH SINGLES it
may pay to keep the one whose twin the opponent is likely to throw, take the twin from the open pile next turn
and throw the pair the turn after, instead of always throwing my highest single.

Files: `training/yaniv/contest_fable_twin.py` (probes, mechanism count, designs), raw tables
`results/contest-fable-logs/twin_*.txt`, records `twin_*.pkl`. Nothing is fitted or searched; every number below
is a count from played rounds (real game, 5 cards, Yaniv at 7).

## 0. Definitions (everything a 125-input ReLU net can compute)

- high single: a rank I hold exactly once, worth >= 5 points (classes 5-7, 8-9, 10 = T/J/Q/K). No runs exist in
  this game, so a single can only ever leave as part of a set of its own rank.
- unseen copies of rank X (`unseen_X` x 4) = 4 - mine - in the discard pile (`gone_X` + the open card) - known in
  the opponent's hand (`known_X`).
- the situation: the opponent holds 5 or 4 cards, my move throws ONE card of a rank I hold once (>= 5 points),
  and I hold at least one more such single of another rank. The choice: which one goes first.
- what the current net (hm5 = contest_fable.build()) does there: the throw part of its score is, per move,
  -(points kept) - LIVE x (unseen copies of the thrown rank) [LIVE = 1 against 4-5 cards] - FEED (1 + 0.5 x points
  if the opponent is KNOWN to hold the thrown rank) + small plan terms; so between two high singles it throws the
  one with the larger (points - unseen copies): it already keeps the more 'live' rank at equal points, and keeps
  a rank the opponent is known to hold unless the other single is ~6+ points heavier.

## 1. The mechanism: how often does the twin come? (`contest_fable_twin mech`)

Command: `python -m training.yaniv.contest_fable_twin mech hm5 OPP --rounds 40000 --seed 11 --workers 2`
(tables: twin_mech_hm5.txt, twin_mech_counter.txt, twin_mech_trained.txt). One record per hero turn and per high
single held AFTER the hero's move, opponent on 4-5 cards. opp1 = the opponent's very next throw is of that rank
(so its top card is my twin and I can take it); opp2 = next throw or the one after, while I still hold the single.

Result (hm5 as the hero; 40,000 rounds each; 6.0-6.5 held high singles a round counted once per hero turn):

P(the opponent's very next throw is my single's rank), opponent NOT known to hold it, unseen copies u:

| opp cards, my single | u=3 vs hm5 | u=3 vs counter | u=3 vs trained | u=2 (hm5 / counter / trained) | u=1 |
|---|---|---|---|---|---|
| 5 cards, ten-card (T/J/Q/K) | 11.7% | 12.7% | 12.0% | 3.7 / 3.4 / 3.4% | 2.0 / 2.1 / 2.2% |
| 5 cards, 8-9 | 9.0% | 10.3% | 9.2% | 3.5 / 3.8 / 2.9% | 1.0 / 1.6 / 1.3% |
| 5 cards, 5-7 | 5.8% | 6.7% | 6.1% | 4.0 / 2.9 / 2.9% | 1.3 / 1.1 / 1.2% |
| 4 cards, ten-card | 11.8% | 11.1% | 12.4% | 4.1 / 4.1 / 4.1% | 1.9 / 2.2 / 2.0% |
| 4 cards, 8-9 | 10.2% | 10.9% | 10.6% | 4.2 / 4.5 / 4.2% | 1.6 / 1.9 / 1.8% |
| 4 cards, 5-7 | 8.4% | 9.9% | 8.8% | 4.1 / 3.7 / 3.7% | 1.7 / 1.5 / 2.1% |

(standard errors 0.1-0.4 points of a percent.) Within TWO opponent turns (opp2), u=3: ten-card 15-16%, 8-9
15-17%, 5-7 11-13% against 5 cards. By exact points against 5 cards (all u, vs hm5): 5: 4.9%, 6: 6.4%, 7: 7.6%,
8: 8.6%, 9: 9.8%, ten-cards: 11.1%. By ply (ten-card, u=3, vs hm5): ply 0-1 13.6%, 2-3 11.5%, 4-5 7.9%, 6+ 5.5-7.5%.
The round ends by the opponent's call instead: 0.3-0.6% against 5 cards, 2.5-3.5% against 4.

- When the opponent is KNOWN to hold my single's rank (it took one from the open pile): it throws that rank on
  its next turn 100% (hm5), 100% for 8+ / 50% for 5-7 (counter), 99-100% (trained). Only 0.03-0.07 such holdings a
  round. The net already keeps such a single (FEED charges -(1 + 0.5 x points) for throwing it).
- When the twin does come (0.53-0.60 times a round over all my high singles): hm5 takes it 98% of the time against
  5 cards (91-95% against 4), and after taking it throws the pair on its very next turn 99.5% (ten-cards), 99%
  (8-9), 97-98% (5-7) against 5 cards; 96 / 95 / 90% against 4 cards. So 'do I really get to throw the pair': yes.
- What the mechanism can be worth: seen copies matter a lot (u=3 -> 12%, u=2 -> 3.5%: once the opponent has thrown
  one copy it does not hold another), but POINTS hardly matter among the high cards (ten-card 12% vs a 9 at 10%,
  an 8 at 9%, a 7 at 8%). Choosing between two u=3 high singles changes the twin's chance by 1-4 points of a
  percent; with a twin worth roughly one heavy card of the race (~5 points) that is 0.05-0.2 points, less than
  the 1-5 points of extra exposure from keeping the heavier card. The only lever is unseen copies, and the net's
  LIVE term already uses it (8 points of a percent per copy x ~5-8 points ~ 0.4-0.7 a copy, plus my own deck
  chance to pair: LIVE = 1 a copy is the right size).
- Checked a suspicious cell ('ply 2-3, u=2: 17.9%' in the by-ply table): that table included known cards; with
  known excluded and the seen copy split by who threw it (16,000 rounds, seed 21, hm5 vs hm5, ply 2-3, ten-cards):
  opponent threw the seen copy 1.5%, I threw it 0.4%, it was the first open card 5.8%, nothing seen 10.5%.

## 2. Which high single to throw: both moves played out (`contest_fable_twin which`)

Command: `python -m training.yaniv.contest_fable_twin which hm5 OPP --rounds N --seed 11 --workers 2`
(`contest_fable_lab.paired` with `contest_fable_twin:probe_which`; tables twin_which_OPP.txt).
At every hero decision in the situation of section 0, every 'throw the other high single, same draw' is played
out to the end on the same hidden cards (hero's own policy afterwards). gain = alternative - played, in points of
margin. Notation: the net PLAYED rank c; the alternative throws rank a and KEEPS c. dp = points(a) - points(c),
du = unseen(a) - unseen(c). 'clean' = the open card is neither rank.

Runs: vs hm5 30,000 rounds, vs trained 20,000, vs counter 16,000 (seed 11 each; a first batch was stopped by a
30-minute job limit and re-run with these sizes). About 2.9 alternatives a round. Combined table (made from the
saved records, also in twin_which_combined.txt); gain per occurrence +- standard error (occurrences a round):

| class (a priori) | vs hm5 | vs trained | vs counter |
|---|---|---|---|
| **opponent on 5 cards** | | | |
| all alternatives | -0.46 +-0.08 (1.58) | -0.65 +-0.09 (1.59) | -0.58 +-0.10 (1.86) |
| T1 THE IDEA, plain: keep the HIGHER single, throw the lower; equal unseen | -0.28 +-0.09 (1.10) | -0.37 +-0.12 (1.11) | -0.26 +-0.12 (1.27) |
| T1 at ply 0-1 | -0.02 +-0.13 (0.64) | +0.00 +-0.15 (0.65) | -0.02 +-0.17 (0.69) |
| T1 at ply 2-3 | -0.53 +-0.18 (0.30) | -0.72 +-0.22 (0.30) | -0.45 +-0.23 (0.36) |
| T1 at ply 4+ | -0.84 +-0.24 (0.17) | -1.21 +-0.29 (0.16) | -0.69 +-0.28 (0.22) |
| T1 keep a ten-card, throw an 8-9 | -0.35 +-0.16 (0.31) | -0.35 +-0.20 (0.32) | -0.54 +-0.22 (0.36) |
| T1 keep a ten-card, throw a 5-7 | -0.20 +-0.14 (0.51) | -0.46 +-0.18 (0.51) | -0.24 +-0.19 (0.58) |
| T1 keep an 8-9, throw a lower one | -0.48 +-0.22 (0.21) | -0.14 +-0.26 (0.21) | +0.08 +-0.28 (0.25) |
| T2 keep the higher AND more live single (net threw it anyway) | -4.2 +-1.7 (0.004) | -5.3 +-1.7 (0.004) | -6.9 +-1.4 (0.012) |
| T3 keep the higher but LESS live single | -1.87 +-0.25 (0.16) | -2.29 +-0.29 (0.16) | -1.94 +-0.29 (0.22) |
| E1 equal points, equal unseen (coin flip, control) | -0.14 +-0.18 (0.23) | -0.28 +-0.22 (0.23) | -0.35 +-0.23 (0.25) |
| E3 equal points: keep the LESS live one (undo LIVE) | -0.22 +-0.47 (0.03) | -1.03 +-0.56 (0.03) | -1.74 +-0.55 (0.03) |
| L1 the net kept the HIGHER single because it is more live; undo that | -0.99 +-0.40 (0.05) | -1.38 +-0.49 (0.05) | -0.62 +-0.45 (0.06) |
| K1 throw a rank the opponent is KNOWN to hold | -3.0 +-1.0 (0.013) | -6.5 +-1.1 (0.014) | -6.6 +-1.0 (0.023) |
| **opponent on 4 cards** | | | |
| all alternatives | -0.69 +-0.08 (1.29) | -0.71 +-0.10 (1.25) | -1.00 +-0.12 (1.10) |
| T1 keep the higher, equal unseen | -0.47 +-0.10 (0.81) | -0.40 +-0.12 (0.78) | -0.70 +-0.17 (0.66) |
| T1 at ply 0-1 / 2-3 / 4+ | -0.36 / -0.30 / -0.65 | +0.22 / -0.12 / -0.86 | -0.16 / -0.60 / -0.92 |
| T3 keep higher but less live | -1.57 +-0.16 (0.27) | -1.64 +-0.20 (0.26) | -2.07 +-0.24 (0.27) |
| E1 coin flip (control) | -0.02 +-0.25 (0.10) | -0.06 +-0.32 (0.10) | +0.50 +-0.41 (0.08) |
| E3 equal points: keep the less live | -1.04 +-0.37 (0.03) | -1.17 +-0.46 (0.03) | -1.09 +-0.58 (0.03) |
| L1 undo 'kept the higher because more live' | -0.44 +-0.34 (0.06) | -1.06 +-0.43 (0.06) | +0.23 +-0.52 (0.05) |
| K1 throw a rank the opponent is known to hold | -3.4 +-1.1 (0.009) | -6.6 +-1.2 (0.009) | -7.2 +-1.1 (0.016) |

Other a priori splits (per opponent, in twin_which_OPP.txt): by exact (dp, unseen kept, unseen thrown), by copies
gone, by the kept rank's points; none has a positive cell beyond 2 standard errors that repeats for a second
opponent (best single cell vs hm5: equal points, equal unseen, ply 4-5: +1.15 +-0.58, n 531, a coin-flip class).

What the net really does in this situation (vs hm5, share of the alternatives' records): equal unseen -> throws
the higher single 99.9%; equal points, different unseen -> throws the LESS live rank 92%; the higher single is
also the more live one -> keeps the higher 88% (3,070 records vs 438), about 0.10 decisions a round.

Reading:
1. The plain idea (T1: keep the higher single hoping for its twin, throw a lower high single first) LOSES against
   all three opponents: about -0.3 a time against 5 cards and -0.4 to -0.7 against 4, 0.8-1.3 times a round. The
   only place it is free is the very first turn against 5 cards (ply 0-1: -0.02 / 0.00 / -0.02, +-0.15): there
   the order of two high singles does not matter at all, in either direction; from ply 2 it costs 0.5-1.2.
2. The part of the idea that is true is the part the net already has: unseen copies. Undoing the net's LIVE
   choice loses in 11 of 12 cells (E3, L1: -0.2 to -1.7; pooled over the three opponents and both card counts
   about -0.9 +-0.15), and throwing a rank the opponent is known to hold (K1) loses 3-7 points.
3. Going further than LIVE = 1 (T2: keep the higher, more live single where the net threw it) loses 1-7 points
   in the few cases it changes (0.004-0.014 a round): the net throws such a card only when the points gap is
   large.

## 3. The open card IS my single's twin: take it or draw from the deck (`contest_fable_twin take`)

Command: `python -m training.yaniv.contest_fable_twin take hm5 OPP --rounds 20000 --seed 12 --workers 2`
(`probe_take`: the open card has the rank of a single I keep this turn; alternative = the same throw with the
other draw; tables twin_take_OPP.txt). hm5 takes the twin in every recorded case with 30+ samples (the 'played
deck' classes are empty), so the gain below is 'deck instead of take':

| twin's points | vs hm5 | vs trained | vs counter |
|---|---|---|---|
| ten-card, opp 5 cards | -4.81 +-0.33 (0.18 a round) | -4.39 +-0.32 (0.19) | -4.23 +-0.29 (0.22) |
| 8-9, opp 5 cards | -4.36 +-0.53 (0.08) | -3.98 +-0.50 (0.08) | -4.03 +-0.43 (0.10) |
| 5-7, opp 5 cards | -3.98 +-0.48 (0.09) | -4.37 +-0.46 (0.09) | -4.81 +-0.40 (0.12) |
| ten-card, opp 4 cards | -4.94 +-0.43 (0.10) | -4.82 +-0.45 (0.09) | -4.85 +-0.48 (0.09) |
| 8-9, opp 4 cards | -5.03 +-0.52 (0.07) | -4.69 +-0.51 (0.07) | -4.96 +-0.54 (0.07) |
| 5-7, opp 4 cards | -5.11 +-0.39 (0.12) | -4.95 +-0.37 (0.13) | -5.29 +-0.38 (0.13) |

So taking the twin is worth 4-5 points each time whatever its points (also for twins worth 1-4: -3.2 to -6.2),
which is the net's PAIR_OPEN = 5, and the net already always does it. By ply for twins >= 8, single thrown: ply < 4
-4.0 to -4.8, ply 4-7 -4.4 to -5.4, ply 8+ -2.4 to -3.5. And the pair does get thrown: section 1 (97-99.5% on the
very next turn against 5 cards, 90-96% against 4).

Value check of the whole idea from these two numbers: a twin taken is worth ~4.5 points; keeping single X instead
of single Y changes the chance the twin comes by at most ~8 points of a percent per unseen copy (and by ~1-4
points of a percent between point classes), i.e. ~0.35 points a copy and 0.05-0.2 points between point classes.
That is what the played-out numbers show: nothing to gain from points, and the copies are already charged.
