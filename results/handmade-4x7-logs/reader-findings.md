# reader: what the trained net (yaniv-net-4cards-yaniv7) does differently from the card counter

Method: 300 mirrored match pairs (600 matches, 7,528 rounds, 85,837 decisions) of trained net vs counter.
At every decision of either player I recorded what the net would play and what the counter would play
(features.action_key), plus the true opponent hand. Scripts: SCRATCH/reader/probe.py, a1..a11.py.
"Call rate" tables mix both players' positions (it is "what would each do here").

## 0. Where the +1.3..1.5 a round comes from
| (net vs counter, 7,528 rounds)         | rounds | share | avg pts | per round |
|---|---|---|---|---|
| net calls, wins                         | 3745 | 49.7% | counter pays 12.1 | +6.02 |
| net calls, caught (Assaf)               | 183  | 2.4%  | net pays 35.9     | -0.87 |
| counter calls, wins                     | 3238 | 43.0% | net pays 12.9     | -5.56 |
| counter calls, caught                   | 362  | 4.8%  | counter pays 35.4 | +1.70 |
Net margin +1.29/round in this sample. Counter-vs-counter baseline: 6.4% of calls caught, 12.5 pts paid by the loser.
- Net's caught rate is 4.7% of its calls; the counter's is 10.1% against the net (vs 6.4% against itself).
- Rough split vs the counter-vs-counter baseline (±4.69/round each side): the net's own calls are worth +0.46/round more
  (fewer Assafs), the counter's calls against the net are worth +0.83/round less to the counter (it gets caught 60% more
  often because the net sits on low hands without calling = an "Assaf trap"). So ~2/3 of the gain is Yaniv timing.

## 1. Yaniv: the net is MUCH more conservative than the counter, and its threshold is ~flat in my points
Call rate when Yaniv is legal, by my points:
| pts | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| net | 1.00 | .95 | .75 | .63 | .52 | .51 | .46 | .42 |
| counter | 1.00 | 1.00 | .97 | .87 | .78 | .69 | .62 | .48 |
The net almost never calls where the counter wouldn't (2.7% of legal spots); the counter calls where the net doesn't in 21%.

Net call rate by `assaf` (uniform catch chance) x my points:
| assaf | pts 2 | 3 | 4 | 5 | 6 | 7 | counter (pts 2..7) |
|---|---|---|---|---|---|---|---|
| 0 (sure) | .99 | .99 | .99 | .98 | .96 | .96 | 1 |
| 0-0.02 | .84 | .88 | .78 | .86 | .90 | .94 | ~1 |
| 0.02-0.05 | .34 | .35 | .27 | .27 | .41 | .53 | 1,1,.99,.92,.92,.65 |
| 0.05-0.10 | .32 | .11 | .10 | .11 | .11 | .10 | 1,.99,.87,.51,.45,.28 |
| 0.10-0.15 | 0 | 0 | .03 | .02 | .07 | .01 | 1,1,.37,.15,.14,.04 |
| >0.15 | ~0 | | | | | | counter still calls at 2-3 pts up to assaf 0.2-0.3 |
So: net threshold is about assaf < 0.02-0.03 for ALL point totals (slightly MORE willing at 6-7 pts than at 3-5);
the counter's threshold rises steeply at low points (0.40 at 1 pt, 0.28 at 2, 0.18 at 3, 0.12 at 4).
TRUE caught rate (opponent really at <= my points) is ~3x the `assaf` number: assaf 0-0.02 -> 6%, 0.02-0.05 -> 15%,
0.05-0.1 -> 22%, 0.1-0.15 -> 31%, 0.15-0.2 -> 42%. (Opponents keep low cards.) Caught rate of the calls each makes:
net 1.2-2.7% at 1-4 pts, 5% at 5, 9-10% at 6-7; counter 4-7% at 2-4 pts, 7.5% at 5, 11% at 6-7.

The net's threshold tightens with ply (later = opponent's hidden cards are lower):
| assaf | ply<=5 | 6-9 | 10-14 | 15-20 | 21+ |
|---|---|---|---|---|---|
| 0-0.01 | .98 | .97 | .92 | .81 | .67 |
| 0.01-0.02 | .89 | .77 | .66 | .54 | .37 |
| 0.02-0.03 | .89 | .65 | .47 | .34 | .14 |
| 0.03-0.05 | .62 | .52 | .30 | .12 | .16 |
| 0.05-0.075 | .44 | .22 | .15 | .10 | .08 |
Counter's actual caught rate by ply: 4% (ply<=5), 6%, 9.5%, 12%, 14% (ply 21+); net's stays 3-7%.
Also more willing when (opp_est - my pts) is big: at assaf 0.01-0.02, gap 2-4 -> .46, 4-6 -> .63, 6-8 -> .78, 8+ -> .92.
Slightly less willing when my match score is high (assaf .01-.02: .76 at score<=30, .57 at 80+). Opp card count matters
little once assaf is known.
Value of declining: in 1,246 net positions where the counter would call but the net didn't, calling at once would have
averaged +3.1 (caught 16%), while the net's actual round result averaged +6.2. Calls the counter makes that the net
wouldn't: caught 19%, worth only +2.0 each; calls both would make: caught 6%, worth +10.7.

## 2. Taking the open card: the net takes an UNPAIRED open card only if it is <= 3 (counter: <= 5)
Take-open rate by open card points x copies of its rank already in my hand:
| open pts | 0-2 | 3 | 4 | 5 | 6 | 7-10 |
|---|---|---|---|---|---|---|
| net, unpaired | .97-1 | .90 | .35 | .07 | .02 | 0 |
| counter, unpaired | 1 | 1 | .95 | .66 | .09 | 0 |
| net, pairs a held card | 1 | .96 | .95 | .91 | .91 | .85-.90 |
| counter, pairs a held card | 1 | 1 | 1 | 1 | .89 | .76-.79 |
| net, makes a triple | | .88 | .84 | .45 | .73 | .5-.9 |
| counter, makes a triple | | 1 | 1 | .77 | .70 | .30-.46 |
Unpaired 4: net takes more when the deck is worse (unseen_avg > 5.5: ~.35-.45, < 5.5: ~.1), when I'm low (<=15 pts .47,
20+ .13), mid-round (ply 4-12 ~.45; ply<=3 .08, ply 21+ .07). Unpaired 3 drops to .36 if unseen_avg < 5.
Net takes the open card 21.7% of throws, counter 29.2%. Half (53%) of all throw disagreements are "same throw, other draw".
Net takes PAIRING open cards even when high: it will throw a LOWER single to take an open K that pairs its K
(A45K open K: net throws 5 takes K; counter throws K from deck). ~900 such spots.

## 3. Pairs: the net throws any pair of 3s or higher instead of a higher single (shrinks the hand)
Rate of throwing the pair, hand with exactly one pair + singles (rows pair rank pts, cols highest single):
| pair \ single | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|
| net, AA | .02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| net, 22 | .96 | .72 | .29 | .34 | .32 | .28 | .35 | .41 |
| net, 33 | - | .98 | .97 | .94 | .91 | .86 | .86 | .82 |
| net, 44+ | .99 | .99 | .97 | .93-.96 | .92-.96 | .9-.95 | .94-.96 | .89-.97 |
| counter, 22 | 1 | .05 | .05 | .06 | 0 | .02 | .01 | 0 |
| counter, 33 | - | .98 | .94 | .07 | .07 | .05 | .03 | .05 |
| counter, 44 | .98 | - | .98 | .98 | .88 | .06 | .06 | .07 |
| counter, 55 | | | | .97 | .97 | .94 | .89 | .11 |
(counter throws the pair only if 2x pair >= single; jokers/aces: both keep.) Example: 3358 -> net throws 33 (keeps 58),
counter throws 8 (keeps 335). 2,600 throw disagreements are this; it's the main "which card" difference.
The net never breaks a pair (0.07%). Throwing sets: net 24% of throws, counter 21%. keeps_pair after throw: net 7.6%,
counter 10.8%. callable_next chosen: net 5.3%, counter 7.7% (net does NOT chase callable_next).
low_top: net leaves a <=4 card face up more (15% vs 12%), a side effect of throwing low pairs - the net does not
avoid feeding low cards much. Feeding known opponent ranks: no real difference (20% vs 21%), except 10-pointers
(net throws a 10 the opponent is known to hold 68% vs 94%; n=142).
Equal-value singles (J vs K etc.): net prefers throwing T/K over J/Q - arbitrary, unimportant.

## 4. Disagreement rate
Both throw: 16% disagree. More late (ply 21+: 24%) and with 1-2 opp cards (21%), and when my hand is <= 7 (29%).

## 5. Built from these rules (training/yaniv/handmade_designs/reader.py), official test: margin +1.33, rounds 1060/2000, calls 991 caught 40 (4.0%), matches 468-332
What mattered (official margin): v1 base model +1.02 -> pair/triple handling (2s count half a heavy card, pairing-open bonus 9, triple +5) +1.21
-> callable_next bonus only vs a 4-card opponent +1.29 -> flatter Yaniv threshold (0.015 floor late) +1.32. Calling MORE like the trained
net early (v2) cost 0.2; calling less (T_EARLY .03) cost 0.15. Head-to-head vs the trained net: -0.10 a round.
Note: the official rounds test always starts at 0-0, so score features (lands_on_cut, my/opp score) cannot show up there.
