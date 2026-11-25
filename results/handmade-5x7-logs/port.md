# port: the 4-card net with its constants re-measured for the real game

Design file: `training/yaniv/handmade5_designs/port.py`. Net: `SCRATCH/port.json`
(SCRATCH = `/tmp/claude-0/-home-user-yaniv-master/a532e2e0-83d9-59dd-8a1a-414723f6a44f/scratchpad/port`),
shape [125, 214, 92, 48, 35, 33, 53, 29, 7, 1], 511 hidden units.

**Angle.** Keep `handmade.py`'s structure. Write every constant that was read off 4-card play as a table by the
opponent's card count, re-measure it in the real game, and find out which ones matter.

**Short answer.** The 4-card constants hold in the real game, and the net is insensitive to them in the 5-card
phase: re-measured tables change between 0.04% and 1% of decisions and none gives a measurable gain. The two
changes that do something are not re-measurements of an old constant: against a 5-card opponent a 2 counts
as a full heavy card, and a single whose rank can still be paired costs a little to throw. Together they are
worth about +0.2 points a round by paired playouts, which is at the edge of what the working test can see.
"The 4-card net with re-measured constants" is therefore about as good as the 4-card net, no better.

## Final numbers (working test: 6,000 rounds seed 61000, 1,500 matches seed 61001; against the card counter)

| player | margin | rounds won | caught | matches |
|---|---|---|---|---|
| the bar: 4-card net, whole game | +1.49 | 55.2% | 4.5% | 1029-471 (68.6%) |
| `j`: TWO_EXTRA only, switch | +1.52 | 55.6% | 4.5% | 1039-461 (69.3%) |
| `j`: TWO_EXTRA only, alone | +1.54 | 55.6% | 4.5% | 1033-467 (68.9%) |
| **port (final), switch** | **+1.65** | **55.9%** | 4.3% | **1062-438 (70.8%, 68.4%-73.0%)** |
| **port (final), alone** | +1.66 | 55.9% | 4.3% | 1057-443 (70.5%, 68.1%-72.7%) |
| port against the 4-card net, head to head (6,000 rounds, no matches) | +0.07 | 50.4% | 7.0% | - |

Noise on this test is about +-0.2 margin and +-2.5 points of matches. The final net is +0.16 margin, +0.7 points
of rounds and +2.2 points of matches above the bar: the right sign and the size the paired playouts predict
(+0.25 a round), but still inside the noise of this test. Head to head against the 4-card net it is level
(+0.07): both changes lean on how the counter plays (it cannot call with 5 cards; it throws high cards first).
The switch player and the net alone are nearly the same player by construction: port differs from the 4-card
net only when the opponent holds 5 cards, and that is always before the handover.

## How variants were measured

The working test cannot see a change of 0.1 a round, and most variants here change under 1% of decisions. So
besides the working test I used a paired measurement (`SCRATCH/diag2.py`): the base net plays the counter
(with the handover); at each of its decisions in the 5-card phase the variant is asked for its move, and where
the two differ both moves are played out to the end of the round with the same hidden cards and the base
policy afterwards. The sum of (variant's result - base's result) over those decisions, per round, is the
first-order gain of the variant; its error is about 15 x sqrt(differing decisions) / rounds. It is a
measurement of a hand-set variant, one number per variant; nothing loops over coefficients and it writes no
weights. It also says where (by card counts) a variant differs. Caveat: it is first-order (the variant does
not play the rest of the round), and it ignores match-score play (rounds start at 0-0).

## Measurements (real game, the 4-card net against the counter; `SCRATCH/measure.py`, 3,000 rounds)

They agree with `measure.md`, which arrived later.

1. **P(the counter calls on its next turn)**, by its card count and the bucket of `opp_est` (n in brackets):

| its cards | <5 | 5-8 | 8-11 | 11-14 | 14-17 | 17-21 | 21+ | all |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.63 (155) | 0.62 (8) | | | | | | 0.63 |
| 2 | 0.43 (83) | 0.33 (1098) | 0.27 (183) | | | | | 0.33 |
| 3 | | 0.24 (155) | 0.17 (1978) | 0.11 (1344) | 0.05 (317) | 0.01 (82) | | 0.14 |
| 4 | | | 0.13 (135) | 0.09 (1480) | 0.04 (1991) | 0.02 (1991) | 0.00 (848) | 0.044 |
| 5 | | | | 0.07 (72) | 0.03 (490) | 0.01 (1139) | 0.005 (5424) | 0.007 |
| handmade.py | 0.55 | 0.30 | 0.18 | 0.08 | 0.04 | 0.01 | 0.01 | |

   The estimate already carries the card count: the rows agree with the 4-card row within about 0.05 wherever
   a bucket is common. Nothing to re-measure.
2. **The counter's total when it calls**, by its card count: mean 2.7 / 4.5 / 5.5 / 6.05 / 6.3 for 1..5 cards
   (4 cards: 7 in 46% of calls, 6 in 30%). The 4-card net's catch threshold is one number, 5.
3. **Real catch rate of my calls**, by the opponent's card count: against 5 cards 3 of 250 calls caught (1.2%;
   by my total never above 4%); 4 cards 4.4%; 3 cards 7.6%. `assaf` under 0.01 means 2% real against 5 cards,
   5% against 4, 3% against 3; 0.01-0.02 means 13% against 3 and 4 alike. The net already calls every time it
   may against a 5-card opponent, and in the 5-card phase that is nearly all of its calls. Nothing to change.
4. **Turns left**: my turns still to come after a decision: 5.1 at (me 5, opp 5), 3.9 at (5, 4), 3.6 at (4, 5),
   2.9 at (4, 4). Share of my decisions made while either hand holds 5 cards: 55%; (5, 5) alone 23%.
5. **Twins for a kept single** (`SCRATCH/live.py`, 4,000 rounds): P(the open card at my next turn has the rank
   of a single I kept), by its points and the opponent's cards:

| opp cards | 0-2 pts | 3-5 | 6-9 | 10 |
|---|---|---|---|---|
| 5 | 0.003 | 0.037 | 0.074 | 0.127 |
| 4 | 0.004 | 0.057 | 0.094 | 0.096 |
| 3 | 0.003 | 0.051 | 0.079 | 0.084 |

   (a deck draw pairs it about 6% of the time at any points.) The counter throws its high cards first, so early
   in a round my high singles get twins from its throws; my 2s and aces never do.

## Where the 4-card net loses in the 5-card phase (diagnostics)

- `SCRATCH/diag.py` (3,000 rounds): at every 5-card-phase decision, simple alternatives to the net's move were
  played out (the other draw; the biggest set instead of a single; the highest other single instead of a set;
  the next lower / the highest single; call / do not call). **Every one loses on average**, by 1 to 10 points
  a decision, at every pair of card counts. Calling against 5 cards is right (not calling: -6). The only
  alternatives within noise of zero: taking an open 2 or 3 at (5, 5) (deck instead: +1.5 +- 2.0, +2.9 +- 2.4),
  and throwing a pair of aces/2s instead of a single at (5, 5) (-0.9 +- 1.0).
- `diag2.py` with the **counter** as the variant: where the counter would play differently (2,308 decisions in
  4,000 rounds) its move is worse by 0.67 +- 0.21 a round in total. Nothing to take from it.
- `diag2.py` with the **trained net** as the variant: 1,867 differing decisions, its moves better by
  +0.46 +- 0.19 a round, +0.38 of it at (5, 5). `SCRATCH/diag3.py` broke down the "it throws a set, I throw a
  single" cases: almost all are a **pair of 2s** (against a 10: 154 cases, +3.6 +- 1.6 each; against 7-9: 97,
  +1.4; against <= 6: 77, +5.2), positive only when the opponent holds 5 cards (0 to +0.5 against fewer).
  After TWO_EXTRA went in, the trained net's remaining differences are worth +0.03 +- 0.15 a round: nothing
  left to read off it in this phase. (This is a diagnostic in reader's sense: no weight copied or fitted.)

## Every variant measured

"Paired" = `diag2.py`: gain a round +- error, (differing decisions / decisions in the 5-card phase, rounds).
Base `v0` = port.py with every table at the 4-card value (plays as the 4-card net: 2 of 2,053 decisions differ).
Base `j` = v0 + TWO_EXTRA[5] = 0.5.

| # | variant (one change from its base) | reason / expectation | result | kept? |
|---|---|---|---|---|
| 0 | 4-card net, whole game (the bar) | | work: +1.49, 55.2%, 68.6% | bar |
| a | `pi`: P(call next) by card count, measured rows | the 4-card row had no card count; expected little, since opp_est carries it | paired vs v0: +0.004 +- 0.005 (6 / 14,188, 4,000 r). Whole game vs j: -0.03 +- 0.02 (30 / 34,287, 5,000 r) | no: no effect |
| b | `catch`: threshold drawn from the measured caller's-total distribution by card count | a 3-4 card caller is at 6-7, so holding 6-7 catches most calls; expected a gain after the handover only | paired vs v0, 5-card phase: +0.03 +- 0.02 (9 decisions). Whole game vs j: +0.07 +- 0.07 (362 / 34,287) | no: not measurable |
| c | CARD_BY[5] = 7 | more turns left, shrinking worth more? | paired vs v0: +0.03 +- 0.04 (80); vs j: -0.02 +- 0.05 (205 / 17,734), +0.07 +- 0.04 (488 / 42,840) | no: 0 to +0.07, the same decisions LIVE moves |
| d | CARD_BY[5] = 3 | the opposite | paired vs v0: -0.06 +- 0.05 (164) | no |
| e | CALLABLE also against 5 cards (3) | a call against 5 cards is caught 1-2% | paired vs v0: +0.05 +- 0.03 (50) | no: inside noise |
| f | OPEN_TELL_BY[5] = 0.75 | points held matter less against 5 cards | paired vs v0: 0.00 +- 0.04 (63) | no |
| g | LOW_TOP_BY[5] = 0 | a 5-card opponent gains less from a low open card? | paired vs v0: +0.04 +- 0.04 (44) | no: inside noise |
| h | CARD_BY[4] = 7 | | paired vs v0: 0.00 +- 0.03 (41) | no |
| i | CARD_BY[4] = 3 | | paired vs v0: +0.01 +- 0.03 (39) | no |
| j | TWO_EXTRA[5] = 0.5 (a 2 is a full heavy card against 5 cards) | the trained-net diagnostic above; reason: a 5-card opponent cannot call soon, so the card count matters, not the 4 points | paired vs v0: **+0.15 +- 0.06** (295 / 17,679, 5,000 r), +0.12 of it at (5, 5). Work: switch +1.52, 55.6%, 69.3%; alone +1.54, 55.6%, 68.9% | **yes** |
| k | TWO_EXTRA[5] = 1.0 | is a 2 worth even more to throw? | paired vs v0: +0.14 +- 0.09 (596) | no: no better than j |
| l | TWO_EXTRA[4] = TWO_EXTRA[5] = 0.5 | the same against 4 cards? | at (5, 4): -0.06 +- 0.05 (236) | no |
| m | j + ACE_EXTRA[5] = 0.5 | the same for aces? | paired vs j: -0.01 (18 decisions) | no |
| o | j + LIVE 0.5 a copy, every card count | tie-break between equal singles; keep the rank with more unseen copies | paired vs j: +0.16 +- 0.06 (409, 5,000 r), +0.07 +- 0.04 (834, 12,000 r); nearly all at (5, 5) and nearly all "set before single", not tie-breaks | superseded by u |
| p | j + LIVE 1.0 a copy | | paired vs j: +0.16 +- 0.08, +0.07 +- 0.05 | no: no better than o |
| t | j + LIVE[5] = 0.5 a copy (5-card opponent only) | the gain was only against 5 cards | paired vs j: +0.06 +- 0.03 (582 / 42,840), +0.08 +- 0.03 (581 / 41,989) | superseded by u |
| u | j + LIVE[5] = 0.5 x points / 10 a copy | measurement 5: twins come for high singles, not low ones | paired vs j: **+0.10 +- 0.04** (981 / 41,989, 12,000 r) | **yes = port** |
| e2 | port + CALLABLE against 5 cards (3), second sample | e's +0.05 was 1.3 errors: replicate | paired vs port: -0.02 +- 0.02 (173 / 42,230, 12,000 r) | no |
| g2 | port + LOW_TOP_BY[5] = 0, second sample | g's +0.04 was 1.1 errors: replicate | paired vs port: +0.03 +- 0.02 (150 / 42,230); both samples together about +0.035 +- 0.015 | no: 0.03 a round at 2 errors is not worth a constant |
| - | port (= u) | | work: switch +1.65, 55.9%, 70.8%; alone +1.66, 55.9%, 70.5%; head to head against the 4-card net +0.07, 50.4% of rounds | **final** |
| - | trained net's moves against port's (diagnostic) | is anything left to read off the trained net? | paired vs port: -0.14 +- 0.10 (5,105 / 42,230); at (5, 5) -0.16 +- 0.07 | port's 5-card phase is no worse than the trained net's |

Not measured on the working test: a to i, k to t (their paired gains are far under its +-0.2).

## What worked

- **A 2 is a full heavy card against a 5-card opponent** (TWO_EXTRA): the net throws a pair of 2s before a
  high single at the start of a round. About +0.15 a round by paired playouts.
- **A charge for throwing a single that can still be paired** (LIVE, 5-card opponent only): against the
  counter a kept 10 gets a twin from the open pile one turn in eight. About +0.1 a round.
- **The paired playout diagnostic.** It found both, showed where (me 5 against 5), and showed that the
  re-measured tables do nothing, at a tenth of the working test's noise.

## What did not

- **Re-measuring the 4-card constants.** P(call next) by card count, the catch threshold by card count, the
  card charge against 4 or 5 cards, the callable bonus against 5 cards, the open-card charges against 5
  cards: the measured values are close to the old ones, the net's decisions hardly move, and no gain can be
  measured. The Yaniv thresholds need no change either: in the 5-card phase nearly every call is against
  5 cards, is caught 1-2% of the time, and is already made.
- **Taking anything from the counter's play** in this phase: its different moves are worse.
- On the working test the two kept changes together are +0.16 margin and +2.2 points of matches over the
  bar (`j` alone: +0.03 and +0.7), inside its noise. The honest claim is "not worse than the bar, probably
  +0.15 to +0.25 a round against the counter, nothing against the 4-card net itself".

## What I would try next

- The first-order gains left in this structure are small. The trained net no longer shows a better move in the
  5-card phase, and simple one-move alternatives all lose, so a better 5-card phase needs something neither
  net does: for example valuing two throws ahead (which pair to build, not only which group leaves next), or
  using what the opponent's throws say about its hand.
- The paired diagnostic against a stronger reference than the trained net (a search player), to find more
  patterns like the pair of 2s.
- The measured catch-threshold table for the phase after the handover (+0.07 +- 0.07 over the whole game):
  that is the 4-card net's ground, and would need a 20,000-round paired run to settle.
