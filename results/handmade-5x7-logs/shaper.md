# shaper: log

Design: `training/yaniv/handmade5_designs/shaper.py` (464 hidden units, layers [125, 199, 86, 42, 30, 28, 48, 24, 7, 1]).
Saved net: `/tmp/claude-0/-home-user-yaniv-master/a532e2e0-83d9-59dd-8a1a-414723f6a44f/scratchpad/shaper/shaper.json`
(= `v2.json` there; rebuild with `python -m training.yaniv.handmade5_designs.shaper OUT.json`).

Angle: hand shaping with 5 cards. The net is the 4-card net's throw value, Yaniv rule and match-score units
(copied so the constants can be set) plus two units: **throw a pair of 2s while both players hold 5 cards**
(+6), and +2 for any set thrown then. Everything else I tried from the angle's list either showed no gain in
the matched comparisons or no effect on the working test.

## Result (working test: 6,000 rounds seed 61000, 1,500 matches seed 61001, against the counter)

| player | margin | rounds won | caught | matches |
|---|---|---|---|---|
| bar: 4-card net, whole game | +1.49 | 3315/6000 (55.2%) | 4.5% | 1029-471 (68.6%) |
| **shaper with the handover** (`switch:shaper:4-card net`) | **+1.61** | 3352/6000 (55.9%) | 4.5% | 1041-459 (69.4%, 67.0%-71.7%) |
| shaper alone, whole game | +1.61 | 3352/6000 (55.9%) | 4.5% | 1041-459 (69.4%) |

Switch and alone are the same games: shaper differs from the 4-card net only while both players hold 5 cards.
**The gain over the bar (+0.12 margin, +0.7% rounds, +0.8 points of matches) is inside the working test's noise
(+-0.2, +-2.5).** The evidence that it is real is the matched comparison below (t = 4.5 on new deals), which
predicts about +0.2 margin.

## Method: matched comparisons, not more variants

`SCRATCH/shaper/diag.py`: the 4-card net plays the counter; at 40% of its decisions made while either hand
holds 5 cards, its move and every other legal move are each played to the end of the round on the real deck
(same players afterwards), and the difference in (counter's points - mine) is recorded with the hand, the
open card, unseen counts. `diag2.py`: the same with one alternative only, the move the trained net would play.
Means are grouped by class of (chosen move, alternative). A class mean has +-0.5 to +-2 noise; with hundreds
of classes, t = 2.7 turns up by chance, so a class counts only if it repeats on new deals.

**Finding 1: the 4-card net's moves are locally right in almost every class** (36,000 rounds in three batches; the table is from
the first 8,000, 93,000 alternatives, and the later batches agree). Mean gain of the alternative, 4-card net as hero:

| class (while a hand holds 5 cards) | alternative's gain |
|---|---|
| threw a pair, alt: the highest single instead (pair 8-10 / 5-7 / 3-4, me 5 opp 5) | -3.3 +-0.5 / -1.7 +-0.8 / -4.0 +-1.2 |
| threw a pair with opp on 4, alt: highest single | -6.2 +-0.8 |
| threw a single, alt: a pair of aces | -3.4 +-1.0 (5 v 5), -6 to -9 otherwise |
| drew from the deck, alt: take an unpaired open 10 / 9 / 8 / 7 / 6 / 5 / 4 | -2.3 / -1.7 / -2.7 / -2.2 / -2.9 / -1.0 / -1.2 (+-0.3 to 1.0) |
| took an open card that pairs (3 to 10), alt: deck | -3.6 to -7.0 |
| took an open ace / joker / 3, alt: deck | -3.9 / -7.7 / -1.8 (+-1.5 to 2.6); open 2: +0.4 +-2.1 |
| kept a pair one more turn (alt to throwing it: top single), 2 of its rank unseen, opp on 5 | -1.1 +-0.6 (pair 8+), -1.7 +-0.5 (pair < 8): waiting for the triple does not pay |
| two pairs, threw the higher, alt: the lower | -2.2 +-0.8 (opp 5), -7 to -10 (opp 4 or fewer) |
| threw the highest single, alt: a lower single | -1 to -12 in every cell |
| called Yaniv, alt: play on | -5 to -22 |

So the card charge of 5, the open-card thresholds (take 3 or less, or a pair-maker), "pair before a higher
single", and "do not wait for a triple" all carry over to 5 cards unchanged.

**Finding 2: one class is wrong: a pair of 2s while both hold 5 cards.** The 4-card net counts a 2 as a light
card to keep (half a heavy card) and throws a high single; throwing the 2s instead:

| me / opp cards | first 16,000 rounds | new 20,000 rounds |
|---|---|---|
| 5 / 5 | +3.3 +-0.9 (n=459) | **+3.9 +-0.9 (n=511)** |
| 5 / 4 | -1.1 +-1.0 | +0.6 +-0.9 |
| 5 / 3 | -4.7 +-1.3 | -1.9 +-1.1 |
| 4 / 5 | -0.4 +-1.5 | +0.2 +-1.4 |

The same gain at 5 v 5 in every band of my other points (+2.9 / +4.1 / +3.4) and every ply band. Why: with
both on 5 cards nobody can call for a couple of turns, a 5-card hand that keeps 2-2 cannot get to 7 points, and
the 4-card net never throws the 2s later either, so the hand stays a card longer for the whole round.
`diag2.py` agrees and sizes it: the trained net's moves, where they differ from the 4-card net's, gain +0.85
+-0.28 a time at 5 v 5, and nearly all of it is this one class (single -> pair of 2s: +5.0 +-0.75, n=574 in
16,000 rounds; 1,612 single -> set deviations x 3.4 = about +0.35 a round, against -0.17 for its different
singles). With the unit in (v2) the trained net's deviations at 5 v 5 gain +0.10 +-0.26: nothing left there.

**Finding 3 (too rare to matter):** between two unpaired singles of equal points, throwing the rank with fewer
unseen cards (the one that can no longer be paired, by me or by the opponent) gains about +2 +-0.9 a time
(+3.8 +-1.7 n=72, +1.2 +-1.0 n=131; the reverse choice -2.1 +-1.0 n=173), but it comes up ~0.02 times a round.

## Variants measured (working test; rounds only unless matches are given; switch player)

| # | change | reason / expectation | result | kept? |
|---|---|---|---|---|
| bar | 4-card net, whole game (measured myself; same numbers as BRIEF) | baseline | +1.49, 55.2%, caught 4.5%, matches 1029-471 (68.6%) | - |
| v0 | shaper.py with every constant at the 4-card value, no shaping unit | must rebuild the 4-card net exactly | weights identical to the shipped file (checked array by array) | base |
| v1 | TWOS55 = 4: +4 for a set of 2s while both hold 5 | finding 2; expected +0.1 to +0.2 | +1.57, 55.6% | superseded |
| v2 | TWOS55 = 6, SET55 = 2 (+2 for any set at 5 v 5) | v1's +4 flips too little: the pair's score trails the single by 4 - 0.94 x my second-highest card, up to ~6; SET55 also tips K,Q,3,3,x (single -> 3-3: +3.0 +-1.9, n=97). Expected ~+0.2 over the bar | **+1.61, 55.9%, caught 4.5%; matches 1041-459 (69.4%)** | **kept (final)** |
| v3 | v2 + OUTS = 38: 1 point per unseen card of each kept rank (39 units) | finding 3; expected <= +0.05, not measurable | +1.59, 55.7% | dropped: no effect, 39 units |
| v4 | v2 with TWOS55 = 12: the 2s go before another pair too | trained net's subset +7.5 +-2.4 (n=54), then +2.9 +-2.7; all alternatives +0.5 to +1.5 +-1.5 to 2.3: expected neutral to slightly up | +1.59, 55.9% | dropped: neutral |
| v5 | v2 + TWOS45 = 7: the same bonus while I hold 4 and the opponent 5 | trained net's subset +6 +-1.7; all alternatives 0 +-1.4: expected neutral | +1.54, 55.8% | dropped: neutral |

All five differences are inside +-0.2: the working test cannot rank v1-v5. v2 was chosen on the matched
comparison, not on these numbers. No constant went through more than three values.

Checks on v2: `diag.py` again with v2 as hero (16,000 rounds): its pair of 2s thrown at 5 v 5, alt the top single:
-3.1 +-1.3 (the choice holds up); best of 431 classes t = +2.7 (chance level). `diag2.py` v2 against the
trained net's moves: all cells within +-1.3 of zero, 5 v 5 +0.10 +-0.26.

## What worked
- The diagnostic. It showed in half an hour that six of the seven ideas on my list have nothing to gain over
  the 4-card net, and found the one class that does.
- Throwing 2-2 at 5 v 5 (v2): about +3.6 points each time it comes up, 0.06 times a round.

## What did not (each with the comparison that says so)
- Two-step plan: in every class where a longer plan would change the move (pair v higher single, lower pair v
  higher pair, which single) the 4-card net's move is already the better one. Both orders of "pair then
  single" reach the same hand, so the second step adds no information.
- Keeping a pair for a triple: -1.1 to -1.7.
- Deck-draw pairing chance as value / unseen counts (OUTS): real but about 0.04 a round.
- A different card charge with 5 cards, or a general "first shrink" bonus: set-v-single classes are all on the
  4-card net's side except 2-2 (and 3-3 by one point in one hand shape).
- Taking more or fewer open cards: every open-card class is on the 4-card net's side.
- The 2-2 bonus outside 5 v 5 (v5), or ahead of another pair (v4): neutral.
- Not measured: turns-to-callable as an explicit unit (no class pointed at it), leaving the opponent a
  pairable card beyond the existing known-card penalty.

## What I would try next
- The trained net alone scores +1.92 margin on the working test and +1.69 with the handover, so about half of
  its round edge over the 4-card net is in play after both hands are at 4 or fewer, where this design changes
  nothing. `diag2.py` pointed at the 4-card phase (hero = 4-card net, alternative = the trained net's move)
  would show which classes carry it; in the 5-card phase v2 leaves it nothing.
- Cells with the opponent on 5 and me on 2-3 cards (taking an open 4-6: +2 to +3 +-1.7 in one run, -1 to -2
  in the next) need 5x the sample to say anything.
- The gain here is small against the test's noise; confirm v2 against v0 on the lead's larger test.

Scripts (scratch dir): `diag.py`, `diag2.py`, `ana.py` ... `ana7.py` (class tables), result files `d_*`, `e_*`,
`f_*` (4-card net, all alternatives), `t_*`, `u_*` (trained net's moves v v1 / v2), `g_*` (v2, all alternatives).
