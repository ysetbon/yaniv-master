# Designer `counts`: a net organised around the two card counts

Design file: `training/yaniv/handmade5_designs/counts.py` (558 hidden units, 8 hidden layers). Saved net:
`/tmp/claude-0/-home-user-yaniv-master/a532e2e0-83d9-59dd-8a1a-414723f6a44f/scratchpad/counts/counts.json`
(rebuild with `python -m training.yaniv.handmade5_designs.counts OUT.json`).

## Result (working test: 6,000 rounds seed 61000, 1,500 matches seed 61001, against the card counter)

| player | margin | rounds won | caught | matches |
|---|---|---|---|---|
| the bar: 4-card net, whole game (measured by me) | +1.49 | 3315/6000 (55.2%) | 4.5% | 1029-471 (68.6%) |
| **`switch:counts.json:4-card net`** (primary) | **+1.64** | 3365/6000 (56.1%) | 4.4% | **1062-438 (70.8%)** |
| `counts.json` alone, whole game | +1.71 | 3385/6000 (56.4%) | 4.5% | 1056-444 (70.4%) |

Ahead of the bar by +0.15 to +0.22 margin, about 1 point of rounds and 2 points of matches. Each of those is at
or inside the noise of this test (0.2 margin, 2.5 points of matches); they all point the same way, and a separate
5,000-round diagnostic on another seed gave +1.93 against +1.66 for the same two players. Treat it as "probably a
little better, not proven". Alone and with the handover are level.

## The idea

The opponent's card count says how long the round still lasts, so the valuation is gated by it (regimes 5, 4, 3,
1-2 cards, from `opp_has_k`). Starting point: the 4-card net's valuation, rebuilt with every table per regime.
With neutral tables my file reproduces the 4-card net's scores to 0.3 points (checked on 365 decisions).

Value of a throw = - light points held (aces, 2s) - ALPHA[c] x heavy points held + (ALPHA[c] - w[c, est]) x G
- CARD[c] per heavy card - LIVE[c] per unseen copy of the thrown rank + catch bonus with P(call | c, est)
+ the 4-card net's small terms. Yaniv rule and match-score units: the 4-card net's, imported unchanged.

## Measurements (5,000 real-game rounds, 4-card net against the counter, seed 9001; scratch `diag.py`)

P(the counter calls on its next turn), by its card count and my `opp_est` of it (n in the cell >= 20):

| its cards | <5 | 5-8 | 8-11 | 11-14 | 14-17 | 17-21 | 21+ |
|---|---|---|---|---|---|---|---|
| 5 | | | | 0.03 | 0.02 | 0.01 | 0.00 |
| 4 | | | 0.15 | 0.08 | 0.05 | 0.02 | 0.01 |
| 3 | | 0.27 | 0.18 | 0.11 | 0.06 | 0.01 | |
| 2 | 0.43 | 0.35 | 0.27 | | | | |
| 1 | 0.66 | 0.65 | | | | | |
| 4-card net's table (no card count) | 0.55 | 0.30 | 0.18 | 0.08 | 0.04 | 0.01 | 0.01 |

So `opp_est` already carries nearly all of the card-count dependence: the 4-card table was right for the real game.

P(the counter ends the round before my second next turn): 0.01-0.05 against 5 cards, 0.07-0.26 against 4,
0.23-0.38 against 3, about 0.55 against 2, 0.9 against 1.

By cell (my cards, its cards) at my throw: P(it calls next) / P(I call on my next turn) / my remaining turns:
(5,5) 0.003 / 0.010 / 5.6; (5,4) 0.03 / 0.02 / 4.2; (5,3) 0.12 / 0.03 / 2.6; (5,2) 0.33 / 0.01 / 1.4;
(4,5) 0.007 / 0.06 / 4.3; (3,5) 0.013 / 0.17 / 3.1; (2,5) 0.004 / 0.41 / 2.2; (4,4) 0.05 / 0.09 / 3.4.

Where rounds end and what they cost (4-card net; share of rounds, my points, its points): I still hold 5 cards
in 6.3% of rounds and pay 15-17 points there (it pays 1-3); it still holds 5 in 8.5% and pays 14.5-16.6 (I pay
0-3). Ending (3,4) or (2,4) is worth +5 to +9 a round; ending (4,3) or (4,2) costs 4 to 8. The round is a race
in card count; whoever is one card ahead at the end wins about 5 points.

My Yaniv calls against a 5-card opponent: 405 calls, 3 caught (0.7%). Against 4: 4.6%, against 3: 6.4%.
The counter's calls: its total averages 6.3 with 5 cards, 6.0 with 4, 5.5 with 3, 4.8 with 2, 3.5 with 1.

The counter takes my thrown card (1,500 rounds, `take.py`): a card worth 5+ is taken 87% of the time when it
holds that rank and 4% when it does not; a card worth 4 or less is taken 94% / 100%.

## Every variant measured (switch player, working test, rounds only unless said; bar +1.49, 55.2%)

| variant | change and reason | result | verdict |
|---|---|---|---|
| bar | 4-card net alone, with matches | +1.49, 55.2%, caught 4.5%, 1029-471 (68.6%) | reference |
| A | P(call) table per regime (measured, above). Expected ~0: the table barely differs | +1.50, 55.3%, 4.5% | no effect; kept (it is the measurement) |
| B | A + ALPHA 0.15/0.35/0.6/0.85: heavy points outside G are only exposed if the round ends within two turns | +1.56, 55.9%, 4.5% | +0.06, noise; kept on the reasoning |
| C | B + CALLABLE 8/5/2 x P(my hand after the draw <= 7), sure-callable +3 dropped: a call against 5 cards is never caught and wins ~14 | +1.56, 55.9%, 4.8% | no effect; dropped |
| E | B + a 2 nearly free against 5/4 cards (0.5 / 1.0 instead of 2.5): keep 2s as the core of a callable hand | +1.43, 55.0%, 4.5% | -0.13, noise, wrong way; dropped |
| F | B + LIVE 1/1/0.5/0 per unseen copy of the thrown rank (the counter takes a matching card 87%) | +1.64, 56.1%, 4.4% | +0.08 on B; **kept = final** |
| G | F + CARD 7/6 against 5/4 cards: more turns remain, a card slot should cost more | +1.62, 56.0%, 4.4% | no effect; dropped |
| H | F with LIVE 2/2/1 | +1.54, 55.8%, 4.4% | -0.10, noise; dropped (1 is the reasoned value) |
| I | F with ALPHA 0.05/0.2/0.5/0.85 (the raw measurement, no step up) | +1.64, 56.2%, 4.5% | same; F's values kept |
| final, switch | F, with matches | +1.64, 56.1%, 4.4%, 1062-438 (70.8%) | |
| final, alone | F, whole game, with matches | +1.71, 56.4%, 4.5%, 1056-444 (70.4%) | |

Diagnostic of F as the switch player on the diag seed (5,000 rounds): margin +1.93 (4-card net: +1.66); rounds
that end while I still hold 5 cards fell from 315 to 251.

## What worked
- **The live-rank charge** is the one new idea with a measured mechanism behind it: every high single I throw is
  a pair for the counter 87% of the time if it holds the rank. It only has room to act because ALPHA makes the
  points of a heavy single nearly irrelevant against 4-5 cards (K against 9 is 0.15 points, one unseen copy is 1).
- **ALPHA** (heavy points weigh little while the opponent holds many cards) is harmless to mildly positive.
- Together +0.15 margin and +2.2 points of matches on the switch player: at the edge of the noise.

## What did not
- Per-regime P(call) tables: the 4-card table on `opp_est` is already right in the real game.
- A bonus for being able to call next turn (even 8 points against a 5-card opponent): no effect. Holding few
  points is already what the valuation pushes for.
- A higher card charge against 4-5 cards, cheaper 2s, a larger live charge: nothing or slightly worse.
- I did not find a regime where the 4-card net is clearly wrong. The first-principles regime story is true in the
  statistics (the tables above), but the 4-card net already gets most of it through `opp_est` and its card charge.

## What I would try next
- The same live-rank idea for what I keep: prefer keeping the rank with more unseen copies when two throws tie,
  and count `gone` copies when taking an open card onto a pair.
- A real diagnostic of single decisions (rollouts of both candidate moves from the same position), restricted
  to (5,5) and (5,4): the end-cell table says the losses are rounds where I am one card behind, but not which
  move put me there.
- The counter takes any card worth 4 or less 94% of the time: a larger charge for leaving one face up against
  4-5 cards (LOW_TOP is 1).
- Yaniv against a 5-card opponent is caught 0.7%: the rule could call at any total there, but it nearly does.
