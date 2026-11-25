# merge (round 2): one net from port, counts, watcher and shaper

Design file: `training/yaniv/handmade5_designs/merge.py`. Saved net: `SCRATCH/merge.json`
(SCRATCH = `/tmp/claude-0/-home-user-yaniv-master/a532e2e0-83d9-59dd-8a1a-414723f6a44f/scratchpad/merge`;
rebuild with `python -m training.yaniv.handmade5_designs.merge OUT.json`), layers
[125, 211, 87, 42, 30, 28, 48, 24, 7, 1], 477 hidden units: the 4-card net's 462 + 15 (rule 1: 1, rule 2: 12, the cap: 1, one input carried forward).

**Short answer.** The four designs found the same single effect by four routes: while both players hold 5
cards, shrink the hand first (a pair of 2s, or any low set, before a high single). On top of that only one
thing stacks: port's charge for throwing a single whose rank can still be paired. So the merge is the 4-card
net plus port's two rules, both gated on "the opponent holds 5 cards", plus a cap on the Yaniv threshold's
`high_call` term that changes no move. It plays as port does (17 of 70,824 decisions differ, +0.001 +- 0.002),
with fewer units (477 against 511) because it imports the 4-card net's units instead of rebuilding them as
tables. Everything else was measured on top and adds nothing.

## Final numbers

Working test (6,000 rounds seed 61000, 1,500 matches seed 61001, against the card counter; noise about +-0.2
margin, +-2.5 points of matches). Lines other than merge are copied from the designers' logs.

| player | margin | rounds won | caught | matches |
|---|---|---|---|---|
| the bar: 4-card net, whole game | +1.49 | 3315/6000 (55.2%) | 4.5% | 1029-471 (68.6%) |
| port, switch / alone | +1.65 / +1.66 | 55.9% / 55.9% | 4.3% | 1062-438 (70.8%) / 1057-443 (70.5%) |
| counts, switch / alone | +1.64 / +1.71 | 56.1% / 56.4% | 4.4% / 4.5% | 1062-438 (70.8%) / 1056-444 (70.4%) |
| watcher, switch = alone | +1.63 | 56.0% | 4.5% | 1060-440 (70.7%) |
| shaper, switch = alone | +1.61 | 55.9% | 4.5% | 1041-459 (69.4%) |
| **merge, switch** (primary) | **+1.66** | 3355/6000 (55.9%) | 4.3% | **1063-437 (70.9%, 68.5%-73.1%)** |
| **merge, alone** | +1.66 | 3355/6000 (55.9%) | 4.3% | 1063-437 (70.9%) |

Switch and alone are the same games: the merge differs from the 4-card net only while the opponent holds 5
cards, which is always before the handover. Against the bar: +0.17 margin, +0.7 points of rounds, +2.3 points of matches, each inside this test's noise,
as for every round-1 design. The evidence is the paired playouts:

| paired playouts (`diag2.py`, hero hands over to the 4-card net; gain a round +- error) | 20,000 rounds, seed 910001 | 20,000 rounds, seed 910005 |
|---|---|---|
| merge over the 4-card net | +0.264 +- 0.050 (3,126 of 70,722 decisions differ) | **+0.427 +- 0.050** (3,183 of 71,130) |
| ... of it at (me 5, opp 5) | +0.254 +- 0.043 | +0.330 +- 0.043 |
| ... at (4, 5) / (3, 5) | +0.022 +- 0.023 / -0.012 +- 0.010 | +0.088 +- 0.024 / +0.009 +- 0.009 |
| port.json over the 4-card net | (the same net to 17 decisions) | +0.428 +- 0.050 |
| merge over port.json (seed 910004: port's move where it differs from merge's) | port's moves +0.001 +- 0.002 (17 of 70,824) | |

The two samples differ by 2.3 errors; together about +0.35 +- 0.035 a round. The gain is first-order (the
variant makes one move, the base plays on) and at 0-0 scores.

## Method

`SCRATCH/diag2.py` is port's paired-playout script, copied unchanged: the base net plays the counter with the
handover; at each of its decisions while either hand holds 5 cards (all such cells: I hold 5 and the opponent
1-5, or I hold 1-4 and the opponent 5; checked in the output, e.g. `me5 opp3`, `me2 opp5`) every variant is
asked for its move, and where it differs both moves are played to the end of the round on the same hidden
cards, the base policy afterwards. Seeds 910001-910005, 20,000 rounds a run (error +-0.01 to +-0.07 a round
depending on how many decisions differ). Variants are built by hand from `merge.py` with `NAME=value`; nothing
loops over coefficients, no constant was given more than the values listed, and nothing is fitted.
`merge.py` with the round-1 constants reproduces the round-1 nets: 0 of ~2,050 decisions differ from the
4-card net, watcher.json and shaper.json; 3 of 2,057 from port.json and 4 of 2,054 from counts.json (their
catch bonus has a 0.01 base chance the 4-card net leaves out, and counts has its own P(call) table).

## Every variant measured

"vs X" = paired gain a round over base X, +- error, (differing decisions). Cells are (my cards, opponent's).

**Run 1, seed 910001, base = the 4-card net** (`v0`). Question: how big is each round-1 design by the same
instrument, and in which cells? Expected: all about +0.15 to +0.25, at (5, 5).

| variant | result | read as |
|---|---|---|
| `a1` 2s whole against 5 cards (port's rule 1) | +0.153 +- 0.037 (1,392); (5,5) +0.142 +- 0.029, (4,5) +0.018 +- 0.020, (3,5) -0.007 +- 0.009 | port's number (+0.15 +- 0.06) again |
| `a2` the same, only while I hold 5 (shaper's phase) | +0.142 +- 0.029 (812) | all of it is (5,5) |
| `a3` the same while I hold 5 and the opponent 4-5 (watcher's phase) | +0.144 +- 0.042; (5,4) **+0.002 +- 0.030** (928) | nothing at 5 v 4 (port: -0.06 +- 0.05; watcher's three samples: +0.6, +2.4, +2.7 a decision; shaper -1.1, +0.6) |
| `a8` a1 + heavy-card charge 8 against 5 cards | +0.216 +- 0.046; (5,5) +0.219 +- 0.039, (4,5) +0.006 +- 0.023 | the charge adds about +0.07 at (5,5) |
| `a9` a8, only while I hold 5 | +0.219 +- 0.039 (1,435) | |
| watcher (rebuilt) | +0.209 +- 0.053; (5,4) -0.011 +- 0.036 | = a9 + nothing at 5 v 4 |
| shaper (rebuilt) | +0.224 +- 0.039 (1,441) | = a9 by another unit |
| port (rebuilt) | **+0.264 +- 0.050** (3,126); (5,5) +0.254 +- 0.043 | the largest |
| counts' ALPHA + LIVE in merge.py / counts.json | +0.177 +- 0.075 / +0.181 +- 0.075 (6,500); (5,5) +0.236 +- 0.055, the other eight cells together -0.06 | the same (5,5) gain, twice as many changed decisions |

**Run 2, seed 910002, base = `a8`** (2s whole + charge 8, against 5 cards). Question: what stacks on the
"shrink first" rule? Expected: port's and counts' live-rank charges are the same gain; ALPHA about 0.

| variant | result | read as |
|---|---|---|
| `b1` + port's LIVE (0.5 x points / 10 a copy, singles, against 5 cards) | **+0.067 +- 0.025** (1,284); (5,5) +0.069 +- 0.023 | stacks. By kind: taking a low open card and throwing the lower single that can no longer be paired (770 decisions, +0.06); which of two tens (101, +2.7 each) |
| `b2` + counts' LIVE against 5 cards (1 a copy, any card, sets too) | +0.012 +- 0.023 (953) | not the same gain: flat and on sets it also turns "single + open card" into "set + deck" (116, -4.1 each) |
| `b3` + counts' LIVE 1 / 1 / 0.5 against 5 / 4 / 3 cards | -0.016 +- 0.030; (5,4) -0.032 +- 0.018 | no |
| `b4` + ALPHA[5] = 0.15 | **-0.277 +- 0.063**; (5,5) -0.036 +- 0.051 (2,416), (4,5) -0.177 +- 0.032, (3,5) -0.060 +- 0.019 | nothing at (5,5); with 2s charged whole and other points at 0.15 a 2 is dearer to keep than a king, wrong when I hold 3-4 cards and want to call |
| `b5` + ALPHA 0.15 / 0.35 / 0.6 / 0.85 | -0.294 +- 0.070; over b4: (5,4) -0.020 +- 0.028, (5,3) +0.005 +- 0.011 | nothing against 4 / 3 cards either |
| `b6` + shaper's SET55 = 2 | -0.023 +- 0.011 (123) | a set is already early enough |

**Run 3, seed 910002, base = `a1`** (2s whole only). Question: are charge 8 and LIVE the same gain?

| variant | result | read as |
|---|---|---|
| + charge 8 (= a8) | +0.068 +- 0.026 (745) | |
| + port's LIVE (= port) | **+0.143 +- 0.032** (1,712) | port's own number: +0.10 +- 0.04 |
| + both (= b1) | +0.144 +- 0.036 (1,990) | **charge 8 adds 0.00 once LIVE is in**: LIVE already tips the low set ahead of the live high single |
| + ALPHA[5] = 0.15 | -0.192 +- 0.068; (5,5) +0.033 +- 0.057, (4,5) -0.164 +- 0.033 | as b4 |
| port + ALPHA[5] = 0.15 | -0.755 +- 0.142 (21,842) | not a fair test and not used as one: 0.05 a point a copy x 3 copies is exactly ALPHA's 0.15 a point, so every single ties |

**Run 4, seed 910003, base = merge.** Questions: the gate of rule 1; LIVE's reach; do the round-1 nets play
any cell better than the merge? Expected: nothing.

| variant | result | read as |
|---|---|---|
| `d1` rule 1 only while I hold 5 | -0.015 +- 0.014 (253); (4,5) -0.014 +- 0.012 | level; the plain gate "opponent holds 5" kept |
| `d2` LIVE against 4 cards too | +0.017 +- 0.013 (363) | not worth the special case |
| `d3` LIVE on sets too | -0.009 +- 0.007 (63) | no |
| counts.json's moves | +0.053 +- 0.061 (4,447); (5,3) +0.035 +- 0.012, (5,4) +0.047 +- 0.034, (5,5) -0.023 +- 0.045 | a hint at (5,3) / (5,4): re-run below |
| watcher.json's moves | -0.056 +- 0.045 (2,563); (5,4) +0.013 +- 0.029 | merge at least as good |
| shaper.json's moves | -0.090 +- 0.035 (1,824) | merge better |

**Run 5, seed 910003, 12,000 rounds, base = merge.** The trained net's and the counter's moves, read by cell:
is there anything to learn where I hold 4 or fewer against 5?

| variant | all | (4,5) | (3,5) | (2,5) | (5,5) | (5,4) |
|---|---|---|---|---|---|---|
| trained net's moves | -0.233 +- 0.094 (5,053) | -0.074 +- 0.029 | -0.025 +- 0.021 | +0.019 +- 0.017 | -0.079 +- 0.067 | -0.079 +- 0.045 |
| counter's moves | -0.859 +- 0.125 (8,091) | -0.141 +- 0.038 | -0.019 +- 0.020 | -0.005 +- 0.009 | -0.457 +- 0.094 | -0.144 +- 0.055 |

**Run 6, seed 910004, base = merge.** The hint from run 4 on new deals, and its parts.

| variant | result | read as |
|---|---|---|
| `f1` + ALPHA 0.35 / 0.6 against 4 / 3 cards | +0.003 +- 0.028 (833) | nothing |
| `f2` f1 + LIVE against 4 and 3 cards too | -0.016 +- 0.035 (1,654) | nothing |
| counts.json's moves | -0.056 +- 0.059; (5,3) -0.000 +- 0.012, (5,4) +0.015 +- 0.032, (4,5) -0.057 +- 0.017 | the hint did not repeat |
| port.json's moves | +0.001 +- 0.002 (17) | the same player |

**Run 7, seed 910005, base = the 4-card net**: merge +0.427 +- 0.050, port.json +0.428 +- 0.050 (table above).

## The questions, answered

1. **Do port's two rules and counts' ALPHA / LIVE stack?** No. ALPHA is the 2s rule by another route (heavy
   points nearly free against 5 cards makes the 4 points of 2-2 worth shedding): 0.0 +- 0.05 at (5,5) on top
   of rule 1, and harmful with it when I hold 3-4 cards. Against 4 / 3 cards it does nothing. Counts' LIVE and
   port's "can still be paired" charge are the same idea, but only port's form (scaled by the card's points,
   single throws only, against 5 cards) gains on top of the shrink-first rule: +0.07 +- 0.025 against
   +0.01 +- 0.02.
2. **Watcher's charge of 8, the measurer's 7.5 / 5.5?** +0.07 on rule 1 alone, 0.00 once LIVE is in (run 3).
   Against 4 cards port measured 7: 0.00 +- 0.03. Left at 5.
3. **Where does the 2s rule belong?** The gain is at (5,5). (5,4): +0.002 +- 0.030, left out. (4,5) and below:
   +0.015 +- 0.014 for having it, so the simplest gate, "the opponent holds 5 cards" with no test of my own
   count, is kept.
4. **Anything for (<= 4, 5)?** No. The 2s rule, the charge of 8, LIVE, ALPHA, counts' whole net, the trained
   net's and the counter's moves: every one is within +-0.03 of zero or negative there (port also tried the
   callable bonus against 5 cards twice: +0.05 +- 0.03, -0.02 +- 0.02). Yaniv is already called whenever legal
   against 5 cards.
5. **The `high_call` wart.** `merge.py` caps the term at 7 points (`cap_high_call`, one layer-1 unit feeding
   the two `call` units). `SCRATCH/wart.py`, 3,000 rounds with random match scores, seed 910050: 20,457
   decisions, 2,151 with 33+ points in hand (up to 49): capped and uncapped nets choose the same move every
   time. On a Yaniv row my points are at most 7, so the cap cannot change a call. The uncapped unit could
   only have mattered through float32 rounding around 2,000; with the cap throw rows stay within -100..+60.

## What is in the merge and why

- The 4-card net, imported unit for unit from `handmade.py` (its Yaniv rule and match-score units unchanged).
- RULE 1 `twos_whole` (1 unit): a 2 kept is charged as a whole heavy card while the opponent holds 5 cards.
- RULE 2 `live_<rank>_vs5` (12 units): 0.5 x points / 10 per unseen copy of a single thrown, same phase.
- `high_call_over` (1 unit): the cap.

## What was left out and why

Charge 8 / 7.5 (the same decisions as rule 2), SET55 (-0.02), ALPHA (0 to -0.2), counts' flat LIVE and LIVE
against 4 / 3 cards (0), the 2s rule at 5 v 4 (0), any gate on my own card count (level, one more special
case), per-count P(call) and catch tables (port and counts: no effect; not re-measured here).

## What I would try next

- The first-order gains in the 5-card phase look used up: the trained net's moves are worth -0.23 +- 0.09
  against the merge's. What is left is beyond one move (which pair to build over two turns) or in the phase
  after the handover (the trained net alone is +1.92 on the working test, +1.69 with the handover).
- The same paired playouts with match scores in play (they start at 0-0 here).
