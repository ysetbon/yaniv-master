# Designer `mine`: 5-card-phase play and the draw decision

Hero hm5 in every play-out; gain = B's margin - A's margin (A = hm5's move), points per occurrence (+- se).
"p5" = either hand still holds 5 cards; "end" = both hands <= 4 (only draw-decision classes are mine there).
Helper: training/yaniv/contest_opus_mine.py. Scratch: scratchpad/mine/.

## 1. Broad map: the trained net's move as B (10,000 rounds, seed 5200)

Disagreements: 0.87 a round vs counter, 0.95 vs trained. Classes with |gain| > 2 se on both opponents:

| class (B = trained net's move) | per round | vs counter | vs trained |
|---|---|---|---|
| p5: hm5 throws a set, trained a lighter single | 0.04 | +2.43 (0.88) | +3.82 (1.00) |
| p5: hm5 draws the deck, trained takes an open 4-6 | 0.03-0.04 | +1.52 (0.80) | +2.95 (1.14) |
| p5: single vs single, same points (another rank) | 0.17 | -1.12 (0.37) | +0.07 (0.36) |
| p5: single vs a heavier single | 0.08 | -1.47 (0.60) | -0.59 (0.58) |
| end: deck -> open 4-6 (draw only) | 0.08 | +1.17 (0.66) | +1.01 (0.67) |

hm5's own margin in these runs: +1.83 vs counter, +0.06 vs trained.

## 2. Broad map: the card counter's move as B (10,000 rounds, seed 5200)

Disagreements: 1.36 a round vs counter, 1.51 vs trained. Almost every class is negative for the counter's move
(hm5 is the better player there), with one exception, the same one as the trained net's:

| class (B = counter's move) | per round | vs counter | vs trained |
|---|---|---|---|
| p5: hm5 throws a set, counter a lighter single | 0.04 | +2.54 (0.97) | +4.72 (1.07) |
| p5: set -> heavier single | 0.17 | -1.44 (0.45) | -2.43 (0.42) |
| p5: deck -> open 4-6 | 0.11 | -1.05 (0.59) | -1.13 (0.57) |
| end: deck -> open 4-6 | 0.30 | -0.77 (0.35) | -0.83 (0.31) |

## 3. What the "set -> lighter single" class is (trained net's move as B, finer tags, seed 5200)

Nearly every case: I hold a pair of R, **the open card is also an R**; hm5 throws the pair and draws the deck,
the rival throws another single and takes the open R (three R's to throw next turn). E.g. vs counter
pair of 10-pts + single 5..8: +2.9 to +6.6 each (n 25-45), vs trained +3.3 to +15.

## 4. Rule TRIPLE as an explicit alternative (seed 5300, 10,000 rounds each)

B = keep all my R's (set of R matching the open card), throw my heaviest card of another rank, take the open R.
| sub-class | vs counter | vs trained |
|---|---|---|
| heaviest other card >= 4 pts, opp 4-5 cards, R >= 4 pts | +2.72 (0.86), 0.046/rd | +2.99 (0.96), 0.040/rd |
| same, opp <= 3 cards | +3.71 (4.29), n 24 | +6.10 (3.56), n 30 |
| heaviest other card <= 3 pts, opp 4-5 | -1.71 (1.34), 0.013/rd | -0.51 (1.37) |
| heaviest other card <= 3 pts, opp <= 3 | -6.00 (3.32) | -6.95 (2.40) |
| R <= 3 pts | negative, small n | |

So: R worth 4+, the card thrown instead worth 4+. hm5's score gap A - best such B in these positions (3,000
rounds vs counter, 167 positions): median 3.3, 90% 4.75, 97% 5.2: a bonus of one card charge (5) flips most.

## 5. The draw decision alone: same throw, the other draw (8,000 rounds, seed 5400)

6.3 flips a round (every decision). By by open-card points x does it pair a kept card: hm5's draw is right in every
class with n > 100 on both opponents; the best are flat (deck->open 3: +0.10 (0.82) / +0.42 (0.64); open->deck
4 unpaired: +0.22 (0.75) / +0.48 (0.80)). Split further by my / its card count: no cell positive on both beyond
noise. Taking a card that pairs a kept card is right by -3 to -6 points a time (hm5 always does).
Trained net's draw-only disagreements (seed 5500): open->deck when the open card would make me callable
(held <= 7, open 3-5): +1.5 (1.2) / +2.0 (0.7), +1.1 (1.6) / +3.7 (1.1) - that is the Yaniv/endgame area of
another designer, and not significant vs counter; not pursued.

## 6. Block `add_mine_units` (TRIPLE), 10 units, layer 1

One unit per rank R in 4..K: relu(take_open + open_R + 4 hand_R - 4 throw_R - 4 [throw is A/2/3/joker] - 3),
output +5 (one card charge: the third R leaves with the pair next turn, so the extra card costs no turn).
Built: [125, 228, 98, ...], 568 hidden (hm5 558 + 10). An earlier draft with a layer-2 points step cost 42
units (inputs carried to layer 2): replaced by the thrown-rank test in layer 1.

Variant `mine_triple5` (bonus 5): play-out of the changed decisions (hero hm5, B = the variant's move, seed 5600,
10,000 rounds each):
| changed decisions | vs counter | vs trained |
|---|---|---|
| all (TRIPLE: 0.048 / 0.043 a round; other: 1 in each run) | +3.40 (0.85) | +2.90 (0.89) |
| p5, opp 5 | +4.09 (0.98), 0.032/rd | +4.10 (1.15), 0.025/rd |
| p5, opp 4 | +0.33 (2.78) | +0.75 (2.25) |
| end (both <= 4), opp 4 | +1.49 (2.44) | +0.46 (2.08) |
=> about +0.15 / +0.12 a round.

Whole game, contest_opus_eval seed 310 (4,000 rounds, 800 matches, same seed for both):
| opponent | hm5 | mine_triple5 |
|---|---|---|
| counter | +1.78, 56.6%, matches 73.0% | +1.89, 57.0%, 75.1% |
| trained | +0.08, 51.4%, 65.5% | +0.24, 52.1%, 65.2% |
| hm5 | 0.00, 50.0% | +0.15, 50.5%, 51.1% |

Variant bonus 8 (hand-picked second value, to flip the ~5% of positions with a gap above 5): differs from
bonus 5 in 0.002 decisions a round (seed 5900): no change worth having. Kept 5.

## 7. Other rules tested (all failures)

- DENY (seed 5700, 8,000 rounds): hm5 leaves face up a rank the opponent is known to hold (despite its FEED
  charge); B = hm5's best move that does not. 0.035 / 0.024 a round; -5.3 (1.2) / -4.0 (1.2) end,
  -3.5 (1.8) / +1.8 (3.2) p5. hm5 is right (those are cheap cards it must shed).
- KEEPLIGHT (seed 5700): hm5 throws an A / 2 / joker while holding 4+ cards; B = hm5's best move keeping them.
  0.12 a round; -2.5 (0.6) / -1.6 (0.6) (mostly a pair of 2s: the known "shrink first" rule holds). A single 2:
  +0.8 (2.5) / +2.9 (1.7), n 70: noise.
- Trained net's finer tags with mine_triple5 as hero (seed 5800): no class beyond 1.5 se on both opponents.
  Best: both take the open R for a triple but throw different singles, +1.75 (1.44) / +1.53 (1.72), n ~ 50.
- Inference from the opponent's draws (it passed an open X, so it likely holds no X): not in the 125 inputs
  (no draw history beyond known_), so it cannot be a rule of this net.
