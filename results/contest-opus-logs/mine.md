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
