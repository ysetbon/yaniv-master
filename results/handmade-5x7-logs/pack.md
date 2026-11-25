# Packaging log: the switch player as one model file

Files: `training/yaniv/handmade5_pack.py`, `training/yaniv/test_handmade5_pack.py`,
`public/model/yaniv-net-handmade.json` (for now a stand-in, see the end), one line in
`src/lib/yaniv/engine.ts`, two changes in `src/components/yaniv/game.tsx`. Nothing trained, nothing searched.

## How the game uses a decider
- `moe.py`: weights = softmax(decider(96 state inputs)) in float32 (`z - max`, `exp`, divide by the sum). In play
  (`_play_forward`) the experts' layers are stacked with `np.stack`, so **in Python the experts must have one
  shape**; a decision's rows share one table, so the decider runs once.
- `src/lib/ai/net.ts`: the decider's outputs are float32, the softmax is float64 (`Math.exp`). Each expert is run
  on its own (they need not share a shape there). Every unit is summed in float64 and kept as float32.

## The decider (hand-written, 96 -> 2 -> 2)
S = 8000, G = 2000; `my_cards` and `opp_cards` are count / 5. For "both" (the default):

    my_big  = ReLU(40000 * my_cards  - 8000 * (at + 0.5))    0 with `at` cards or fewer, at least 4000 with more
    opp_big = ReLU(40000 * opp_cards - 8000 * (at + 0.5))
    output 0 (main net)    = my_big + opp_big        ("mine" / "opp": only that unit)
    output 1 (endgame net) = 2000

"either" is the mirror image (units that fire on a small hand feed output 1; output 0 is 2000). The outputs are
always at least 2000 apart. exp(-2000) is exactly 0 in float32 (anything below about -104 is: exp(-103) = 1e-45,
exp(-104) = 0) and in float64 (anything below about -745.2 is), so the weights are exactly (1, 0) or (0, 1), and
1 * chosen + 0 * other is the chosen net's score bit for bit while the other score is finite (0 * a Yaniv row of
+-1000 is 0; a unit test adds 1e30 to the other net and nothing changes). Checked for every rule, every `at`
0..5 and all 36 pairs of card counts, with the other 94 inputs random: weights exactly 1 and 0 every time.

## One shape for both nets
Narrower layers get units with weight 0 in and out. The shallower net gets pass-through layers (weight 1) after
one hidden layer (exact, hidden units are never negative); the layer is picked so the common shape is smallest.

## Proof
`python -m training.yaniv.handmade5_pack MAIN.json --out X.json --check 20000` plays real-game rounds (5 cards,
no env vars; both seats the switch player, 1 move in 10 random, random match scores 4 times in 10) and compares
at every decision. "padded" = against `switch_chooser` of the file's own two experts; "original" = against
`switch_chooser` of the nets as given.

| main stand-in (shape) | decisions (after hand-over) | score rows | weights not exactly 1/0 | padded: moves differ / max score diff | original: moves differ / max score diff |
|---|---|---|---|---|---|
| (a) the 4-card net itself | 20,020 (10,602) | 197,815 | 0 | 0 / 0 | 0 / 0 |
| (b) `network.default_net(seed=3)` [125,128,128,1] | 20,019 (7,096) | 180,179 | 0 | 0 / 0 | 5 / 6.1e-5 |
| (b) `handmade_designs.planner` [125,126,112,124,118,115,113,133,112,96,1] (deeper: the 4-card net is deepened) | 20,006 (10,729) | 195,330 | 0 | 0 / 0 | 5 / 6.1e-5 |
| (b) `handmade_designs.scorer` [125,155,100,46,45,1] | 20,005 (8,866) | 205,808 | 0 | 0 / 0 | 5 / 6.1e-5 |
| (b) random [125,64,40 x9,20,1] | 20,006 (5,266) | 200,697 | 0 | 0 / 0 | 6 / 6.1e-5 |

**Finding: in Python a padded net is not bit-identical to itself unpadded.** numpy's matrix product (OpenBLAS)
rounds the last bit differently for matrices of different shapes (I measured it directly: the same weights with
zero columns added change up to a third of the outputs in the last bit; it depends on the row count and widths).
6.1e-5 is one float32 step at scores near 1000. So against the nets in their own shapes about 1 move in 4,000
differs: near-ties decided by the last bit. Against the same nets in the common shape (`--nets PREFIX` writes
them) the packed model is bit for bit. This cannot be fixed from the packing side; it would need `moe.py`'s
`_play_forward` to run experts of different shapes one by one instead of stacking them (then no padding is
needed at all, and the file is smaller). I did not change `moe.py`.

**In the browser there is no such effect**: `net.ts` sums each unit input by input, so zero units change nothing.
Checked with `net.ts` itself (tsx; scratch `check_ts.ts`) on 600 real-game positions (4,883 rows, 273 after the
hand-over) for the shipped file and three padded ones: weights exactly 1/0 every time, and 0 rows differ from
the chosen net run alone in its own shape.

`handmade_eval --game 5x7 --workers 2` (shared seeds):

| player | margin | rounds | caught | matches |
|---|---|---|---|---|
| `public/model/yaniv-net-handmade.json` (stand-in a) | +1.52 | 1096/2000 (54.8%) | 48/1044 (4.6%) | 293-107 (73.2%) |
| `switch:4card:4card` | +1.52 | 1096/2000 | 48/1044 | 293-107 |
| the 4-card net alone (same thing) | +1.52 | 1096/2000 | 48/1044 | 293-107 |
| packed planner stand-in | +1.05 | 1072/2000 (53.6%) | 47/1030 (4.6%) | 141-59 of 200 |
| `switch:` of the two padded nets | +1.05 | 1072/2000 | 47/1030 | 141-59 |
| `switch:planner:4card` (own shapes) | +1.05 | 1072/2000 | 47/1030 | 141-59 |

Identical in all three rows of both groups (on these games no last-bit tie changed a result; it can on others).

## Python against the browser (not caused by packing)
On the same 600 positions the browser's scores and numpy's differ by up to 0.94 on rows scoring about -600 and
by about 0.01 on ordinary rows, for the 4-card net alone as much as for any packed file. Cause: the 4-card net's
first layer has weights of 10,000 and sums of that size that cancel; numpy keeps float32 throughout (one step at
10,000 is 0.001), the browser sums in float64. The best move was the same in 600 of 600 positions for the shipped
file and 599 of 600 for the planner stand-in. The repo's fixture test (`npm run test:rules`) compares a small
random decider + experts to 1e-5 and still passes; I did not add the packed file to it (not my files).
Designers of the main net: sums of huge numbers that cancel cost accuracy in Python, not in the browser.

## File size
Stand-in (a): 540,609 bytes. With a main net like `planner` (9 hidden layers of about 115): 1.54 MB; like
`default_net`: 1.60 MB (both experts are stored at the larger width of every layer). If a real main net pushes
it past about 3 MB: narrower main-net layers, or store the experts unpadded (the browser already allows it;
Python needs the `moe.py` change above).

## Browser
- `src/lib/yaniv/engine.ts`: `GAMES` gets `{ id: "5x7-handmade", hand: 5, yanivAt: 7, net: "-handmade",
  label: "Standard: hand-written net" }`, which loads `/model/yaniv-net-handmade.json`.
- `src/components/yaniv/game.tsx`: the "Small game: ..." part of the subtitle now shows only when the rules are
  not the real game's (it would have said "Small game: 5 cards, Yaniv at 7"); and no training report is fetched
  for a hand-written net (it gave a 404 console error, also for `4x7-handmade`).
- Checked with Playwright/Chromium on `npm run dev`: `/?game=5x7-handmade` fetches the model (200, kind moe,
  decider [96,2,2], 2 experts), subtitle "You against a neural net whose weights were written by hand, not
  trained.", 10 of my turns played with the net answering each (pairs thrown, Yaniv called at 3 cards), no
  console errors or warnings. `/` (standard) still loads `yaniv-net.json` and its report, no errors.

## Tests
- `python -m unittest discover -s training/yaniv -t . -p "test_*.py"`: 88 tests OK (3 skipped), 10 of them new
  (4 s).
- `npm run test:rules`: moe ok 6 cases, parity ok. `npx tsc --noEmit` and `npx eslint src`: clean.

## When the real main net arrives
    python -m training.yaniv.handmade5_pack MAIN.json --out public/model/yaniv-net-handmade.json --check 20000
(add `--when mine|opp|either`, `--at N` if the chosen hand-over differs; `--nets PREFIX` for the padded copies.)
The shipped file is now stand-in (a): the 4-card net in both seats, i.e. it plays as the 4-card net alone.
