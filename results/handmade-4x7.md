# A hand-written net for the 4-card game (Yaniv at 7 or less)

**Result.** A net whose weights were all written by hand, with no training, beats the card counter and wins more matches against it than the trained net does. From a 0-0 score it plays rounds about as well as the trained net; its extra match wins come from playing to the match score.

| Against the card counter | Margin a round | Rounds won | Yaniv calls caught | Matches won (95% range) |
|---|---|---|---|---|
| **Shared seeds** (2,000 rounds seed 41000; 800 matches seed 41001) | | | | |
| card counter itself | 0.00 | 999/2000 (50.0%) | 7.2% | 400–400 (50.0%) |
| trained net `yaniv-net-4cards-yaniv7.json` | +1.52 | 1108/2000 (55.4%) | 5.1% | 490–310 (61.3%, 57.8%–64.6%) |
| **hand-written net** `yaniv-net-4cards-yaniv7-handmade.json` | **+1.44** | **1104/2000 (55.2%)** | 5.0% | **549–251 (68.6%, 65.3%–71.7%)** |
| **Final check on new games** (1,000 rounds seed 31337; 800 matches seed 31338) | | | | |
| trained net | +1.28 | 539/1000 (53.9%) | 4.9% | 483–317 (60.4%, 56.9%–63.7%) |
| **hand-written net** | **+1.39** | **545/1000 (54.5%)** | 4.5% | **537–263 (67.1%, 63.8%–70.3%)** |

Margin is the counter's points minus the net's, a round. The final check was run once, after the design was chosen.

- **Does it beat the counter?** Yes, clearly: about +1.4 points a round and two matches in three.
- **Does it beat the trained net?** On matches against the counter, yes: 68.6% against 61.3% on the shared seeds and 67.1% against 60.4% on new games, with ranges that do not overlap on the bigger test below. On rounds from 0-0 the two are level within noise (+1.44 against +1.52, then +1.39 against +1.28).
- **Head to head** (hand-written against trained, 12,000 rounds and 3,000 matches, seeds 52000/52001): rounds are even (6011/12000, margin −0.12 a round for the hand-written net), and the hand-written net wins 1901–1099 matches (63.4%, 61.6%–65.1%).

## What "by hand" meant

`training/yaniv/handmade.py` builds the net from named ReLU units (`handmade_kit.py` turns them into weights): 462 hidden units in 8 layers, shape [125, 197, 86, 42, 30, 28, 48, 24, 7, 1]. No gradient step, no fitting to moves or outcomes, and no automated search over weights was used anywhere.

What was used, and where it comes close to the line:

- **Reasoned units.** Each unit detects something stated in words (cards of a rank left after the throw, the heaviest group I can throw next turn, "my score plus my hand is exactly 50").
- **Measurement of whole designs.** About 90 variants were played against the counter across eight sub-designs, each one a change with a stated reason, logged with its result. That is a person's trial and error, not an optimiser, but it is still many looks at the same measuring stick.
- **Measured tables turned into hand-picked numbers.** Two examples: how often the counter calls on its next turn, by its estimated total (0.55 under 5 points down to 0.01 above 17), and how far the `assaf` input understates the real catch chance (3 to 8 times).
- **Watching the trained net.** One sub-design (`reader`) recorded what the trained net plays and described it in rules ("throws a pair of 3s or more before a higher single", "takes an unpaired open card only if it is worth 3 or less"). Those rules were written as units with weights chosen by reasoning. No weight was copied and nothing was fitted to its outputs, but the trained net did inform the design.

## How it plays

A throw's score, in points:

- **Points held after the move**, counted against me.
- **A one-step plan:** the heaviest rank group left can be thrown next turn, so it is credited, weighted by the chance the opponent does not call first (read from its estimated total).
- **A charge of 5 per heavy card kept**, so pairs go out early and the hand shrinks for good.
- **A catch bonus** for holding 5 or less when the opponent is likely to call: its call becomes an Assaf.
- **Small terms:** take an open card that pairs a kept one; avoid unpaired open cards above 3; do not leave face up a rank the opponent is known to hold.

Yaniv is called when the call cannot be caught, when `assaf` is under a threshold of 0.015–0.04 that falls through the round, or when the opponent's estimated total is clearly above mine.

Match score: a move that leaves my score plus my hand on exactly 50 or 100 gets a bonus of about 50 points times the chance the opponent calls next turn, because its call would then cut my score by 50. A Yaniv call is forced when even a caught call would land on a cut.

## Scoreboard of designs

All against the card counter. "Shared" is the shared-seed test above. "Bigger" is 12,000 rounds (seed 52000) and 3,000 matches (seed 52001), used to choose between close designs because the shared test moves by about ±0.3 margin and ±3.4 points of matches. Files are in `training/yaniv/handmade_designs/`; every variant tried is in `results/handmade-4x7-logs/`.

| Design | Idea | Shared: margin, rounds, matches | Bigger: margin, rounds, matches |
|---|---|---|---|
| card counter | the rival | 0.00, 50.0%, 400–400 | 0.00, 50.0%, 1500–1500 |
| trained net | the target | +1.52, 55.4%, 490–310 (61.3%) | +1.39, 54.3%, 1816–1184 (60.5%) |
| `scorer` base | readable copy of the counter's idea | −0.12, 50.1%, 380–420 (47.5%) | −0.40, 49.2%, 1419–1581 (47.3%) |
| `scorer` | the same + match-score units | −0.12, 50.1%, 463–337 (57.9%) | −0.40, 49.2%, 1705–1295 (56.8%) |
| `caller` | simple throws + a Yaniv rule on the opponent's estimated total | +0.70, 50.6%, 416–384 (52.0%) | +0.26, 49.3%, 1623–1377 (54.1%) |
| `planner` | first-principles plan value | +1.03, 54.4%, 444–356 (55.5%) | +0.63, 53.1%, 1677–1323 (55.9%) |
| `reader` | rules read off the trained net's behaviour, 30 units | +1.33, 53.0%, 468–332 (58.5%) | +0.88, 51.4%, 1730–1270 (57.7%) |
| `reader` + score units | | +1.36, 53.0%, 495–305 (61.9%) | +0.88, 51.4%, 1849–1151 (61.6%) |
| `clone` | exact rebuild of the counter's evaluation + 4 reasoned changes, 11 layers | +1.18, 55.1%, 482–318 (60.2%) | +0.89, 54.1%, 1754–1246 (58.5%) |
| `clone` + score units | | +1.18, 55.1%, 513–287 (64.1%) | +0.89, 54.1%, 2002–998 (66.7%) |
| `merge_small` | `reader` + plan + catch bonus + both Yaniv rules + score units | +1.44, 55.2%, 521–279 (65.1%) | +1.32, 54.4%, 1931–1069 (64.4%) |
| `merge_exact` | `clone` + heavy-card charge + stricter Yaniv rule + score units, about 2,400 units | +1.48, 53.8%, 562–238 (70.2%) | +1.30, 53.1%, 2062–938 (68.7%) |
| **`closer` = `handmade.py`** | `merge_small` with the score units fixed | **+1.44, 55.2%, 549–251 (68.6%)** | **+1.32, 54.4%, 2064–936 (68.8%)** |

The `scorer` rows on the shared test were reported by the sub-agent that built it; every other number was re-run for this table.

`merge_exact` and the final net are level on the bigger test. The final net was chosen there because it is a fifth of the size and readable. On the final check `merge_exact` scored +1.21, 531/1000 rounds, 541–259 matches (67.6%), again level.

## What worked

- **Playing to the 50 and 100 cuts.** This was the largest single gain: 4 to 10 points of matches won on every base it was added to, with rounds from 0-0 unchanged. The counter never does it. Whether the trained net does was not measured.
- **A stricter Yaniv rule.** The `assaf` input assumes the opponent's hidden cards are random; they lean low, so the real catch rate is 3 to 8 times higher at the low end. The counter's own thresholds call at 18–26% real risk. Cutting at 0.015–0.04, or requiring the opponent's estimated total to be clearly higher, dropped caught calls from about 7% to 4–5%.
- **A charge per card kept.** Throwing a pair of 3s rather than a single 10 shrinks the hand for the rest of the round. The counter almost never does this (5%); the trained net does (82%).
- **Reading the opponent's estimated total as the chance it calls next turn**, and using that to weigh points held now against the plan for next turn.
- **The catch bonus:** sitting on a low hand when the opponent is about to call. Two thirds of the trained net's edge over the counter was found to be the counter calling into an Assaf.

## What didn't

- **Changing Yaniv calls by match score** (riskier calls when far behind, a penalty for passing 100): no measurable effect.
- **A flat cost on taking the open card**, a bonus for being able to call next turn on top of the plan, and a card charge that falls when the opponent is close to calling: neutral or worse.
- **Copying the trained net's call rates directly:** −0.2 a round. The call threshold is a narrow sweet spot.
- **Terms that fight each other.** The charge for taking a high open card outweighed the cut bonus and halved the cut plays until it was waived for moves that land on a cut (64.4% → 67.0% of matches). A pairing bonus double-counted the plan (+0.12 margin when reduced).

## Honest comparison with the trained net

- **Rounds from 0-0:** level. The trained net is ahead on the bigger test (+1.39 against +1.32) and on the shared seeds, behind on the final check, and even head to head. None of those gaps is outside the noise.
- **Matches:** the hand-written net is ahead by about 7 points against the counter, and that lead is entirely the match-score units. Without them the designs score 58–60% on the bigger test, level with or just under the trained net's 60.5%.
- **Against other opponents:** only the counter and the trained net were tried. The cut play pays most against an opponent that calls Yaniv on schedule; the hand-written net's own calls were caught 7.7% of the time by the trained net, against 5% by the counter.
- **Selection:** the first-round designs were tuned by hand on the shared seeds, which flatters them a little there (the trained net was also picked on those seeds). The bigger test and the final check use other seeds and agree.
- **Size and speed:** 462 hidden units against 256, in 8 layers against 2.
- **Not done:** the browser game does not load the hand-written file; it sits next to the trained one in `public/model/`.

## Reproduce

```bash
python -m training.yaniv.handmade                 # writes public/model/yaniv-net-4cards-yaniv7-handmade.json
python -m training.yaniv.handmade_eval public/model/yaniv-net-4cards-yaniv7-handmade.json            # shared seeds
python -m training.yaniv.handmade_eval public/model/yaniv-net-4cards-yaniv7-handmade.json --final    # new games
```

`handmade_eval.py` splits each test the way the trained net's gate replay did (13 game processes), so it reproduces that net's 490–310 on any machine. The design files in `handmade_designs/` save their nets to a scratch folder path that no longer exists; change `SCRATCH` in a file to run it.
