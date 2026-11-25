# Brief for the team: a hand-written net for the real Yaniv game (5 cards, Yaniv at 7)

Repo: `/home/user/yaniv-master` (branch `claude/handmade-net-5x7`). You are one of several agents working at
the same time. **Do not run git commands that change anything (no commit, checkout, stash, reset). Only touch
the files your task names as yours.** The lead commits.

## Goal
A player for the real game (5 cards each, Yaniv at 7 or less) whose net weights are written by hand:
1. a new hand-written net that plays while either player still holds 5 cards;
2. a handover to the existing hand-written 4-card net (`public/model/yaniv-net-4cards-yaniv7-handmade.json`,
   built by `training/yaniv/handmade.py`) once both hands hold 4 cards or fewer (a hand never grows).

## "Without training" - the hard rule
No gradient descent; no fitting a net to labelled moves or game outcomes (never call train_dataset,
train_value, main.py or the search / self-play / genetic recipes); **no automated weight search: no genetic
algorithm, hill climbing, random search, or scripted sweeps/loops over coefficients.** Weights are chosen by
reasoning about what each unit should detect and how much it matters. Playing evaluation games to measure a
design and then revising it by hand is testing, not training. Hand-picking a handful of coefficients after
looking at results is fine; a loop that tunes them is not. Measuring game statistics (how often the counter
calls at a given estimate, how often a call is caught) and writing the rounded numbers in by hand is fine.
Never copy weights from a trained net. **Log every variant you measure, with its reason and its result**, in
your log file, including the ones that failed.

## Read first
- `results/handmade-4x7.md` and `results/handmade-4x7-logs/*.md`: the previous experiment (4-card net). Which
  ideas worked, which did not, how each designer worked.
- `training/yaniv/handmade.py` (the 4-card net, named units), `training/yaniv/handmade_kit.py` (`Kit`, `INPUTS`:
  names and scales of the 125 inputs), `training/yaniv/handmade_designs/` (the eight earlier designs).
- `training/yaniv/engine.py` (rules), `features.py` (the 125 inputs), `heuristic.py` (the card counter = the
  measuring stick; note `hand_badness` weights by opponent card count: 0.08 / 0.22 / 0.40 / 0.62).
- `training/yaniv/hybrid.py`, `parallel.py` (`switch_spec(main, endgame, at, when)`), `handmade_eval.py`.

## Gotchas
- Rules are read from env vars when `training.yaniv.engine` is first imported. The real game is the default:
  leave `YANIV_HAND_SIZE` / `YANIV_CALL_AT` unset in your own scripts. Library modules must never set them on
  import (it leaks into the unit tests).
- Scripts that start worker processes need `if __name__ == "__main__":`.
- `Workers` splits a test by its process count, so results only compare when the split is the same;
  `handmade_eval.py` fixes the split (`--split-as`, default 11 for the real game), whatever `--workers` is.
- The machine has 4 cores shared by about 6 agents. Use `--workers 2`. Do not start more than one evaluation
  at a time. Never leave background processes running when you finish.
- Scratch files (candidate nets, scripts you do not want kept) go in the scratch dir your task names, not in
  the repo and not in /tmp.

## The shared test (real game)
```bash
python -m training.yaniv.handmade_eval --game 5x7 --workers 2 NET.json                 # shared: 2,000 rounds seed 41000, 800 matches seed 41001 (use --matches 400: the trained net's gate replay size)
python -m training.yaniv.handmade_eval --game 5x7 --workers 2 --work NET.json          # working test: 6,000 rounds seed 61000, 1,500 matches seed 61001
python -m training.yaniv.handmade_eval --game 5x7 --workers 2 --work --matches 0 NET.json   # rounds only (4x quicker)
python -m training.yaniv.handmade_eval --game 5x7 --workers 2 --work "switch:NET.json:public/model/yaniv-net-4cards-yaniv7-handmade.json"   # with the handover (also :mine, :opp, :either)
python -m training.yaniv.handmade_eval --game 5x7 --workers 2 --work A.json --against B.json   # head to head
```
A player is a net file, `counter`, or `switch:MAIN:ENDGAME[:when[:at]]`. Margin = counter's points minus the
player's, per round.

**Noise is large in the real game.** 1,000 rounds moves by about +-0.5 margin, 2,000 rounds by +-0.35, 400
matches by +-5 points. **Decide on the working test (`--work`)**: about +-0.2 margin, +-2.5 points of matches.
A difference inside that is not a finding; say so in your log. The lead keeps two more tests that you must NOT
run: `--dev` (seeds 52000/52001, for choosing between finished designs) and the held-out final
(`--final`, `--held-out`).

## Baselines (real game, against the card counter)
Shared seeds (2,000 rounds seed 41000; 400 matches seed 41001):
| player | margin | rounds | caught | matches |
|---|---|---|---|---|
| card counter | 0.00 | 50.0% | 7.1% | 200-200 |
| trained net `public/model/yaniv-net.json` (a decider + 3 expert nets) | +1.60 | 54.5% | 4.8% | 256-144 (64.0%) |
| 4-card hand-written net, playing the whole real game unchanged | +1.52 | 54.8% | 4.6% | 293-107 (73.2%) |
| trained net with handover to the 4-card hand-written net (both <= 4) | +1.36 | 54.8% | 4.8% | 281-119 (70.2%) |

The working-test and dev numbers for these four are appended to this file / `baselines-dev.txt` when ready.
Note the 4-card hand-written net, unchanged, already beats the trained net on matches (its match-score units).
**The bar for a new 5-card net is that 4-card net playing the whole game**: about +1.5 margin and 73% of
matches on the shared seeds. A new net that does not beat it (alone, or with the handover) is not needed -
that is a legitimate finding too, but try to beat it.
(Fact: `public/model/yaniv-net-4cards-yaniv7.json`, the "trained 4-card net", holds exactly the same weights as
`yaniv-net.json`; its run shipped generation 0, the start net. So there is only one trained net.)

## What differs with 5 cards (starting points, not conclusions)
- A player holding 5 cards can almost never call (needs <= 7 points in 5 cards), so while the opponent has 5
  cards my held points barely matter, only what my hand can become; the counter weights points 0.08 then.
- More pairs and triples; longer rounds; shrinking the hand (sets) matters more; throwing a pair costs a card
  count the opponent sees.
- The 4-card net's constants were read off 4-card measurements: P(opponent calls next turn | opp_est) =
  0.55/0.30/0.18/0.08/0.04/0.01 at cuts 5/8/11/14/17 with no dependence on its card count; catch threshold 5;
  card charge 5 (-2 when the opponent has 1-2 cards); Yaniv thresholds on `assaf`. These need re-measuring in
  the real game, by opponent card count.
- The match-score units (aim at exactly 50 or 100) should carry over.
- `results/handmade-5x7-logs/measure.md` is being written by the measuring agent (tables of real-game
  statistics). Check whether it exists when you start and again later; use it, and measure anything else
  you need yourself.

## Baselines on the working test (--work: 6,000 rounds seed 61000; 1,500 matches seed 61001)
```
public/model/yaniv-net-4cards-yaniv7-handmade.json: margin +1.49 a round, rounds won 3315/6000 (55.2%), Yaniv calls 3184 of which caught 142 (4.5%); matches 1029-471 (68.6%, 95% range 66.2%-70.9%)  [235s]
public/model/yaniv-net.json: margin +1.92 a round, rounds won 3345/6000 (55.8%), Yaniv calls 3160 of which caught 124 (3.9%); matches 948-552 (63.2%, 95% range 60.7%-65.6%)  [421s]
switch:public/model/yaniv-net.json:public/model/yaniv-net-4cards-yaniv7-handmade.json: margin +1.69 a round, rounds won 3355/6000 (55.9%), Yaniv calls 3227 of which caught 140 (4.3%); matches 1014-486 (67.6%, 95% range 65.2%-69.9%)  [605s]
```
The first line (the 4-card hand-written net playing the whole game) is the bar.
