# counter2: a second, stronger card counter

**Result.** `training/yaniv/heuristic2.py` (`bot_spec("counter2")`) is a rule-based player that beats the old counter by about 2 points a round and wins 74% of matches against it. In the real game it also beats the trained net and both hand-written nets on rounds. On matches it beats the trained net clearly and is level with the hand-written nets. In the 4-card game it is weaker: clearly ahead of the old counter, level with the nets.

It acts on `engine.View` only, remembers nothing, and uses no random numbers. Nothing was trained or fitted; the constants are reasoned or read from measured tables, and every version measured is in `results/counter2-logs/versions.md`.

## Final check, on seeds not used while developing

Run once, after the rules were frozen (commit `counter2: final rules`). Real game: 20,000 mirrored rounds from 0-0 (seed 99000) and 4,000 mirrored matches (seed 99001). Margin is the opponent's points minus the player's, a round; ± and the brackets are 95% ranges. Raw output: `counter2-logs/heldout-5cards.txt`.

| Real game (5 cards, Yaniv at 7) | Margin a round | Rounds won | Own calls caught | Matches won |
|---|---|---|---|---|
| **counter2** against the old counter | **+2.02 ±0.20** | 58.0% (57.3–58.6) | 3.6% | **2964–1036, 74.1% (72.7–75.4)** |
| **counter2** against the trained net `yaniv-net.json` | **+0.54 ±0.18** | 53.7% (53.0–54.4) | 5.5% | **2623–1377, 65.6% (64.1–67.0)** |
| **counter2** against `yaniv-net-5cards-handmade.json` | **+0.31 ±0.19** | 52.2% (51.5–52.9) | 5.0% | 2026–1974, 50.6% (49.1–52.2) |
| **counter2** against `yaniv-net-handmade.json` | **+0.39 ±0.19** | 52.7% (52.0–53.4) | 5.0% | 2076–1924, 51.9% (50.4–53.4) |
| old counter against itself | 0.00 | 50.0% | 7.2% | 2000–2000, 50.0% |
| old counter against the trained net | −1.61 ±0.19 | 45.3% (44.6–46.0) | 11.4% | 1412–2588, 35.3% (33.8–36.8) |
| old counter against `yaniv-net-5cards-handmade.json` | −1.60 ±0.19 | 44.2% (43.5–44.9) | 10.2% | 1101–2899, 27.5% (26.2–28.9) |
| old counter against `yaniv-net-handmade.json` | −1.56 ±0.19 | 44.6% (43.9–45.3) | 10.4% | 1174–2826, 29.3% (28.0–30.8) |

Reading it as a measuring stick: against the old counter the nets score +1.6 a round; against counter2 they score −0.3 to −0.5. A net's matches fall from 65–72% against the old counter to 34% (trained) or 48–49% (hand-written) against counter2.

The old counter catches 3.6% of counter2's calls; counter2 catches 10.6% of the old counter's.

### The 4-card game (YANIV_HAND_SIZE=4, YANIV_CALL_AT=7)

Same rules, same constants, same seeds and sizes (`counter2-logs/heldout-4cards.txt`).

| 4 cards, Yaniv at 7 | Margin a round | Rounds won | Matches won |
|---|---|---|---|
| **counter2** against the old counter | **+1.35 ±0.19** | 56.1% (55.4–56.8) | **2774–1226, 69.3% (67.9–70.8)** |
| **counter2** against `yaniv-net-4cards-yaniv7-handmade.json` | +0.16 ±0.17 | 51.9% (51.2–52.6) | 2003–1997, 50.1% (48.5–51.6) |
| **counter2** against `yaniv-net-4cards-yaniv7.json` (trained) | +0.07 ±0.16 | 51.3% (50.7–52.0) | 2528–1472, 63.2% (61.7–64.7) |
| old counter against the 4-card hand-written net | −1.32 ±0.17 | 45.4% (44.7–46.1) | 1248–2752, 31.2% (29.8–32.7) |
| old counter against the trained net | −1.26 ±0.17 | 45.8% (45.1–46.5) | 1516–2484, 37.9% (36.4–39.4) |

**It is weaker there.** It beats the old counter by about as much as the nets do (+1.35 against their +1.3) and is level with both nets on rounds (the ranges include zero). It still serves as a second opponent in that game, but it is not a harder one than the nets. Nothing was tuned on the 4-card game: the call table and the clock were measured in the real game.

## How it plays

1. **The opponent's total.** An exact distribution of its points up to 12, over every way to fill its hidden cards from the unseen ones. Hidden cards lean low (`features.HIDDEN_BETA`), and the cards it kept are capped by the single it last threw. The modelled catch risk matched the true one bin by bin without correction (0.045 believed / 0.042 true, 0.123 / 0.131, 0.248 / 0.290).
2. **Its call.** P(it calls next turn) = P(it holds t) × P(it calls on t | the cards I hold). The second factor is a measured table and the biggest single finding here: against 4–5 cards every opponent calls whenever it may; against 2 cards the old counter calls 56% of the time on 4 points and 6% on 6; against 1 card almost never above 2.
3. **A throw** is the exact expectation over the card drawn of: its call next turn (my points lost, or an Assaf if I hold no more than it), otherwise my next turn, the better of calling and throwing on.
4. **Throwing on** is read from a table: the game played alone against a clock, solved exactly for all 11,494 hands of up to five cards. This is the look-ahead of many moves; it replaces a hand-set cost per card.
5. **Feeding.** Throwing a rank the opponent holds costs (2.4 + 0.5 × points) × P(it holds it); keeping a rank it is likely to throw next earns the same sum × that chance.
6. **Yaniv** when the call cannot be caught, when even a caught call lands on a cut, or when the call is worth the best throw minus 2.5.
7. **Match score.** A hand that lands my score on exactly 50 or 100 is credited 50 × the chance the opponent calls.

Speed: 244 microseconds a move against the old counter's 527 (`counter2-logs/timing.txt`, same deals), plus about 0.3 seconds once per process to build the table. The table is built with NumPy.

## Which earlier findings carried over

| Finding from the hand-written nets | Carried over? |
|---|---|
| The uniform catch chance understates the risk; call tighter | Yes, in another form. With the leaning, capped model the risk is calibrated, and the rule is "call if worth the best throw", not a threshold. Own calls caught fell from 7.2% to 3.6%. |
| Sit on a low hand to catch the opponent's call | Yes. It is part of every throw's value, not a bonus. Counter2 catches 10.6% of the old counter's calls. |
| Hold exactly the points that land on 50 or 100 | Yes: 0.53 cuts a match against the old counter's 0.16 (`cuts-a-match.txt`). |
| Call when even a caught call lands on a cut | Yes, kept as a rule. |
| Throw a low pair before a higher single | Partly. Pairs of 2s and up, yes; pairs of aces and jokers, no: played out, keeping them and throwing a single 6–10 was 2 points better each time. |
| Take unpaired open cards only up to 3 | Not as a rule. The table decides card by card; played out, its choices beat the nets' on open cards of 3–5. |
| Do not feed a rank the opponent may hold | Yes, and stronger than expected: removing the charge cost 0.37 a round, and making it grow with the card's points gained 0.49 a round (played out). |
| Weigh points now against points later by the opponent's estimated total | Yes for the next turn (the exact call chance). For later turns, reading it from the model instead of the card count changed nothing (−0.01 / +0.01 a round). |
| Changing calls by match score has no effect | Confirmed. Valuing every outcome by the chance of winning the match gave 74.8 / 66.5 / 51.1% of 3,000 matches against 74.4 / 66.9 / 52.5% without it. Not kept. |

New here: the call table by my card count, the clock table, and the pairing bonus for a kept rank.

## What did not work

- **Re-setting the card costs to match the true value of a position:** −0.17 a round. The relative costs mattered more than the scale.
- **A slower clock, and a clock started at the round's own turn number:** within noise or worse.
- **The match-chance table** (above).
- **A charge on my own call for handing the opponent a cut.** The hand-written net gets 1.07 cuts a match from counter2, mostly from counter2's calls. Charging for that made matches worse (47.1%, and 38.2% at three times the charge, against 52.5%). This is why counter2 is only level with the hand-written nets on matches while ahead on rounds, and it is the clearest thing left to improve.

## Honest limits

- **Opponents.** Developed against four opponents, two of which share a design. The call table is an average of three players' habits; an opponent that calls very differently will be misread.
- **Many looks at the dev seeds.** About 20 versions and as many played-out comparisons used seeds 91000/91001 and 930000+. The final check on other seeds came in a little lower on every row (for example +2.02 against +2.10, +0.31 against +0.56 against the 5-card hand-written net), so part of the dev numbers was selection luck.
- **Measured tables.** The call table, the clock's first steps and the call edge come from games of counter2's own earlier versions against these opponents. That is counting and playing moves out, not fitting, but it is tuned to this field.
- **Played-out comparisons are first-order.** They change one move and let counter2 play on, so they can miss gains that need several changed moves.
- **No memory.** It cannot tell whether the opponent drew from the deck or took the open card last turn, and assumes the deck three times in four.

## Reproduce

```bash
E="python -m training.yaniv.heuristic2_eval"
N="public/model/yaniv-net.json public/model/yaniv-net-5cards-handmade.json public/model/yaniv-net-handmade.json"
$E counter2 counter --against counter $N --rounds 20000 --matches 4000 --seed-rounds 99000 --seed-matches 99001
$E counter2 counter --hand 4 --against counter public/model/yaniv-net-4cards-yaniv7-handmade.json \
   public/model/yaniv-net-4cards-yaniv7.json --rounds 20000 --matches 4000 --seed-rounds 99000 --seed-matches 99001
$E counter2 --against counter $N           # the dev test (seeds 91000 / 91001)
$E counter counter2 --time                 # microseconds a move
python -m training.yaniv.heuristic2_study calib                    # belief against truth
python -m training.yaniv.heuristic2_study calls --against counter  # call or throw, both played out
python -m training.yaniv.heuristic2_study differ --alt public/model/yaniv-net.json --detail
```

`heuristic2_eval.py` cuts each test into fixed pieces, so the deals do not depend on the number of processes. Its seeds are not the ones `handmade_eval.py` uses, so rows here are not the same deals as in `handmade-5x7.md`. Variants in the log that name a removed constant (the card costs, the match table) need the commit before `counter2: final rules`.
