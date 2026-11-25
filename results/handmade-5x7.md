# A hand-written net for the real game (5 cards, Yaniv at 7), with a handover to the 4-card net

**Result.** A player whose weights were all written by hand, with no training, beats the card counter in the real game and wins more matches against it than the trained net does. On rounds from 0-0 it is level with the trained net or slightly behind. Handing the end of the round to the 4-card net does **not** help the new net: the new net alone is the better player.

All numbers are against the card counter unless a row says otherwise. Margin is the counter's points minus the player's, a round.

| Player | Margin a round | Rounds won | Calls caught | Matches won (95% range) |
|---|---|---|---|---|
| **Held-out check, run once** (20,000 rounds seed 77000; 4,000 matches seed 77001) | | | | |
| card counter itself | 0.00 | 50.0% | 7.3% | 2000–2000 (50.0%) |
| trained net `yaniv-net.json` | +1.67 | 54.8% | 4.2% | 2544–1456 (63.6%, 62.1%–65.1%) |
| 4-card hand-written net, playing the whole game unchanged | +1.31 | 54.2% | 4.6% | 2825–1175 (70.6%, 69.2%–72.0%) |
| **new 5-card net alone** `yaniv-net-5cards-handmade.json` | **+1.49** | **55.4%** | 4.6% | **2923–1077 (73.1%, 71.7%–74.4%)** |
| **new net, then the 4-card net** `yaniv-net-handmade.json` (the browser's file) | +1.42 | 55.1% | 4.6% | 2871–1129 (71.8%, 70.4%–73.1%) |
| trained net, then the 4-card hand-written net | +1.37 | 54.8% | 4.7% | 2784–1216 (69.6%, 68.2%–71.0%) |
| **The trained net's own final check** (1,000 rounds seed 31337; 400 matches seed 31338) | | | | |
| trained net (reproduces its published 557/1000 and 256–144) | +1.75 | 55.7% | 4.2% | 256–144 (64.0%) |
| 4-card hand-written net, whole game | +1.03 | 53.8% | 5.0% | 274–126 (68.5%) |
| new net alone | +1.48 | 55.5% | 4.5% | 290–110 (72.5%) |
| new net, then the 4-card net | +1.50 | 55.3% | 4.5% | 282–118 (70.5%) |
| **Shared seeds: the trained net's gate replay** (2,000 rounds seed 41000; 400 matches seed 41001) | | | | |
| trained net | +1.60 | 54.5% | 4.8% | 256–144 (64.0%) |
| 4-card hand-written net, whole game | +1.52 | 54.8% | 4.6% | 293–107 (73.2%) |
| new net alone | +1.67 | 56.5% | 4.8% | 308–92 (77.0%) |
| new net, then the 4-card net | +1.60 | 56.0% | 5.0% | 310–90 (77.5%) |

Head to head (12,000 rounds seed 78000; 3,000 matches seed 78001):

| Pairing | Margin a round | Rounds won | Matches won (95% range) |
|---|---|---|---|
| new net with handover against the trained net | 0.00 | 51.0% | 1845–1155 (61.5%, 59.7%–63.2%) |
| new net alone against the trained net | +0.11 | 51.6% | 1893–1107 (63.1%, 61.4%–64.8%) |
| 4-card net against the trained net | −0.16 | 49.9% | 1791–1209 (59.7%, 57.9%–61.4%) |
| new net with handover against the new net alone ("switch vs main alone") | −0.07 | 49.4% | 1428–1572 (47.6%, 45.8%–49.4%) |

## The three questions

- **Does the hand-written real-game player beat the card counter?** Yes: about +1.4 to +1.5 points a round and 72–73% of matches on the held-out check.
- **Does it beat the trained net?** On matches, yes: 73.1% (alone) and 71.8% (with the handover) against 63.6% versus the counter, and 61–63% head to head. On rounds from 0-0, no: against the counter the trained net is ahead on margin (+1.67 against +1.49 / +1.42 on the held-out check), while the hand-written player wins slightly more rounds (55.4% against 54.8%); head to head the rounds are even. The match lead is the match-score play, which the trained net does not have.
- **Does handing the end of the round to the 4-card net help?**
  - **For the new hand-written net: no.** With the handover it scores 0.07 less margin and 1.3 points fewer matches on the held-out check, and loses head to head to itself without the handover (47.6% of matches, range 45.8%–49.4%). The new net's rules also apply after both hands reach 4 cards, and they play that part of the real game slightly better than the 4-card net does.
  - **For the trained net: yes on matches, no on rounds.** The handover lifts it from 63.6% to 69.6% of matches (it gains the 4-card net's match-score play) and costs it 0.30 margin a round.
  - **Is a new net needed at all?** The 4-card net unchanged already beats the trained net on matches. The new net adds about +0.2 margin a round and 2.5 points of matches over it (held-out: +1.49 against +1.31, 73.1% against 70.6%).

## How often the handover happens

Both hands reach 4 cards or fewer in about 81% of rounds, at a median of 6 throws, and about 48% of the player's decisions come after it (`measure.md` part d, `handover.md` part 2). About 16% of Yaniv calls are made while a hand still holds 5 cards. Other rules: "my hand ≤ 4" 91% of rounds / 64% of decisions; "opponent ≤ 4" 90% / 64%; "either" 99% / 80%; "both ≤ 3" 34% / 14%.

## What "by hand" meant, and where it comes close to the line

`training/yaniv/handmade5.py` builds the net from named ReLU units: 558 hidden units in 8 layers, shape [125, 218, 98, 66, 35, 33, 53, 29, 26, 1]. No gradient step, no fitting to moves or outcomes, and no automated search over weights was used.

- **Reasoned units, measured tables.** Constants are rounded real-game measurements (how often the counter calls next turn by its card count and estimated total; how often it ends the round within two turns) or reasoned numbers, each commented in the code.
- **Measurement of whole designs.** Roughly 80 to 100 variants and move comparisons were measured across five design files and the handover study, each a stated change with its result logged in `results/handmade-5x7-logs/`. That is a person's trial and error, not an optimiser, but it is many looks at the same opponent.
- **Playing both moves out.** Designers compared two candidate moves by playing the round to the end both ways on the same hidden cards. This is testing a stated rule, not fitting, but it is a finer instrument than whole-game tests and it steered which rules were kept.
- **Watching the trained net.** Three designers looked at where the trained net's moves differ from the 4-card net's and tested each kind of difference. One rule came out of that ("throw a pair of 2s before a high single early in the round"). No weight was copied and nothing was fitted to its outputs. The shipped design (`counts`) does not contain that rule; whether its exposure weights produce the same play in those positions was not checked.
- **Selection.** The shipped design was chosen on the 12,000-round test and a 30,000-round run on other seeds, where it led; on the held-out check it is still ahead of the 4-card net but by less than on the selection tests (+1.49 against +1.78 and +1.64), so part of that lead was selection luck.

## How the new net plays

It is the 4-card net's valuation with its tables set by the opponent's card count (5, 4, 3, 1–2):

- **Exposure of heavy points.** Heavy cards outside my best rank group are weighed 0.15 / 0.35 / 0.6 / 0.85 by the opponent's card count: they only cost me if the round ends before I can throw them, which is unlikely against 5 cards.
- **A charge for throwing a "live" rank.** 1 / 1 / 0.5 / 0 per unseen copy of the rank thrown. The counter takes a thrown high card 87% of the time when it holds that rank (4% otherwise) and throws the pair.
- **Unchanged from the 4-card net:** the one-step plan, the charge of 5 per heavy card kept, the catch bonus, the Yaniv rule and the match-score units.

## Scoreboard of designs

"Working" is 6,000 rounds seed 61000 and 1,500 matches seed 61001, the designers' test (about ±0.2 margin, ±2.5 points of matches). "Bigger" is 12,000 rounds seed 52000 and 3,000 matches seed 52001, re-run by the lead. Each design is listed with the handover to the 4-card net unless it says "alone". Files are in `training/yaniv/handmade5_designs/`; logs in `results/handmade-5x7-logs/`.

| Design | Idea | Working: margin, matches | Bigger: margin, rounds, matches |
|---|---|---|---|
| card counter | the rival | 0.00, 50.0% | 0.00, 50.0%, 50.0% |
| trained net | the target | +1.92, 63.2% | +1.70, 55.2%, 63.3% |
| trained net with handover | | +1.69, 67.6% | +1.55, 55.5%, 70.4% |
| 4-card net, whole game | the bar | +1.49, 68.6% | +1.39, 54.7%, 70.4% |
| `shaper` | +6 for throwing a pair of 2s and +2 for any set, at 5 against 5 | +1.61, 69.4% | not run |
| `watcher` | a 2 counts as a heavy card and the card charge is 8 while I hold 5 and the opponent 4–5 | +1.63, 70.7% | +1.63, 55.9%, 71.1% |
| `port` | every 4-card constant re-measured by opponent card count; two rules against a 5-card opponent survive | +1.65, 70.8% | +1.60, 55.8%, 71.0% |
| `port` alone | | +1.66, 70.5% | +1.58, 55.7%, 71.1% |
| `merge` | the 4-card net plus `port`'s two rules, 477 units | +1.66, 70.9% | +1.59, 55.8%, 70.9% |
| `counts` | tables by opponent card count: exposure weights and the live-rank charge | +1.64, 70.8% | +1.70, 56.2%, 70.9% |
| **`counts` alone = `handmade5.py`** | | **+1.71, 70.4%** | **+1.78, 56.8%, 71.5%** |

A second selection run (30,000 rounds, seed 53000, rounds only): `counts` alone +1.64 (56.0%), `counts` with handover +1.52, `merge` with handover +1.55, trained net +1.59, 4-card net +1.34. Head to head, `counts` alone won 51.1% of 12,000 rounds and 50.6% of 30,000 against `merge`.

## What worked

- **The 4-card net carries over almost whole.** Unchanged, it scores +1.3 to +1.5 a round and 70% of matches in the real game. Its table for the chance the counter calls next turn fits the real game at every card count; re-measuring it changed 30 of 34,000 decisions.
- **The match-score units pay as much as in the 4-card game:** 68.6% of matches with them against 61.3% without, at the same margin (`measure.md` part g). They produce 0.48 score cuts a match against the counter's 0.10.
- **Shrinking first at 5 against 5.** All four designers found the same gap by different routes: the 4-card net keeps a pair of 2s and throws a high single; throwing the 2s is worth +3.6 to +4.7 points each time it comes up.
- **Not throwing a rank the opponent may pair:** about +0.1 a round.
- **Exposure weights by opponent card count** (`counts`): small, but it is the one change that also improves play after both hands reach 4 cards.

## What did not

- **The handover, for a hand-written main net** (see above).
- **Re-measuring the 4-card constants by card count:** the catch threshold, the card charge (3, 7 or 8 against 4–5 cards), the callable bonus and the open-card charges all measured within ±0.07.
- **Other handover points.** Both / mine / opp / either at 4 cards gave +1.65 to +1.69 on the working test with a hand-written main net; the round-1 nets only differ from the 4-card net where someone holds 5 cards, so the rule hardly matters.
- **Five-card shaping ideas:** keeping a pair to wait for a third (−1.1 to −1.7 each time), a two-step plan (no position where it changed a move for the better), a bonus for being able to call next turn, a bonus for throwing any set.
- **Adding the designs' rules together:** the pair-of-2s rules, the charge of 8 and the exposure weights are the same gain found several ways and do not add up.

## Honest comparison with the trained net

- **Rounds from 0-0:** the trained net is ahead on margin against the counter on every large test (+1.67 against +1.49 held-out; +1.70 against +1.78 on the test used for selection, which flatters the hand-written net). Rounds won favour the hand-written net by about half a point. Head to head they are even.
- **Where the trained net is still better:** after both hands reach 4 cards, by about +0.13 a round. About half of that is one kind of move: where the 4-card net calls Yaniv with some catch risk, the trained net throws its highest single and takes a lower open card (+3.8 points each time, 0.017 times a round; `handover.md` part 3). That rule was not added; the 4-card net was left as it is.
- **Matches:** the hand-written player leads by 8–10 points against the counter and wins 61–63% head to head. All of that is match-score play.
- **Opponents:** only the counter and the trained net were tried.
- **Size:** 558 hidden units in 8 layers against the trained net's three experts of 256 units in 2 layers. The combined file is 0.68 MB.

## The one-file player

`training/yaniv/handmade5_pack.py` packs the main net, the 4-card net and a hand-written decider ([96, 2, 2]) in the decider-plus-experts format the browser loads. The decider's two outputs are always at least 2,000 apart, so the softmax weights are exactly 1 and 0 in float32 and float64. The two nets are widened with zero units and deepened with pass-through layers to one shape.

- Against the switch player with both nets in the padded shape: 0 of 20,005 decisions differ.
- Against the nets in their own shapes: 6 of 20,005 moves differ, near ties decided by the last float32 bit of numpy's matrix product (largest score difference 6e-5). On the shared seeds the file and the unpadded switch differ by one match (310–90 against 311–89).
- In the browser (`net.ts`), the packed file scores every row the same as the chosen net alone (600 positions).
- `/?game=5x7-handmade` ("Standard: hand-written net") was checked in Chromium: the file loads, the net plays, no console errors.

A wart found on the way: one Yaniv unit of the 4-card net also fires on throw rows when my hand is above about 35 points, adding the same amount to every throw. It changes no decision (0 of 20,457 checked) and was left alone.

Also noted: `public/model/yaniv-net-4cards-yaniv7.json` holds exactly the same weights as `yaniv-net.json` (that run shipped its start net), so there is only one trained net and a trained-to-trained handover changes nothing.

## Reproduce

```bash
python -m training.yaniv.handmade5        # writes yaniv-net-5cards-handmade.json and yaniv-net-handmade.json
E="python -m training.yaniv.handmade_eval --game 5x7"
$E --matches 400 public/model/yaniv-net-handmade.json     # shared seeds
$E --final public/model/yaniv-net-handmade.json           # the trained net's final check
$E --held-out public/model/yaniv-net-handmade.json        # the held-out check
$E NET.json "switch:NET.json:public/model/yaniv-net-4cards-yaniv7-handmade.json:opp"   # another handover rule
python -m training.yaniv.handmade5_pack public/model/yaniv-net-5cards-handmade.json --out OUT.json --check 20000
```

`handmade_eval.py` splits each real-game test as 11 processes would, which reproduces the trained net's published final check on any machine. The design files in `handmade5_designs/` default to a scratch path that no longer exists; pass an output path. Raw outputs of every run above are in `results/handmade-5x7-logs/` (`baselines-*.txt`, `round1-dev.txt`, `round2-*.txt`, `final-*.txt`).
