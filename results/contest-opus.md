# Team opus: one hand-written net for the real game (5 cards, Yaniv at 7)

**The net.** `training/yaniv/contest_opus.py` → `public/model/yaniv-net-contest-opus.json`. It is one plain
`network.MLP` of shape **[125, 264, 105, 66, 35, 33, 53, 29, 26, 1]**: **611 hidden units** (the limit is 1,020),
ReLU hidden layers and a linear output. No decider and no handover. It starts from `handmade5.py` ("hm5", 558 units)
and adds three blocks, each with its own reason and measurement:

| block | file | units | what it does |
|---|---|---|---|
| rule S, the swap | `contest_opus_endgame.py` | 12 | Fires when Yaniv is legal, the opponent holds 3–4 cards, the uniform catch chance is ≥ 0.005 and the open card is 2+ points under my highest card. Then I throw that card and take the open card instead of calling. |
| TRIPLE | `contest_opus_mine.py` | 10 | Fires when I hold a pair of R (R worth 4+) and the open card is an R. Then I keep the pair, throw another card worth 4+, take the open R and throw all three next turn. |
| the cut veto | `contest_opus_match.py` | 31 | Never make a winning Yaniv call that could drop the opponent's score exactly onto 50 or 100 (it would fall to 0 / 50). It is judged from the opponent's known cards when 0, 1 or 2 of its cards are hidden. |

Nothing was trained, fitted or searched. Every weight is written by reasoning, and the thresholds are hand-picked
from measurements (see "How it was made" below).

## Final check (fresh seeds, run once: 12,000 rounds seed 777001, 3,000 matches seed 777002)

Margin is the opponent's points minus the net's, a round; both nets play the same deals (mirrored pairs).
Log: `results/contest-opus-logs/final-s777.txt`.

| opponent | opus: margin, rounds won, matches won (95% range) | hm5 on the same deals |
|---|---|---|
| **hm5 (head to head)** | **+0.14, 50.3%, 1739–1261 = 58.0% (56.2–59.7%)** | 0.00, 50.0%, 50.0% |
| trained net `yaniv-net.json` | +0.35, 52.1%, 63.6% (61.9–65.3%) | +0.13, 51.4%, 63.9% |
| card counter | +2.02, 57.3%, 71.8% (70.2–73.4%) | +1.80, 56.5%, 71.0% |
| 4-card hand-written net (hm4) | +0.53, 52.0%, 59.5% (57.7–61.2%) | +0.39, 51.6%, 52.1% |

An earlier check (seed 400: 8,000 rounds, 1,600 matches; `v3-s400.txt`) found the same picture:

| opponent | opus: margin, matches won | hm5 |
|---|---|---|
| hm5 | +0.18, 58.7% | 0.00, 50.0% |
| trained net | +0.29, 62.2% | +0.11, 63.3% |
| card counter | +1.81, 71.8% | +1.69, 71.7% |

Baselines (seed 300; `baseline-s300.txt`):

| player | vs counter: margin, matches | vs trained net: margin, matches |
|---|---|---|
| hm5 | +1.61, 72.7% | +0.11, 62.9% |
| trained net | +1.74, 64.3% | 0.00, 50.0% |
| hm4 | +1.37, 70.9% | −0.31, 59.4% |

The trained net loses to hm5 head to head: −0.11 margin a round and 37.1% of matches.

## The owner's strategy, tested first: keep a high card for its twin (it does not pay, because hm5 already plays it)

Log: `results/contest-opus-logs/twin.md`. Helper: `training/yaniv/contest_opus_twin.py`.

**Definition.**
- I hold 5 cards and the opponent holds 4 or more.
- I hold two or more ranks once each, all worth 8 or more (7 or more also measured).
- The open card pairs none of my cards.
- This comes up **0.56 times a round** (0.74 with the 7+ cut).

**What hm5 already does here** (census, no alternative move):
- Where the unseen counts of my high singles differ, hm5 throws the rank with fewer unseen copies 90% of the time. That is its `LIVE` charge.
- The twin of a kept rank appears within the opponent's next two turns in 27–28% of these positions.
- hm5 takes it 95% of the time and throws the pair the next turn 99% of the time.
- So the owner's plan is already in the net, end to end.

**Play-outs** (8,000 rounds, seed 7101, hero hm5). Both moves are played to the end on the same hidden cards. Gain is the
rule's move minus hm5's, per occurrence ± standard error:

| rule | occurrences a round | vs counter | vs trained net |
|---|---|---|---|
| throw the high single with the fewest unseen copies (≥8) | 0.22 | −1.00 ± 0.39 | −0.69 ± 0.36 |
| same, ≥7 | 0.28 | −1.15 ± 0.35 | −0.98 ± 0.32 |
| throw the most-unseen one (the opposite) | 0.31 | −1.43 ± 0.32 | −0.47 ± 0.30 |
| always throw the highest single (≥8) | 0.16 | −2.37 ± 0.47 | −1.10 ± 0.43 |
| keep the highest, throw a lower high single (two seeds pooled) | 0.27 | +0.18 ± 0.25 | −0.26 ± 0.23 |
| take the open twin where hm5 does not | 0.03 | −2.04 ± 1.24 | −4.32 ± 0.91 |

What the sub-cases show:
- Keeping the live rank is worth about +0.8 a time, and hm5 already does it.
- The disagreements that remain are arbitrary tie-breaks between equal cards: +1.18 ± 0.69 and +0.40 ± 0.64.
- Where a rule overrides a set throw it loses: −2 to −3 a time. Where it overrides a real difference in unseen copies it loses about 6.
- The "keep the highest" version flips sign between seeds.
- Verdict: **not added.**

## What else was tried

Every probe is a paired play-out (`contest_opus_playout.py`) or a whole-game test (`contest_opus_eval.py`). Each
designer's log lists every variant, failures included.

**Kept** (numbers per occurrence ± se; the hero is hm5):
- **Rule S** (`endgame.md`). Its gain repeated on three seeds against both opponents:

  | seed | occurrences a round | vs counter | vs trained net |
  |---|---|---|---|
  | 7101 | 0.007 | +5.13 ± 1.37 | +8.89 ± 1.68 |
  | 7102 (48,000 rounds) | 0.007 | +4.77 ± 1.18 | +6.72 ± 1.17 |
  | built net, 7104 | 0.007 | +5.30 ± 1.21 | +3.49 ± 1.35 |

  That is about +0.02 to +0.035 a round, too small for a whole-game test to show on its own. A version that also swaps for a 1-point saving paid against the counter (+3.2) but not against the trained net (−0.6), so it was dropped.
- **TRIPLE** (`mine.md`). It was found where both the trained net's move and the counter's move beat hm5's against both opponents:
  - As a rule (seed 5300): +2.72 ± 0.86 against the counter and +2.99 ± 0.96 against the trained net, 0.045 times a round.
  - The built net's changed decisions (seed 5600): +3.40 ± 0.85 and +2.90 ± 0.89.
  - Whole game against hm5 on three seeds: +0.18 a round against the counter and against the trained net, and +0.09 head to head. All 9 of 9 pairs were positive.
- **The cut veto** (`match.md`).
  - Upper bound, knowing the opponent's hand: skipping a call that lands the opponent on a cut is worth +14 (at 50) to +26 (at 100) points of match-win rate each time.
  - From what the net can see, with 0 / 1 / 2 hidden opponent cards: +21 / +2 to +19 / +4 to +6 each against hm5. With 3 hidden cards it is negative, so it is left out.
  - Whole matches, 3,200 on seeds 301–302, against hm5's score on the same deals: **+6.8 points head to head with hm5**, −1.2 against the trained net, −1.1 against the counter (about ±1 noise).
  - The narrower version with only 0–1 hidden cards was +3.6 head to head and 0 elsewhere.
  - I chose the wider version because the head-to-head is the main event. Against hm4, which also steers its score onto the cuts, it lifts matches from 52.1% to 59.5% on the final check.

**Tested and not kept** (all with numbers in the logs):
- **Yaniv call vs play on** (`endgame.md`, 24,000 rounds per opponent). Flipping hm5's call either way loses in every cell (−4.4 / −5.9 each). The threshold is right.
- **The trained net's moves after both hands hold 4 cards or fewer.** Overall +0.09 ± 0.08 a round against the counter and −0.07 ± 0.08 against the trained net: hm5 has no clear endgame gap left apart from rule S.
- **Leaving an unpaired open 4–5 late in the round.** +0.027 ± 0.017 a round against each opponent. Not clear enough to build.
- **The draw decision at 5 cards** (`mine.md`, `twin.md`). hm5's draw is right in every class with more than 100 cases.
- **Denying the opponent a rank it is known to hold** (−5 / −4 after 4 cards).
- **Keeping A / 2 / joker** (−2.5 / −1.6).
- **Match play** (`match.md`):
  - Calling more or less by score: every region loses, for example −15 each when the opponent is near 100.
  - Throwing points harder near 100: −1 to −14.5 each.
  - Re-weighting hm5's own cut steering: about right already. Giving up a landing costs −8 to −17 each, and still −5 to −7 against an opponent that has the veto.
- **Mirror and 4-card-phase mining.** These were broad maps only: both designers stopped on a usage limit before grouping by class. Their maps cover 16,000–20,000 rounds per opponent, the alternative being another player's move at every decision. hm5's moves beat the counter's (−1.0 to −1.1 a time against hm5 and the trained net) and match the trained net's (−0.04 ± 0.14 and +0.04 ± 0.14).
- **An open lead, not built.** After my hand holds ≤4 cards, where the trained net draws from the deck instead of taking an open 3–4, the deck draw gains +1.4 to +1.8 a time against the trained net and hm5, but nothing against the counter. That is about 0.03–0.05 a round, and the reason was not found.

## Honest estimate

- **Against hm5** (the likeliest base of the other team's net): about **58% of matches** and +0.14 margin a round. That comes from about 3,000 matches on fresh seeds and 1,600 more on another seed.
  - Roughly 6 to 7 points of the 8 come from the cut veto, which only pays against a player that steers its score onto 50 / 100 as hm5 does.
  - If the other team found and removed that habit, or added the same veto, expect the head-to-head lead to shrink to about +0.1 margin and 51–53% of matches.
- **Against the trained net:** ahead on margin by about +0.2 a round more than hm5 (+0.29 to +0.35 against hm5's +0.11 to +0.13), matches unchanged at about 62–64%.
  - This is the first hand-written real-game net that beats the trained net's margin.
- **Against the card counter:** +0.12 to +0.22 margin a round better than hm5, and about the same in matches (72%).
- **What this is not:** a new theory of the game. hm5 already played the owner's twin idea and the Yaniv threshold well.
  - What we added are three precise exceptions, each verified on the same hidden cards against two or three opponents.
  - The round-margin gain is about +0.15 to +0.2 a round. The match gain is mostly the cut veto.

## How it was made (what "by hand" meant here)

- **The team.** Five designers worked as parallel sub-agents (topics `twin`, `endgame`, `mine`, `match`; `mirror` and `race` stopped on a usage limit before finishing). Each logged every variant. The lead reviewed each block's units and its play-out results before adding it.
- **The instruments.** Paired play-outs (`training/yaniv/contest_opus_playout.py`): at each decision where a stated rule disagrees with the net, both moves are played to the end on the same hidden cards.
  - These tested stated rules; nothing was fitted to them.
  - The match designer also used a match-win table (`match_value.win_table` over hm5-vs-hm5 rounds). It only converts round outcomes into match-win units; no weight was read from it.
- **Hand-picked numbers.** Each block's thresholds were chosen by hand after looking at grouped play-out tables. TRIPLE's bonus is +5, the size of one card charge (+8 was tried and changed 0.002 decisions a round). Rule S has thresholds 0.005 and 2 points, and a 1-point variant was tried and dropped. The cut veto came in two versions, with 0–1 or with 0–2 hidden cards.
  - No loop ran over coefficients, and no gradient, genetic, search or self-play code was used.
- **Logs.** In `results/contest-opus-logs/`:
  - `BRIEF.md`: the brief every designer worked to.
  - `twin.md`, `endgame.md`, `mine.md`, `match.md`: every variant.
  - `baseline-s300.txt`, `v3-s400.txt`, `final-s777.txt`: whole-game runs.
  - `v2-s400-aborted.txt`: a run cut short because the model file changed under it.

## Reproduce

```bash
python -m training.yaniv.contest_opus            # writes public/model/yaniv-net-contest-opus.json (-v: every unit)
python -m training.yaniv.contest_opus_eval public/model/yaniv-net-contest-opus.json hm5 \
       --opps hm5,trained,counter,hm4 --rounds 12000 --matches 3000 --seed 777     # the final check
python -m unittest training.yaniv.test_contest_opus
```
