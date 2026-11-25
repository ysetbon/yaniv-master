# Team fable's contest net (real game: 5 cards each, Yaniv at 7 or less)

**Result.** One plain net, written by hand with no training: `training/yaniv/contest_fable.py` ->
`public/model/yaniv-net-contest-fable.json`, shape [125, 253, 111, 66, 35, 33, 53, 29, 26, 1], **606 hidden units**
(limit 1,020). Against handmade5 (`hm5`, the bar) it wins **57.2% of 10,000 matches** and **+0.24 points a round**.
Most of the match gain is one new rule: do not call Yaniv when the call would hand the opponent its own 50/100 cut.

Margin is the opponent's points minus ours, a round, with its paired standard error. Matches are mirrored pairs
with a 95% range. All runs are `python -m training.yaniv.contest_fable_lab duel ...`; raw lines are in
`results/contest-fable-logs/lead-duels.txt`.

| Pairing | Seed | Margin a round | Rounds won | Matches won (95% range) |
|---|---|---|---|---|
| **contest net vs hm5** | 71 | **+0.243 ±0.035** (40,000) | 52.2% | **5718–4282, 57.2%** (56.2–58.1) |
| contest net vs trained net | 82 | +0.196 ±0.082 (20,000) | 53.5% | 3970–2030, 66.2% (65.0–67.4) |
| hm5 vs trained net, same seed | 82 | −0.051 ±0.081 | 51.2% | 3767–2233, 62.8% (61.6–64.0) |
| contest net vs card counter | 82 | +1.909 ±0.096 (20,000) | 57.7% | 4386–1614, 73.1% (72.0–74.2) |
| hm5 vs card counter, same seed | 82 | +1.724 ±0.096 | 56.3% | 4283–1717, 71.4% (70.2–72.5) |
| contest net vs a sparring partner that also denies cuts (see below) | 81 | +0.086 ±0.048 (20,000) | 51.8% | 5172–4828, 51.7% (50.7–52.7) |

The three top pairings were each run once on the final net, on seeds not used to choose anything.

## How the net plays (what was added to handmade5)

The net is handmade5's valuation of a throw with six changes. Each can be switched off in `build(...)`; all off
rebuilds handmade5 weight for weight (checked).

1. **Deny the opponent its cut** (22 units, `contest_fable_match.add_deny_units`). No Yaniv call when at most 2 of
   the opponent's cards are hidden and its score + estimated hand is within 0.5 of 50 or 100 (within 3 when 1–2
   cards are hidden). A successful call there adds its hand to its score, lands it on the cut and takes 50 off.
   This veto overrides even a call that cannot be caught.
2. **Landing on my own cut, re-weighted** (2 units). Taking the open card or drawing from the deck to leave my
   score + hand exactly on 50/100 is rewarded more, especially at 100, where a point is worth 2–3 times as much
   match-win chance.
3. **Call when a lost round ends the match anyway** (4 units): my score + points > 100 and catch chance < 0.15.
4. **A new Yaniv rule** (7 units, `add_yaniv_units`).
   - Play on when my highest group is 2+ points above the open card: throw it and take the open card instead.
   - Otherwise call when the uniform catch chance `assaf` is under a threshold set by the opponent's card
     count. `assaf` understates the true catch rate about 20–25 times against 4–5 cards, 6–8 times against 3,
     2–2.5 times against 2 and not at all against 1, nearly the same for all three opponents.
5. **The triple take** (9 units, `add_triple_units`). Holding a pair with a third of that rank face up: keep the
   pair, throw another card, take the third, and throw all three next turn. Ranks worth 7+ against 4–5 cards,
   5s and 6s only against 5 cards.
6. **Catch bonus only when I keep 2+ cards** (4 units, `contest_fable_reads.fix_catch`). No opponent calls
   against a 1–2 card hand unless it knows my card, so the bonus for "its call gets caught" was paying for
   something that does not happen.

## The strategy test: keep a high single for its twin (the owner's idea)

**It does not pay beyond what the net already does.** Full log: `results/contest-fable-logs/twin.md`.

Definition tested: the opponent holds 5 or 4 cards; I hold two or more unpaired singles worth 5+ of different
ranks; which one goes first. Both choices were played to the end on the same hidden cards, hm5's policy
afterwards: 30,000 rounds against hm5, 20,000 against the trained net, 16,000 against the counter (seed 11).

| Rule (alternative to the net's move) | vs hm5 | vs trained | vs counter | Times a round |
|---|---|---|---|---|
| Keep the higher single, throw the lower (equal unseen copies), opp 5 cards | −0.28 ±0.09 | −0.37 ±0.12 | −0.26 ±0.12 | 1.1–1.3 |
| ... at ply 0–1 | −0.02 ±0.13 | 0.00 ±0.15 | −0.02 ±0.17 | |
| ... at ply 2–3 | −0.53 ±0.18 | −0.72 ±0.22 | −0.45 ±0.23 | |
| ... at ply 4+ | −0.84 ±0.24 | −1.21 ±0.29 | −0.69 ±0.28 | |
| Same, opp 4 cards | −0.47 ±0.10 | −0.40 ±0.12 | −0.70 ±0.17 | 0.7–0.8 |
| Keep the higher single although it has fewer unseen copies, opp 5 | −1.87 ±0.25 | −2.29 ±0.29 | −1.94 ±0.29 | 0.16–0.22 |
| Throw the higher single where the net kept it because it is more live, opp 5 | −0.99 ±0.40 | −1.38 ±0.49 | −0.62 ±0.45 | 0.05–0.06 |
| Draw from the deck instead of taking an open ten-card twin, opp 5 | −4.81 ±0.33 | −4.39 ±0.32 | −4.23 ±0.29 | about 0.2 |

- **The idea is already in the net.** handmade5 charges 1 point per unseen copy of the rank it throws. When the
  higher single is also the one with more unseen copies it keeps it 88% of the time (about 0.10 times a round).
- **Going further loses.** A charge of 2 or of 0 was worse than 1 against all three opponents on 20,000 rounds
  (seed 31): −0.05 / −0.08 against hm5, −0.06 / −0.15 against trained, −0.13 / −0.10 against the counter.
- **Why.** The opponent's next throw is the twin of a ten-card I hold 12% of the time with 3 copies unseen and
  about 3.5% with 2. That depends on unseen copies and hardly on which high card it is, so keeping the heavier
  card just costs 0.5–1.2 points from ply 2 on.
- **Taking the twin when it appears is right**, and the net does it 98% of the time against 5 cards.

## Scoreboard of variants

"Paired" is play-both-moves-out, points each time ± standard error. Duels are against hm5 unless stated.

| Variant | Reason | Result | Kept |
|---|---|---|---|
| Keep a high single for its twin (all versions above) | owner's idea | negative or neutral, all opponents | no |
| Live-rank charge 0 or 2 instead of 1 | more / less of the same idea | −0.05 to −0.15 a round, all opponents | no |
| Yaniv: swap veto | known gap in handmade5 | paired +4.3 / +4.4 / +4.0 (hm5 / trained / counter), 0.009 a round | yes |
| Yaniv: threshold by the opponent's card count | `assaf` is wrong by a factor that depends on card count | with the veto: +0.112 ±0.020 a round (80,000 rounds, seeds 31, 32); 50.1% of 6,000 matches | yes |
| Yaniv: rule on estimate minus my points alone | simpler | −0.03 to +0.06 a round | no |
| Yaniv: other "play on" moves (swap saving 1 point, set for a lower open card, deck draw, breaking a set) | variants of the veto | −0.8 to −10.9 each | no |
| Yaniv: risk term from a low open card | the opponent just threw it | clear only against the trained net | no |
| Triple take | survey of every legal alternative | paired +3 to +5; duel +0.064 ±0.015 (40,000 rounds, seed 303) | yes |
| Refuse an unpaired open 3 (5 cards v 4) / open 4 (2 cards) | survey | +3.25 → +1.29 and +1.86 → +0.41 on a second seed | no |
| Other throw alternatives (highest single v set, second-highest single, breaking a pair, low pairs, two-turn order, forcing out aces/jokers, waiting for a third from the deck) | survey, about 7,000 cells | none positive against two opponents | no |
| Catch bonus only when I keep 2+ cards | the opponent does not call against 1–2 cards | paired −1.18 / −1.02 / −0.43 for hm5's move; duel +0.043 ±0.029; on the merged net +0.029 ±0.020 (40,000, seed 61) | yes |
| Catch bonus off / only when I keep 3+ | same | about half the gain / loses at 3 cards | no |
| Corrected estimate of "the opponent calls next" (my card count, low open card) | strong signals in the tables | +0.03 a round worse (keeps pairs it should throw) | no |
| Deny the opponent its cut + re-weighted own cuts + call when out anyway | match play | 56.5% of 10,000 matches (seed 31); rounds from 0-0 unchanged | yes |
| Re-weighted own cuts alone | same | 51.5% of 10,000 | yes |
| Veto a call when being caught would end the match; penalty for holding over 100 − score | match play | measured as useless | no |
| Yaniv rule + triple take, before the match rules | first merge | +0.184 ±0.032 a round, 50.0% of 10,000 matches (seed 51) | — |
| **Full net** | everything kept | **+0.243 ±0.035, 57.2% of 10,000 (seed 71)** | shipped |
| Full net with handmade5's Yaniv rule | does the new rule cost matches? | 57.6% vs hm5; full net vs this 50.1% of 10,000 (seed 72) | no |

## What worked

- **Match-score play is where the matches are.** The match rules add about 7 points of match wins against hm5
  while leaving rounds from 0-0 untouched; everything else together adds +0.24 a round and no visible matches.
- **Playing every alternative out** found the triple take: the only one of about 7,000 situation-by-alternative
  cells that was positive against all three opponents and survived a second seed.
- **handmade5's tables are not tuned to the counter alone.** "The opponent calls next turn" and "the round ends
  before my second turn" measure nearly the same against hm5 and the trained net (`reads.md` tables B, C).

## What did not

- The owner's twin-keeping idea, beyond the existing charge per unseen copy.
- Reading the opponent better. The signals are real but the throws do not improve.
- Round gains turning into matches: +0.18 a round was worth 50.0% of 10,000 matches.

## Honest estimate

- **Against handmade5:** clearly ahead, 57.2% of matches and +0.24 a round.
- **Against the trained net:** ahead on both counts on the one seed measured (+0.20 against hm5's −0.05; 66.2%
  against 62.8%). The round margin against the trained net is within about 2 standard errors of zero.
- **Against the other team:** unknown, and probably close. The cut denial only pays against an opponent that
  plays for its cuts. Against a sparring partner built from our own pieces (handmade5 + triple take + cut denial,
  which itself beats hm5 54.7% of 10,000) the contest net wins 51.7% of 10,000 and +0.09 a round.
- **Near the line of "by hand":** the Yaniv thresholds and the deny distances are a handful of constants read off
  paired tables against hm5 and the trained net; the Yaniv thread's whole-rule paired gain against those two is
  in-sample. No loop tuned anything, and no weight was copied from a trained net.

## Disclosures

- **The other team's commit title.** After the restart, `git log` on main showed "Opus contest net: hm5 + rule S +
  TRIPLE + cut veto (611 units)". None of their files, their net or the judging opponent was opened. Our triple
  take and cut denial were already measured by then (`survey.md`, `match.md`). The sparring partner above was
  built after seeing that title, from our own units only.
- **Interruptions.** The account's usage limit stopped all five sub-agents once and the match thread a second
  time. The match thread never sent a final report; its design and duels were merged from
  `contest_fable_match.py` and `match.md`, and the head-to-head was re-run by the lead.
- **Not re-run by the lead:** the paired tables of the twin, Yaniv, survey and reads threads. The lead re-ran
  whole-net duels only.
- **Code layout.** `contest_fable.py` builds the net; the match-score units and the catch gate live in the
  helpers `contest_fable_match.py` and `contest_fable_reads.py`, which it imports.

## Reproduce

```bash
python -m training.yaniv.contest_fable            # writes public/model/yaniv-net-contest-fable.json
L="python -m training.yaniv.contest_fable_lab duel"
$L py:contest_fable.build hm5 --rounds 40000 --matches 10000 --seed 71
$L py:contest_fable.build trained counter --rounds 20000 --matches 6000 --seed 82
python -m unittest training.yaniv.test_contest_fable
```

Logs: `results/contest-fable-logs/` (`twin.md`, `yaniv.md`, `survey.md`, `reads.md`, `match.md`, the duel files,
and `BRIEF.md`, the sub-agents' brief).
