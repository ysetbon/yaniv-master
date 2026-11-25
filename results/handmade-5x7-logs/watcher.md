# watcher: rules read off good players in the 5-card phases

Design: `training/yaniv/handmade5_designs/watcher.py`. Saved net:
`/tmp/claude-0/-home-user-yaniv-master/a532e2e0-83d9-59dd-8a1a-414723f6a44f/scratchpad/watcher/watcher.json`
(468 hidden units, 8 layers). Scripts named below are in that scratch folder.

**In one line.** I watched the trained net (T), the 4-card hand-written net (H, the bar) and the card counter (C)
on the same 84,000 real-game decisions with a 5-card hand on the table, and settled each kind of T-versus-H
disagreement by matched rollouts. T is right in exactly one kind (throw a pair of 2s before a high single while
nobody is near calling); in every other kind H is level or better. The net is therefore the bar plus that one rule.

## Variants measured (working test: 6,000 rounds seed 61000, 1,500 matches seed 61001; against the counter)

"switch" = `switch:NET:yaniv-net-4cards-yaniv7-handmade.json` (handover once both hands hold 4 or fewer).
"first-order gain" = `diag.py`: at every 5-card-phase decision where NET and the bar differ, the round is played out
both ways on the same deal (mover continues as the bar, opponent the counter); the sum of (NET's move - bar's move)
per round. It is matched, so far less noisy than the working test (+-0.05 against +-0.2).

| variant | what changed and why | working test | first-order gain a round | kept? |
|---|---|---|---|---|
| bar (H alone) | the 4-card net, whole game | +1.49, 3315/6000 (55.2%), caught 4.5%, matches 1029-471 (68.6%) | 0 | the bar |
| v1 | bar + "a 2 is a whole heavy card at 5 v 5" (rule 1, charge still 5) + feed penalty cut to a 0.5 tie-break (my misreading, see below). Expected about +0.2 from the rollouts | switch, rounds only: +1.49, 3325/6000 (55.4%), caught 4.5% | +0.06 +- 0.05 (22 thrown: +0.12; feeding a twin: -0.04) | no |
| v2 | v1 + heavy-card charge 8 instead of 5 at 5 v 5. Reason: v1 threw the 22 in only 62% of the trained net's cases; with a second ten in hand the plan credits the next throw and the 22 lost by 2 points | not run | +0.19 +- 0.06 (22 thrown: +0.26; feeding: -0.04) | no |
| v3 | v2 with the bar's feed penalty restored (the diagnostic and a re-read of the rollout sign showed the bar is right to keep a card whose twin the opponent holds) | switch, rounds only: +1.61, 3352/6000 (55.9%), caught 4.5% | about +0.25 | superseded |
| v4 | v3 + rule 1 also at 5 v 4 + a +1 bonus for "throws a set and keeps a pair at 5 v 5" (meant to throw the lighter of two pairs first) | not run | +0.45 +- 0.10 on 6,000 rounds (22 at 5 v 5: +0.28; 22 at 5 v 4: +0.14); the bonus changed no move (both set throws get it) | bonus dropped |
| **v5 = final** | v3 + rule 1 also at 5 v 4 (I hold 5, opponent 4 or 5) | **switch: +1.63, 3362/6000 (56.0%), caught 4.5%, matches 1060-440 (70.7%, 68.3%-72.9%)**; alone (whole game): identical, +1.63, 3362/6000 (56.0%), caught 4.5%, matches 1060-440 (70.7%): rule 1 is gated on my holding 5 cards, so once both hold 4 or fewer the net IS the bar and the handover changes no move | as v4 | **yes** |

Against the bar the working-test differences (+0.14 margin, +0.8 points of rounds, +2.1 points of matches) are each
inside the noise (+-0.2, +-2.5); the matched rollouts are what say the rule is real (+4.7 +- 0.5 points each time
it applies at 5 v 5). For scale, the trained net in the same seat (`switch:yaniv-net.json:4-card net`) scores +1.69,
55.9%, 1014-486 (67.6%).

## How the watching was done

1. `probe.py` (seed 7, 12,000 rounds from 0-0: T v counter and H v counter, alternating): at every decision of
   either seat where somebody holds 5 cards (84,109 decisions) record what T, H and C would each play
   (`features.action_key`), the hand, open card, card counts, ply, `opp_est`, `assaf`, the opponent's known cards and
   true total; keep the full state wherever the three do not all agree.
2. `a1.py` tabulates choice frequencies by situation; `kinds.py` names each T-versus-H disagreement; `a2.py` counts them.
3. `roll.py`: for each disagreement, play the round out twice from the recorded state (the stock order is in the
   state, so the deal is the same), once with T's move and once with H's, the mover then continuing as H and the
   opponent as the counter; report mean (T's move - H's move) in points of margin by kind. `roll2.py` does the same
   for positions where T and H agree and the counter differs.
4. `exp.py` (seed 11, 15,000 rounds of H v counter): the same rollouts on constructed alternatives at every position
   that fits a description (not only where T chose it). `regret.py`: one random other move against the net's own at
   each of its 5-card-phase decisions.
No number from these went into the net except through a sentence and a round constant.

## What they do (frequencies; n in the thousands unless shown)

Disagreement T v H on throws: 16.7% at 5 v 5, 11% when I hold 5 and the opponent fewer, 8% when I hold fewer and the
opponent 5. All three agree far more than they differ; the counter differs from both nets on about 20%.

**Yaniv.** While the opponent holds 5 cards everybody calls every time Yaniv is legal (1,553 positions; `assaf` is
0 or under 0.01; really caught 2.7% when `assaf` is not 0). Declining there cost 9-14 points in rollouts. Nothing to change.

**Open card** (take rate, opponent holding 5):

| open card | T | H | C |
|---|---|---|---|
| unpaired, worth 0-2 | 1.00 | 1.00 | 1.00 |
| unpaired 3 | .92 | .86 | .98 |
| unpaired 4 | .18 (ply 0-3: .03, ply 8+: .44-.57) | .00 | .73 |
| unpaired 5 | .08 | .00 | .30 |
| unpaired 6+ | .00-.02 | .00 | .00-.01 |
| pairs a card I keep | .97-1.00 | .99-1.00 | .98-1.00 |
| matches a pair I hold | .89-1.00 | .67-.84 | .87-.94 |
With 2-3 cards against 5, T takes unpaired 4s 75% / 44% and 5s 44% / 19%; H never.

**One pair and singles** (rate of throwing the pair; open card pairs nothing):

| pair | highest single | T 5v5 | H 5v5 | C 5v5 | T 5v4 | H 5v4 | T 5v3 | H 5v3 | T 4v5 | H 4v5 |
|---|---|---|---|---|---|---|---|---|---|---|
| jokers, aces | any | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 22 | 10 | .96 | .00 | .00 | .61 | .00 | .10 | .00 | .69 | .00 |
| 22 | 8-9 | .86 | .00 | .00 | .46 | .00 | .10 | .00 | .56 | .00 |
| 22 | 5-7 | .88 | .01 | .00 | .60 | .01 | .29 | .01 | .50 | .04 |
| 33 | 10 | 1.00 | .80 | .00 | .92 | .88 | .61 | .99 | .97 | .96 |
| 44+ | any | .96-1.00 | 1.00 | follows 2x pair >= single | .97-1.00 | .99-1.00 | .65-1.00 | 1.00 | .95-1.00 | 1.00 |
Nobody ever breaks a pair. With no set in hand all three throw the highest card about 91% of the time.
Two pairs at 5 v 5: T throws the heavier one first only 62% (H 93%, C 88%).

**Kinds of T v H disagreement** (share of decisions in the phase):

| kind | 5 v 5 | 5 v <5 | <5 v 5 |
|---|---|---|---|
| a different ten-point single (T/J/Q/K) | 8.5% | 3.0% | 1.9% |
| T throws 22, H a single | 3.6% | 3.2% | 1.4% |
| two sets, T throws the lighter | 1.5% | 0.6% | 0.1% |
| T throws a lower single (some to take a pairing open card) | 1.5% | 0.4% | 0.5% |
| T takes an unpaired open 4 / 5 / 6 | 0.1% | 0.9% | 2.1% |
| H throws a lower single (keeps a card whose twin the opponent is known to hold) | 0.3% | 0.6% | 0.9% |
| H throws a set of 3+, T a single | 0.6% | 0.7% | 0.6% |

## Who is right (matched rollouts; + = the first-named move is better; points of margin +- standard error)

| kind | result | read as |
|---|---|---|
| T throws 22 instead of H's single | 5 v 5: **+5.1 +- 0.7** (n 983); 5 v 4: +2.7 +- 0.8 (650); 4 v 5: +3.6 +- 1.2 (247); 5 v 3: +0.3 +- 1.3 (212) | T right while the opponent has 4-5 cards |
| ... the same on EVERY position with 22 and a single worth 5+ (`exp.py`) | 5 v 5: **+4.7 +- 0.5** (1,642); 5 v 4: +0.6 +- 0.5 (1,404); 4 v 5: -0.1 +- 0.8 (587); 5 v 3: -2.9 +- 0.6 (839); 5 v 2: -4 | rule for 5 v 5, weakly 5 v 4 (third sample, `diag.py`: +2.4 +- 1.2, n 298), not for 4 v 5 |
| throw AA instead of a single worth 5+ (nobody does it) | 5 v 5: -1.4 +- 0.5; elsewhere -5 to -10 | aces stay light |
| T throws the fed high card, H keeps it and throws a lower one | **-4.0 +- 1.2** (274); by mover H -7.4, T -3.1, C -2.5 | H right: keep the feed penalty. (I first read this sign the wrong way round and cut the penalty in v1-v2; `diag.py` showed -3.1 +- 1.7 for it and I put it back.) |
| a different ten-point single | +0.25 +- 0.23 (4,129) | arbitrary. Neither net prefers the rank with fewer unseen copies (81% same count); throwing the "dead" one: -0.3 +- 0.5 (938) |
| two sets, T throws the lighter | +0.7 +- 0.6 (705); constructed: 5 v 5 +0.9 +- 0.45, 5 v 4 +0.3 +- 0.6, 5 v 3 -3.2 +- 0.9, 5 v 2 -5.3 | not separated; not used |
| T throws a lower single | -0.35 +- 0.74 (475); taking a pairing open card with it: -0.3 +- 1.0 | no difference |
| T takes unpaired open 4 | -0.2 +- 0.8 (508); constructed: -0.8 +- 0.4 (3,372) | H right or level |
| T takes unpaired open 5 / 6 | -1.6 +- 1.2 / -2.6 +- 1.7; constructed 5: -1.3 +- 0.3 (4,634) | H right |
| H throws a set of 3+, T a single | +0.1 +- 0.8 (373) | no difference |
| open card matches my pair, take it where H draws the deck | -3.0 +- 1.4 (199) | H right |
| counter v both nets (where T = H): keeps a single instead of a set of 3+ | -1.1 +- 0.3 (3,252) | nets right |
| counter takes unpaired open 4 / 5 | -1.3 +- 0.55 / -2.1 +- 0.6 | nets right |

`regret.py` on the final net (6,000 rounds, about 12,000 random other moves): no kind of other move with 60+ samples is
better than the net's own. At 5 v 5 mistakes are cheap (throwing a single 4+ points lower: -3.2; taking an
unpaired open 7+: -0.4 +- 0.9) and outside it dear (-6.5; -2 to -4): points barely matter at 5 v 5, which is the
reason behind rule 1.

## The rule as written in the net

RULE 1: while I hold 5 cards and the opponent 4 or 5, a 2 is a heavy card like any other and a heavy card is
charged 8 points instead of 5. Two layer-2 units gated on the phase (`twos_big`: 2s left after the move;
`heavy_big`: heavy cards left, a card taken onto its twin not counted). Constants: `TWO_FULL = 1` (was 0.5: "2s
are thrown like 3s there"), `CARD_BIG = 8` ("a card must be worth more than the 6 points between 22 and a ten;
8 leaves 2 to spare"; 5 and 8 were the only values tried), `BIG_OPP = (4, 5)`. Everything else is the bar's units
imported from `handmade.py`, so outside that phase the net is the bar move for move.

## What worked
- Settling disagreements by matched rollouts on the same deal: thousands of positions give +-0.5 points on one
  kind of decision, where the working test cannot see a +0.2 change.
- The 22 rule: +4.7 points each time it applies at 5 v 5, about +0.3 to +0.45 a round first-order.
- `diag.py` as a check on my own reading: it caught the feed-penalty sign error after one variant.

## What did not
- Cutting the feed penalty (v1-v2): -0.04 a round; a misreading, reverted.
- Rule 1 with the bar's charge of 5 (v1): only 62% of the cases flipped; working test +0.00.
- The "set that keeps a pair" bonus (v4): cannot prefer one pair over the other; and the evidence for throwing
  the lighter pair first is 2 standard errors at 5 v 5 and negative below it. Dropped.
- Everything else the trained net does differently in these phases (unpaired open 4s late, 5s with a short hand,
  taking a third card for a triple, a lower single): level or worse than the bar.

## Found on the way (for the lead)
- In `handmade.py` the Yaniv reason `call` also turns on for THROW rows when my hand is above about 35 points
  (`high_call` = 0.005 a point above 5 is not capped at 7, so with 5 cards 500 x 0.005 x 31 > 100): every throw
  row then gets +2000. It is the same for all throws of a position and Yaniv is never legal there, so play is
  unchanged (checked: 1,710 of 20,029 stored positions show it, same moves), but the "throws score within
  -100..+60" note is not true in the real game. A cap on `high_call` at 7 points would remove it.
- The trained net alone scores +1.92 on the working test and +1.69 with the handover to the bar, so about half
  of its edge over the bar is in the 4-cards-or-fewer phases; in the 5-card phases the only thing it does better
  than the bar is the 22 rule.

## What I would try next
- The same watching in the phases where both hold 4 or fewer, in the real game (the bar was tuned on the 4-card
  game's deals; real-game hands arrive there after sets were thrown).
- 4 v 5 with a pair of 2s: +3.6 where the trained net throws it, 0 over all positions: something tells it when
  (probably the other cards: a low hand close to 7 keeps its 2s). A second rule could be read from a table by
  kept points.
- Two pairs at 5 v 5, lighter first (+0.9 +- 0.45): needs 4x the sample before it is worth a unit.
