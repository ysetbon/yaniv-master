# handover: where to hand over to the 4-card net, and what the trained net does better after it

Script: `training/yaniv/handmade5_handover.py` (`run` / `show` / `swap` / `cuts`). Scratch (result files, ad hoc
grouping scripts `calls.py`, `calls2.py`, `swap.py`):
`/tmp/claude-0/-home-user-yaniv-master/a532e2e0-83d9-59dd-8a1a-414723f6a44f/scratchpad/handover`.
Names: hm4 = the 4-card hand-written net; port = round 1's `lead/port.json`; T = the trained net.
Seeds 920100-922100. Nothing here writes or fits weights; every number is a measurement of existing players.

## Short answers

1. **Handover rule.** For a hand-written main net (port) the rule does not matter: all five options are the
   same player within every test I have, so keep **"both hands <= 4"** (the only one with a reason: the rest
   of the round is then exactly the 4-card game, and it leaves every 5-card rule in force). For T as main the
   rounds do not depend on the rule either, but **"opp <= 4" (or "either")** gives T more cut plays (0.47 a
   match against 0.40) at no cost in rounds.
2. **Frequencies (main = port, 70,000 rounds).** "both <= 4": 81.6% of rounds reach it, 48.7% of my decisions
   come after it. mine 91% / 64%, opp 90% / 64%, either 99.4% / 79.7%, both <= 3: 34% / 14%.
3. **T's edge after the handover is +0.13 +- 0.03 a round** (T plays the whole endgame instead of hm4, same
   cards, 140,000 rounds). One class carries 0.05-0.06 of it and is clearly wrong in hm4: **calling Yaniv with
   a small but real catch risk when a lower open card could be taken instead** (rule in words below). The rest
   (+0.06) is spread over throws late in the round (ply 9+); in endgames that start early (ply < 9) hm4's
   moves are the better ones (-0.045 +- 0.015 for T's).
4. **Match play.** The score-unit gain needs hm4 (or any net with the score units) in charge whenever the
   opponent may call soon, which includes "I hold 5, the opponent 4 or fewer". With T as main, "opp" and
   "either" lift cuts from 0.40 to 0.47 a match (hm4 alone: 0.46); matches 70.0% / 69.7% against 68.2% for
   "both" (3,500 matches each, +-0.8 each: a 1.5 SE difference, not a finding by itself).

## Method

- **Working test** (`handmade_eval --game 5x7 --work`, 6,000 rounds seed 61000, 1,500 matches seed 61001; I
  ran it with `--workers 1`, the split is fixed). Noise +-0.2 margin, +-2.5 points of matches.
- **Paired playouts** (`handmade5_handover run`): hero = main until both <= 4, then hm4, against the counter
  from 0-0. Before the handover hm4 (and T) is asked for its move, after it T is; where the move differs from
  hero's, both are played to the end of the round on the same hidden cards with hero's policy afterwards.
  "A round" numbers are the sum of (other move - hero's move) over differing decisions divided by rounds;
  +- is sqrt(sum of squares) / rounds. First order (one move changed), 0-0 only (no match-score play).
  Once a round, at hero's first decision after the handover, the round is also played out with T making all
  of hero's remaining decisions ("whole endgame").
  Samples: port main 30,000 (seed 920100) + 40,000 (920300); T main 30,000 (920200) + 40,000 (920400).
- **Direct alternative** (`swap`): at every Yaniv call by hero where the open card is lower than hero's
  highest card, the call against "throw the highest single, take the open card". 2 x 60,000 rounds (seeds
  922000, 922100), main = port.
- **Cuts** (`cuts`): 2,000 matches to 100 against the counter, seed 921000 for every player; a cut = my score
  landing exactly on 50 or 100.

## 1. The handover rule

Reasoning first. After "both <= 4" no hand can grow, so the rest is the game hm4 was written for. "mine" would
give hm4 the cells (me <= 4, opp 5), "opp" the cells (me 5, opp <= 4), "either" both; "both at 3" keeps the main
net in the cells where a hand still holds 4. port is hm4 plus rules that fire only when the opponent holds 5
cards, so by construction "opp" and "both at 3" are the same player as "both" (and as port alone), and
"mine" = "either" differs only by port's rules in (me <= 4, opp 5).

Working test, rounds (margin a round; matches where played):

| handover | main = port | main = T |
|---|---|---|
| both <= 4 | +1.65, 70.8% of matches (port's log) | +1.69, 67.6% (BRIEF) |
| mine <= 4 | +1.68 | +1.67 |
| opp <= 4 | +1.66 | +1.68, matches 1024-476 (68.3%) |
| either <= 4 | +1.69 | +1.66, matches 1053-447 (70.2%) |
| both <= 3 | +1.65 | +1.81 |
| no handover | +1.66, 70.5% (port's log) | +1.92, 63.2% (BRIEF) |

Every row for port is inside +-0.04: no difference. For T the four rules at 4 are inside +-0.03; "both at 3"
(+1.81) and alone (+1.92) are higher, in the direction of finding 3 (T is better in the late endgame), but
+0.12 is inside the test's noise.

Paired playouts (gain a round of handing those cells to hm4, i.e. hm4's move instead of the main net's):

| cells | main = port (70,000 rounds) | main = T (70,000 rounds) |
|---|---|---|
| me <= 4, opp 5 ("mine" adds these) | -0.016 +- 0.009 (0.029 differing decisions a round) | +0.020 +- 0.016 (0.082) |
| me 5, opp <= 4 ("opp" adds these) | 0.000 (17 differing decisions in 70,000 rounds) | +0.005 +- 0.019 (0.093) |
| me 5, opp 5 (nobody proposes this) | -0.125 +- 0.019 (0.090) | -0.038 +- 0.029 (0.220) |
| after: a hand holds 4 ("both at 3" gives these back to T) | T's move instead of hm4's: -0.001 +- 0.023 | same cells |
| after: both <= 3 | T's move instead of hm4's: +0.064 +- 0.020 | same cells |

So: for port, "mine"/"either" lose port's small edge in (me <= 4, opp 5) (-0.016 +- 0.009, the two samples gave
-0.036 +- 0.014 and -0.001 +- 0.012: not a finding), "opp" changes nothing. port's (5, 5) rules are worth
+0.125 +- 0.019 a round over hm4's moves there, confirming round 1. T's moves instead of port's before the
handover: (5, 5) -0.022 +- 0.029, (me <= 4, opp 5) -0.011 +- 0.018, (me 5, opp <= 4) -0.057 +- 0.022: port's
5-card phase is no worse than T's anywhere. For T, hm4's moves in the "mine" and "opp" cells are worth the
same as T's: an earlier handover is free in rounds. "both at 3" would not help: T's edge sits in both <= 3,
exactly the cells that rule still gives to hm4.

**Recommendation: keep "both <= 4" for the hand-written player.** If the final player's main net has no
match-score units of its own (T), hand over at "opp <= 4" instead.

## 2. Handover frequencies (hero = port then hm4, against the counter, 70,000 rounds from 0-0, 6.84 decisions a round)

| rule | rounds reaching it | my decisions after it |
|---|---|---|
| both <= 4 | 81.7% / 81.5% (two samples) | 48.8% / 48.7% |
| mine <= 4 | 90.9% / 91.0% | 64.1% / 64.2% |
| opp <= 4 | 90.2% / 90.0% | 64.2% / 64.3% |
| either <= 4 | 99.4% / 99.5% | 79.6% / 79.8% |
| both <= 3 | 33.8% / 34.0% | 14.1% / 14.3% |

The same with T as main: 81.6 / 81.1%, 49.2 / 48.7% for "both"; the others within 0.5 points. They agree with
`measure.md` part d (hm4 alone: 81.3%, 48.4%). 18% of rounds end before the handover.

## 3. Where T beats hm4 after both hands hold <= 4

**Size.** T playing all of hero's decisions after the handover instead of hm4, same cards:

| sample | gain a round |
|---|---|
| port main, 30,000 | +0.078 +- 0.058 |
| port main, 40,000 | +0.128 +- 0.051 |
| T main, 30,000 | +0.146 +- 0.058 |
| T main, 40,000 | +0.147 +- 0.050 |
| all 140,000 | **+0.127 +- 0.027** |

That is the lead's "handover costs T about 0.15" measured directly, and it is in the endgame; 29% of the
rounds that reach the handover end differently. T and hm4 choose differently at 0.37 decisions a round after
the handover; one move at a time (first order) T's moves gain +0.062 +- 0.030.

**By kind (140,000 rounds, T's move instead of hm4's, hm4 continuing afterwards):**

| class | times a round | each | a round |
|---|---|---|---|
| **hm4 calls Yaniv; T throws a single and takes the open card** | 0.0157 | +3.42 | **+0.054 +- 0.007** |
| hm4 calls; T plays on some other way (deck draw or a set) | 0.053 | -0.19 | -0.010 +- 0.011 |
| hm4 plays on; T calls | 0.013 | +0.25 | +0.003 +- 0.007 |
| same throw, T takes the open card, hm4 the deck | 0.119 | +0.03 | +0.003 +- 0.019 |
| same throw, T the deck, hm4 takes the open card | 0.089 | +0.20 | +0.018 +- 0.012 |
| set against single | 0.027 | +0.26 | +0.007 +- 0.008 |
| a different single | 0.055 | -0.21 | -0.012 +- 0.011 |
| by cards: a hand holds 4 / both <= 3 | 0.196 / 0.176 | -0.01 / +0.36 | -0.001 +- 0.023 / +0.064 +- 0.020 |
| by ply: under 9 / 9-15 / 16 or more | 0.106 / 0.167 / 0.100 | -0.42 / +0.32 / +0.53 | **-0.044 +- 0.015** / +0.053 +- 0.021 / +0.053 +- 0.015 |
| by stock, the first class left out: 36+ / 32-35 / 26-31 / under 26 | 0.155 / 0.108 / 0.076 / 0.019 | | -0.051 +- 0.019 / +0.025 +- 0.017 / +0.026 +- 0.014 / +0.009 +- 0.006 |

The first class showed in the first 60,000 rounds (+0.049 +- 0.011) and repeated in each later sample
(+0.054 +- 0.012, +0.060 +- 0.013). It is the only class past the bar (more than 2 errors and 0.03 a round).

**The class, measured without T** (`swap`, 120,000 rounds; hm4 calls 0.53 times a round; in 0.037 a round of
those calls after the handover the open card is lower than hm4's highest card):

| hm4's call, by the `assaf` input (share of possible opponent hands that would catch it) | times a round | swap instead of call, each | a round |
|---|---|---|---|
| assaf = 0 (the call cannot be caught) | 0.0112 | -1.82 | -0.020 +- 0.002 (the call is right) |
| 0 < assaf < 0.01, swap saves 1 point | 0.0088 | -0.26 | -0.002 +- 0.005 |
| 0 < assaf < 0.01, swap saves 2 or more | 0.0050 | +3.64 | +0.018 +- 0.004 |
| assaf >= 0.01, swap saves 1 point | 0.0071 | +2.68 | +0.019 +- 0.005 |
| assaf >= 0.01, swap saves 2 or more | 0.0044 | +5.77 | +0.026 +- 0.004 |
| **assaf >= 0.01, or assaf > 0 and the swap saves 2 or more** | **0.0166** | **+3.80** | **+0.063 +- 0.007** |
| all with assaf > 0 | 0.0254 | +2.40 | +0.061 +- 0.009 |

Both 60,000-round samples agree (assaf > 0: +0.062 and +0.060). Against 3 opponent cards +0.031 +- 0.006,
against 4 cards +0.031 +- 0.006, against 1-2 cards never (-0.001; assaf is 0 there almost always). In the
5-card phase (port's calls) there is nothing: -0.001 +- 0.002. Why it pays: these calls are caught 12.6% of
the time (385 of 3,044; hm4's calls overall 4.5%), a catch costs 43 points against the swap, and after the
swap hm4 holds 1-4 points fewer and calls on its next turn; the swap loses the round only 8% of the time.

**Rule in words for the lead (not built, hm4 not touched):** after the handover, when Yaniv is legal and the
open card is lower than my highest card, do not call but throw my highest single and take the open card, if
the call can be caught at all (`assaf` > 0) and either `assaf` >= 0.01 or the swap lowers my hand by 2 points
or more. Keep calling when `assaf` = 0. Worth about +0.06 a round against the counter at 0-0 (0.017 times a
round, +3.8 each). Not checked: its effect in matches (a call ends the round now; near 50/100 the score units
may want either), and whether the same holds when the swap is a set rather than a single.

**What is left (+0.06 to +0.07 of the +0.13).** No other class reaches the bar. Hints, each found by looking
through about 80 cells, so each needs its own replication before anyone writes a unit:
- by ply, the rest (calls left out): throws at ply 9 or later +0.061 +- 0.022 for T's moves, throws before ply
  9 -0.046 +- 0.015. hm4 was tuned on rounds dealt with 4 cards (thick stock, few known cards); it is the
  better player in endgames that look like that and the worse one late in a real round. The late gain has no
  single kind: at ply 9+ "take an open 3 where hm4 draws" +0.022 +- 0.008 (0.019 a round), "leave an open 4
  where hm4 takes it" +0.020 +- 0.007 (0.031 a round), "throw a higher single and leave the open card"
  +0.016 +- 0.008; before ply 9 T's "higher single" is wrong, -0.017 +- 0.004.
- "hm4 calls, T plays on with a deck draw" at assaf 0.02-0.04: +0.032 +- 0.015 in the one sample that stored
  assaf (40,000 rounds); at assaf < 0.01: -0.028 +- 0.009 (hm4's call is right).

## 4. Match play

2,000 matches against the counter, seed 921000 for all (`cuts`):

| player | matches | my cuts a match (+-0.015) | of them in rounds ending with a 5-card hand | the counter's cuts |
|---|---|---|---|---|
| T alone | 1281-719 (64.0%) | 0.085 | 9% | 0.091 |
| T, handover both <= 4 | 1373-627 (68.7%) | 0.404 | 3% | 0.126 |
| T, handover mine <= 4 | 1403-597 (70.2%) | 0.413 | 4% | 0.122 |
| T, handover opp <= 4 | 1427-573 (71.4%) | 0.472 | 12% | 0.122 |
| T, handover either <= 4 | 1386-614 (69.3%) | 0.468 | 13% | 0.117 |
| hm4 alone | 1408-592 (70.4%) | 0.463 | 17% | 0.112 |
| port, handover both <= 4 | 1406-594 (70.3%) | 0.447 | 14% | 0.126 |

Together with the working test (1,500 matches): T both 68.2% of 3,500, T opp 70.0%, T either 69.7%
(+-0.8 each; +1.5 to +1.8 points, 1.5 errors: suggestive only).

Reading: T with the "both" handover already gets 0.40 of hm4's 0.46 cuts a match, because most rounds end
after the handover. The missing 0.06 are cuts in rounds that end while a hand still holds 5 cards, and they
need hm4 in charge when **the opponent** is small and may call (94% of cuts come from being called on while
holding the right total, `measure.md` g): "opp" and "either" recover them (0.47, +0.065 +- 0.02), "mine"
does not (0.41). So an earlier handover on the opponent's hand gives more cut plays; on my own hand it does
not. For port this question does not arise: it carries hm4's score units itself (0.447 cuts a match, the same
as hm4 within noise).

## Every measurement, in order

| # | what | why | result |
|---|---|---|---|
| 1 | work, rounds: port with mine / opp / either / both at 3 | question 1 | +1.68 / +1.66 / +1.69 / +1.65 (both: +1.65): no difference |
| 2 | work, rounds: T with mine / both at 3 | question 1 | +1.67 / +1.81 (both: +1.69) |
| 3 | work, rounds and matches: T with opp, T with either | questions 1 and 4 | +1.68, 68.3%; +1.66, 70.2% (both: 67.6%) |
| 4 | paired, port main, 30,000 + 40,000 rounds | questions 1-3 | tables above |
| 5 | paired, T main, 30,000 + 40,000 rounds | questions 1 and 3 | tables above |
| 6 | swap against call at hm4's calls, 2 x 60,000 rounds | make the one class found in 4-5 independent of T | +0.063 +- 0.007 for the rule in words |
| 7 | cuts, 7 players x 2,000 matches | question 4 | table above |

## Could not resolve

- Whether "opp" beats "both" on matches for T: +1.7 points at +-1.1. Needs about 10,000 matches each.
- The rule for the Yaniv swap was read off 0-0 rounds against the counter only; its thresholds (0.01, 2
  points) are hand-picked from one table and confirmed on the second sample as a whole, not cell by cell.
- The late-round throw difference (+0.06 a round for T at ply 9+) has no class I could name at this sample;
  the open-3 / open-4 hints are 2.5-3 errors each among about 80 cells looked at.
- All paired numbers are first order and from 0-0; none says anything about match-score play.
