# Thread `twin`: keep the high single whose twin the opponent will throw?

The owner's idea: early in a round players throw their large cards, so when I hold two or more HIGH SINGLES it
may pay to keep the one whose twin the opponent is likely to throw, take the twin from the open pile next turn
and throw the pair the turn after, instead of always throwing my highest single.

Files: `training/yaniv/contest_fable_twin.py` (probes, mechanism count, designs), raw tables
`results/contest-fable-logs/twin_*.txt`, records `twin_*.pkl`. Nothing is fitted or searched; every number below
is a count from played rounds (real game, 5 cards, Yaniv at 7).

## 0. Definitions (everything a 125-input ReLU net can compute)

- high single: a rank I hold exactly once, worth >= 5 points (classes 5-7, 8-9, 10 = T/J/Q/K). No runs exist in
  this game, so a single can only ever leave as part of a set of its own rank.
- unseen copies of rank X (`unseen_X` x 4) = 4 - mine - in the discard pile (`gone_X` + the open card) - known in
  the opponent's hand (`known_X`).
- the situation: the opponent holds 5 or 4 cards, my move throws ONE card of a rank I hold once (>= 5 points),
  and I hold at least one more such single of another rank. The choice: which one goes first.
- what the current net (hm5 = contest_fable.build()) does there: the throw part of its score is, per move,
  -(points kept) - LIVE x (unseen copies of the thrown rank) [LIVE = 1 against 4-5 cards] - FEED (1 + 0.5 x points
  if the opponent is KNOWN to hold the thrown rank) + small plan terms; so between two high singles it throws the
  one with the larger (points - unseen copies): it already keeps the more 'live' rank at equal points, and keeps
  a rank the opponent is known to hold unless the other single is ~6+ points heavier.

## 1. The mechanism: how often does the twin come? (`contest_fable_twin mech`)

Command: `python -m training.yaniv.contest_fable_twin mech hm5 OPP --rounds 40000 --seed 11 --workers 2`
(tables: twin_mech_hm5.txt, twin_mech_counter.txt, twin_mech_trained.txt). One record per hero turn and per high
single held AFTER the hero's move, opponent on 4-5 cards. opp1 = the opponent's very next throw is of that rank
(so its top card is my twin and I can take it); opp2 = next throw or the one after, while I still hold the single.

Result (hm5 as the hero; 40,000 rounds each; 6.0-6.5 held high singles a round counted once per hero turn):

P(the opponent's very next throw is my single's rank), opponent NOT known to hold it, unseen copies u:

| opp cards, my single | u=3 vs hm5 | u=3 vs counter | u=3 vs trained | u=2 (hm5 / counter / trained) | u=1 |
|---|---|---|---|---|---|
| 5 cards, ten-card (T/J/Q/K) | 11.7% | 12.7% | 12.0% | 3.7 / 3.4 / 3.4% | 2.0 / 2.1 / 2.2% |
| 5 cards, 8-9 | 9.0% | 10.3% | 9.2% | 3.5 / 3.8 / 2.9% | 1.0 / 1.6 / 1.3% |
| 5 cards, 5-7 | 5.8% | 6.7% | 6.1% | 4.0 / 2.9 / 2.9% | 1.3 / 1.1 / 1.2% |
| 4 cards, ten-card | 11.8% | 11.1% | 12.4% | 4.1 / 4.1 / 4.1% | 1.9 / 2.2 / 2.0% |
| 4 cards, 8-9 | 10.2% | 10.9% | 10.6% | 4.2 / 4.5 / 4.2% | 1.6 / 1.9 / 1.8% |
| 4 cards, 5-7 | 8.4% | 9.9% | 8.8% | 4.1 / 3.7 / 3.7% | 1.7 / 1.5 / 2.1% |

(standard errors 0.1-0.4 points of a percent.) Within TWO opponent turns (opp2), u=3: ten-card 15-16%, 8-9
15-17%, 5-7 11-13% against 5 cards. By exact points against 5 cards (all u, vs hm5): 5: 4.9%, 6: 6.4%, 7: 7.6%,
8: 8.6%, 9: 9.8%, ten-cards: 11.1%. By ply (ten-card, u=3, vs hm5): ply 0-1 13.6%, 2-3 11.5%, 4-5 7.9%, 6+ 5.5-7.5%.
The round ends by the opponent's call instead: 0.3-0.6% against 5 cards, 2.5-3.5% against 4.

- When the opponent is KNOWN to hold my single's rank (it took one from the open pile): it throws that rank on
  its next turn 100% (hm5), 100% for 8+ / 50% for 5-7 (counter), 99-100% (trained). Only 0.03-0.07 such holdings a
  round. The net already keeps such a single (FEED charges -(1 + 0.5 x points) for throwing it).
- When the twin does come (0.53-0.60 times a round over all my high singles): hm5 takes it 98% of the time against
  5 cards (91-95% against 4), and after taking it throws the pair on its very next turn 99.5% (ten-cards), 99%
  (8-9), 97-98% (5-7) against 5 cards; 96 / 95 / 90% against 4 cards. So 'do I really get to throw the pair': yes.
- What the mechanism can be worth: seen copies matter a lot (u=3 -> 12%, u=2 -> 3.5%: once the opponent has thrown
  one copy it does not hold another), but POINTS hardly matter among the high cards (ten-card 12% vs a 9 at 10%,
  an 8 at 9%, a 7 at 8%). Choosing between two u=3 high singles changes the twin's chance by 1-4 points of a
  percent; with a twin worth roughly one heavy card of the race (~5 points) that is 0.05-0.2 points, less than
  the 1-5 points of extra exposure from keeping the heavier card. The only lever is unseen copies, and the net's
  LIVE term already uses it (8 points of a percent per copy x ~5-8 points ~ 0.4-0.7 a copy, plus my own deck
  chance to pair: LIVE = 1 a copy is the right size).
- Checked a suspicious cell ('ply 2-3, u=2: 17.9%' in the by-ply table): that table included known cards; with
  known excluded and the seen copy split by who threw it (16,000 rounds, seed 21, hm5 vs hm5, ply 2-3, ten-cards):
  opponent threw the seen copy 1.5%, I threw it 0.4%, it was the first open card 5.8%, nothing seen 10.5%.

## 2. Which high single to throw: both moves played out (`contest_fable_twin which`)

Command: `python -m training.yaniv.contest_fable_twin which hm5 OPP --rounds N --seed 11 --workers 2`
(`contest_fable_lab.paired` with `contest_fable_twin:probe_which`; tables twin_which_OPP.txt).
At every hero decision in the situation of section 0, every 'throw the other high single, same draw' is played
out to the end on the same hidden cards (hero's own policy afterwards). gain = alternative - played, in points of
margin. Notation: the net PLAYED rank c; the alternative throws rank a and KEEPS c. dp = points(a) - points(c),
du = unseen(a) - unseen(c). 'clean' = the open card is neither rank.
