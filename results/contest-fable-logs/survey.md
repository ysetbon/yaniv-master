# Thread `survey`: where does hm5 throw or draw wrongly?

Files: `training/yaniv/contest_fable_survey.py` (probe that offers every legal throw x draw and plays each out),
`contest_fable_survey_rules.py` (rules stated a priori, scored from those play-outs),
`contest_fable_survey_facts.py` (plain counts). Nothing is fitted; play-outs and counts only.

## 0. Base facts (plain play)

`python -m training.yaniv.contest_fable_survey_facts hm5 hm5 3000 7`

- Rounds end by Yaniv 2993 / 3000; showdown 7 (0.2%); the stock never emptied (0 / 3000). Mean 13.0 throws a round;
  11% of rounds reach ply 20, 0.9% ply 30, 0.3% ply 40. **Stock-out and showdown play cannot pay: dropped.**
- What hm5 takes from the open pile (as the second player): unpaired 0/1/2-point cards 100%; unpaired 3s 67%
  (31% on its first turn); unpaired 4s 19% (0% first turn); unpaired 5s 2%; unpaired 6+ never. A card of a rank it
  holds: 88-99% at every value. So against a hand-written net a 3-or-less top card is taken, a 4 mostly is not,
  and any card it can pair is taken.

## 1. The survey instrument

`python -m training.yaniv.contest_fable_survey collect hm5 OPP --rounds N --seed 101 --out FILE.pkl`: hero hm5,
rounds from 0-0; at every non-Yaniv decision of the hero EVERY distinct legal (rank, count, draw) is played to the
end on the same hidden cards (hero's own policy afterwards). A rule "in situation S play M(S)" is then scored as
margin(M) - margin(net's move). Rules are written before looking at their outcome; classes that look good on seed
101 are re-checked on a fresh seed (202) before anything is built.

## 2. Survey against hm5 (hero hm5, opponent hm5, 20,000 rounds, seed 101)

`collect hm5 hm5 --rounds 20000 --seed 101`: 140,826 hero decisions, 130,876 non-Yaniv decisions probed, every
alternative played out (2237 s). Scored with `python -m training.yaniv.contest_fable_survey_rules RULES --all
hm5=all_hm5.pkl` ("each" = points gained by the alternative per occurrence, +- one standard error; "rd" = per round).

Rules, stated before scoring (alternative keeps the net's draw unless it says otherwise):

| rule | what the alternative is | result against hm5 (20,000 rounds) |
|---|---|---|
| swap_draw: refuse | same throw, deck instead of the open card the net took | pairs: -2.6 to -5.5 each, every value (n 215-6318). Unpaired A/joker -2.51+-0.62, 2 -1.44+-0.31; **3: +0.29+-0.26 (n 4003)**; 4: +0.03+-0.48 (n 1153); 5-6: +1.85+-0.99 (n 178) |
| swap_draw: take | same throw, open card instead of the deck | unpaired 3: -0.66+-0.45 (n 2085); 4: -1.13+-0.27; 5-6: -1.91+-0.15; 7-9: -2.46+-0.11; 10: -2.40+-0.09 |
| hi_single | highest unpaired single instead of the set / lower single the net threw | negative in all 60 classes (e.g. net pair of 10s -> a 10 single -2.08+-0.27; pair of 2s -> 7-9 single -2.29+-0.74, -> 5-6 single -0.85+-0.71, -> 4 -0.70+-0.70) |
| greedy | the group worth the most points | negative everywhere (same data as hi_single) |
| set_instead | net threw a single; throw a held set | only A/joker pairs and 2-2 are ever kept by hm5: throwing AA instead costs 5-8 each; 2-2 instead of a 10 single -2.92+-1.07, of a 7-9 -2.52+-0.90 |
| single2 | second-highest single instead of the highest | negative in all 24 classes; same-value second single -0.93+-0.15 (10s), -0.97+-0.22 (7-9), -0.51+-0.39 (5-6) |
| break_pair | one card of a held pair | -4 to -14 each, all 30 classes |
| least_live / most_live | another single of the same points with fewer / more unseen copies | -4.1+-0.4 / -0.9+-0.4 |
| low_single | net threw a single worth <= 3: throw another single | negative in all classes |
| pair_take | the open card's rank is in my hand but the net does not take it onto a kept twin: throw the best other group, take it | onto ONE card (net threw the twin): -1.8 to -5.0. **Onto a PAIR (net threw the pair, drew the deck): 10s +5.72+-1.28 (n 268, +0.077 a round); 7-9 +2.19+-1.20 (n 237); 5-6 -0.22+-1.32 (n 176); 4s +3.75+-2.73 (n 63)** |

Keeping a pair to wait for the third card (net set -> hi single), split by opponent cards and unseen copies of the
pair's rank: negative in all 49 cells with n >= 40 (-0.5 to -4.6). Same as the last experiment found.

Splits of every rule by (my cards, opp cards), (opp cards, ply band), (my cards, est), (opp cards, est),
(opp cards, my points), (opp cards, known cards): the ONLY positive classes at 3 standard errors were
- **triple take**: I hold a pair, the open card is a third of that rank, net throws the pair and draws the deck.
  By opponent cards: 10s opp5 +5.49+-1.43 (n 212), opp4 +8.56+-2.94 (n 48); 7-9 opp5 +4.45+-1.68 (n 120), opp4
  +0.75+-2.31 (n 76), opp1-3 -1.73+-2.18 (n 41); 5-6 opp5 +1.52+-2.40, opp4 +1.67+-1.98, opp1-3 -6.05+-2.40.
- **refuse an unpaired open 3** when I hold 5 cards and the opponent 4: +3.25+-1.04 (n 394); the same positions as
  "opp4, ply <= 3" +3.57+-0.93 (n 481) (the opponent threw a pair on its first turn). ~0.07 a round. 3.8 standard
  errors in a look at ~500 cells: needs the other opponents and a fresh seed.
- refuse an unpaired open 4 at 2 cards against 2: +1.59+-0.53 (n 290), 0.02 a round.

Systematic scan (`contest_fable_survey_scan`): situation = hand shape (group sizes in points order) x relation of
the open card to the hand x opponent cards (5 / 4 / 1-3) x the net's move; alternative = (which group, how many, draw).
6,689 cells; those with mean >= 2 se and >= 0.004 a round: 12, of which the triple take is the top (+6.00+-1.19,
n 279 at opp5; +6.62+-2.19 and +4.77+-2.19 at opp4), then the open-3 refusal at 5 v 4 (+4.94+-1.57, n 161 in the
five-singles shape), "five singles, open 4, opp 5: throw the second-highest single" (+2.68+-1.08, n 247), "two
singles, open 4, opp 1-3: refuse it" (+1.68+-0.66, n 364). About a dozen 2-se cells are expected by chance among
that many. No two-turn ordering cell ([2,2,1], [2,1,1,1] which pair first, pair or single first) is positive.

**Reading: against itself hm5's throw and draw are locally right nearly everywhere; the one clear hole is the triple take.**
