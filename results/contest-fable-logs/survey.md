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

## 3. The same survey against trained and counter (seed 101, 12,000 rounds each) and the cross-opponent filter

`collect hm5 trained --rounds 12000 --seed 101` (80,793 decisions probed, 1142 s) and `collect hm5 counter
--rounds 12000 --seed 101` (76,156 probed, 703 s). All rules of section 2 re-scored on the three sets, plain and
split by (opp cards), (my cards, opp cards), (opp cards, ply), (my cards, est), (opp cards, est), (opp cards,
points); the scan re-run on the three sets (7,006 cells).

Classes with the SAME sign against all three, >= 3 se and >= 0.01 a round (hm5 | trained | counter, each +- se, n):

| class | hm5 | trained | counter |
|---|---|---|---|
| **triple take, 10s** (I hold T-T/J-J/Q-Q/K-K, the open card is a third; net throws the pair + deck; alt: throw best other group, take it) | +5.72+-1.28 (268), +0.077 a round | +3.51+-1.10 (287), +0.084 | +4.18+-0.99 (329), +0.115 |
| ... opp has 5 cards | +5.49+-1.43 (212) | +3.15+-1.33 (209) | +4.15+-1.12 (261) |
| ... opp has 4 cards | +8.56+-2.94 (48) | +5.38+-2.05 (64) | +5.05+-2.27 (61) |
| triple take, 7-9, opp 5 | +4.45+-1.68 (120) | +4.15+-1.71 (107) | +2.92+-1.48 (133) |
| triple take, 7-9, opp 4 | +0.75+-2.31 (76) | +2.97+-2.30 (90) | -0.55+-1.93 (98) |
| triple take, 7-9, opp 1-3 | -1.73+-2.18 (41) | -3.88+-3.69 (41) | -5.74+-3.21 (34) |
| triple take, 5-6 (all opp counts) | -0.22+-1.32 (176) | +0.35+-1.45 (167) | -1.35+-1.28 (210) |
| triple take, 4s | +3.75+-2.73 (63) | +2.28+-3.08 (50) | +2.82+-2.53 (56) |
| scan cell "[2,1,1,1] open = the pair's rank, opp 5: throw the best single, take" | +6.00+-1.19 (279) | +4.98+-1.19 (247) | +4.28+-1.09 (296) |
| refuse an unpaired open 3, I hold 5, opp 4 | +3.25+-1.04 (394), +0.064 a round | +2.83+-1.24 (238), +0.056 | +4.98+-2.32 (46), +0.019 |
| (same positions) opp 4, ply <= 3 | +3.57+-0.93 (481) | +3.60+-1.12 (288) | +5.95+-2.42 (42) |

The triple take is the only cell the 3-opponent scan prints at all. The open-3 refusal at 5 v 4 has the same sign
on all three but passes 3 se only against hm5 (a class picked after looking; fresh seed below). Consistent with
it, "take an unpaired open 4 at 5 v 4" (net refuses) is the most negative take cell: -2.99+-0.87 | -4.29+-1.00 | -2.76+-1.45.

Everything else (all three opponents agree):
- refuse a paired open card: -2.3 to -5.5 at every value. Refuse an unpaired A/joker -2.2 to -2.5; an unpaired 2
  -1.44+-0.31 | -0.84+-0.35 | -0.59+-0.61. Refuse an unpaired 3 overall +0.29+-0.26 | +0.35+-0.32 | +0.09+-0.49
  (a coin flip); unpaired 4 +0.03+-0.48 | +1.75+-0.65 | -0.46+-0.64 (trained only).
- take an unpaired open card the net refused: 3: -0.66+-0.45 | -1.59+-0.51 | +0.95+-0.64 (counter only: sign
  differs, not used); 4: -1.13 | -1.37 | -0.52; 5-6: -1.9 | -2.1 | -1.6; 7+: -2.0 to -2.5.
- take the open card onto ONE kept twin where the net threw the twin instead: -4.5 | -4.5 | -4.5 (10s), 7-9 -4 to -6.
- throws (highest single for a set, set for a single, second single, breaking a pair, low pairs, same-value single
  by unseen copies): no class positive against two opponents.
- hands of 1-2 cards: no positive class against two opponents (only "refuse open 4 at 2 v 2" +1.59+-0.53 vs hm5).
- part-2 ideas answered by the same play-outs: two-turn ordering (which of two sets first, set or single first):
  no positive cell; keeping a pair for a third from the deck: -0.5 to -4.6 in all 49 cells; keeping/throwing
  aces and jokers: hm5 almost never throws them while it has another single, and forcing it costs 4-12;
  what to leave face up (second single of the same value, least/most live): negative; stock-out/showdown: 0.2%
  of rounds, dropped. The only two-turn idea that pays is the triple take (take now, throw three next turn).

## 4. Fresh-seed check of the candidates (seed 202; `contest_fable_survey:probe_cand`, hero hm5)

`paired("hm5", OPP, "contest_fable_survey:probe_cand", N, 202, workers=2)`, N = 40,000 (hm5), 30,000 (trained,
counter). "triple" = I hold exactly two of the open card's rank and the net does not take it onto them: throw the
best other group and take it. "refuse" = the net takes an unpaired open 3 / 4: same throw, deck.

| class | hm5 (40,000) | trained (30,000) | counter (30,000) |
|---|---|---|---|
| triple 10s, opp 5 | +3.09+-1.00 (n 389) | +4.07+-0.87 (488) | +3.98+-0.72 (637) |
| triple 10s, opp 4 | +1.07+-1.77 (104) | +4.07+-1.51 (139) | +3.65+-1.55 (138) |
| triple 10s, opp 1-3 | +1.36+-4.35 (25) | -0.21+-3.29 (39) | -2.36+-3.12 (28) |
| triple 7-9, opp 5 | +5.29+-1.27 (222) | +3.90+-1.09 (269) | +2.63+-0.93 (360) |
| triple 7-9, opp 4 | +2.26+-1.59 (143) | +2.86+-1.42 (195) | +2.23+-1.45 (172) |
| triple 7-9, opp 1-3 | -3.74+-2.05 (69) | -2.49+-2.27 (70) | -2.71+-2.82 (61) |
| triple 5-6, opp 5 | +4.57+-1.62 (117) | +3.52+-1.57 (124) | +1.93+-1.14 (159) |
| triple 5-6, opp 4 | +0.33+-1.46 (117) | -0.82+-1.32 (163) | -0.68+-1.22 (181) |
| triple 5-6, opp 1-3 | +0.28+-1.85 (95) | -2.59+-1.55 (106) | -0.81+-2.11 (86) |
| triple <= 4, opp 5 | +2.73+-2.15 (79) | +4.13+-2.57 (68) | +1.22+-1.79 (79) |
| triple <= 4, opp 4 | +5.76+-2.01 (83) | +2.07+-2.04 (73) | +0.95+-1.49 (94) |
| triple <= 4, opp 1-3 | -6.53+-1.69 (78) | -4.65+-1.42 (94) | -3.74+-2.32 (57) |
| refuse 3, me 5, opp 4 | +1.29+-0.74 (785) | +0.89+-0.82 (606) | +1.39+-2.65 (89) |
| refuse 3, me 5, opp 1-3 | +0.00+-0.97 | +0.80+-0.97 | +2.14+-1.37 |
| refuse 3, me 3-4 | -0.82+-0.33 (3075) | -0.54+-0.33 | -2.45+-0.51 |
| refuse 3, me 2 | +0.49+-0.25 (3263) | +0.12+-0.32 | -0.39+-0.47 |
| refuse 3, me 1 | -1.24+-0.54 | +0.76+-0.68 | -1.49+-1.05 |
| refuse 4, me 2 | +0.41+-0.40 (1151) | +0.69+-0.54 (804) | +0.09+-0.52 (942) |
| refuse 4, me 1 | +1.99+-0.78 (222) | +0.87+-0.96 | +0.57+-1.09 |
| refuse 4, me 3-4 | -0.40+-0.72 | -0.67+-0.68 | -2.83+-0.66 |

- **Triple take: confirmed.** Against 5 cards it pays at every rank worth 5+ (and probably at 4 or less, too few
  cases); against 4 cards for 7+ (seven of the eight seed-101/202 cells positive); against 1-3 cards it loses.
  Summed over the cells the design uses (10s and 7-9 at opp 4-5, 5-6 at opp 5): +0.083 a round vs hm5, +0.153 vs
  trained, +0.156 vs counter on seed 202.
- **Refusing an open 3 at 5 v 4: NOT confirmed** (+3.25 on seed 101 -> +1.29+-0.74 on seed 202; trained +2.83 ->
  +0.89+-0.82). Selection luck from the wide look; not built. The same for "refuse an open 4 with 2 cards"
  (+1.86 -> +0.41+-0.40).

## 5. Design: `py:contest_fable_survey_design.build` (hm5 + 9 hidden units = 567)

One layer-1 unit per rank worth 5 or more (5,6,7,8,9,T,J,Q,K):
`trip_r = relu(3 open_r + 2 take_open + 4 hand_r - 10 throw_r + gate - 7)`, gate = [opp has 4 or 5 cards] for
ranks worth 7+, [opp has 5 cards] for 5s and 6s; output weight +40 each (hm5's own count puts the pair-throw
line ahead by up to about 3 x rank + 5 points, so 40 makes the take win everywhere in the class; the other terms
still choose which other card to throw). 1 when I hold two r, take the open r, do not throw r.

Checks:
- `python -m training.yaniv.contest_fable_survey_check 'py:contest_fable_survey_design.build' 3000 3`: 164 of
  42,024 decisions differ from hm5; all 164 are "holds 2 (or 3) of the open rank, opponent has 4-5 cards, hm5
  throws the pair and draws the deck, design throws another card and takes the open card". With TRIP=0: 0 differ.
  (A first version of the unit, `open_r + 2 take + 4 hand_r - 5`, also fired on a held triple with any open card
  taken: 114 differences of which 38 unintended; fixed before any duel.)
- Kit.trace on 19,540 move rows (1,500 rounds, seed 4): the triple unit is on in 75 rows, 0 mismatches with the
  stated condition; e.g. hand 4-8-Q-Q-joker, open Q, opp 5: design throws the 8 and takes the Q (OUT 19.7), hm5's
  move Q-Q + deck scores -17.0.
