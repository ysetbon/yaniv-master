# reader: variant log (official test unless noted: 2,000 rounds seed 41000, 800 matches seed 41001)

v1 (reader.py first version): value = -(points after move) - 5 per heavy card (not A/2/joker) after the move (3 when opp has 1-2 cards) + 7 for taking an open card that pairs a kept card - 1 per point above 3 of an unpaired open card taken; Yaniv when sure, or assaf < T = 0.04-0.002*ply (>=0) + 0.002*(gap-5 capped) + 0.005*(pts-5)+. Why: the rules read off the trained net (reader-findings.md).
  margin +1.02 a round, rounds won 1011/2000 (50.5%), Yaniv calls 919 of which caught 30 (3.3%); matches 417-383 (52.1%, 95% range 48.7%-55.6%)
v2: Yaniv threshold raised to match the trained net's call rates measured on my own positions (T = 0.012 + relu(0.05 - 0.0025*ply) + same gap/6-7pt terms; v1 was relu(0.04-0.002*ply)+...). Why: v1 called less than the trained net at assaf 0.03-0.1 and late plies, and lost the calling race (counter called first in 55% of rounds).
  margin +0.80 a round, rounds won 1032/2000 (51.6%), Yaniv calls 981 of which caught 52 (5.3%); matches 439-361 (54.9%, 95% range 51.4%-58.3%)  -> worse margin, reverted to v1 threshold
v3 (= v1 Yaniv + pair changes): a 2 counts half a heavy card (trained net throws 22 over a single 3 96%, over a 4 72%, over 5+ ~30%); pairing-open bonus 7 -> 9 (trained takes pairing 8-10s ~85-90%, v1 only 59-81%); +5 more when the open card makes a triple (14 units triple_r = relu(open_r + 4*hand_r + take_open - 4*throw_r - 3.5), x2; trained takes triple-making cards 32-92%, v1 never).
  margin +1.21 a round, rounds won 1042/2000 (52.1%), Yaniv calls 962 of which caught 34 (3.5%); matches 459-341 (57.4%, 95% range 53.9%-60.8%)
v4 (= v3 + ...): +3 for a callable_next move while the opponent holds 4 cards (unit relu(callable_next + opp_has_4 - 1)). Why: trained net picks an available callable_next move 77% when opp has 4 cards (a call then is almost never caught) but ~50% at 1-2 cards; v3 had it the other way round (33% at 4 cards, 80% at 1).
  margin +1.29 a round, rounds won 1045/2000 (52.2%), Yaniv calls 965 of which caught 33 (3.4%); matches 454-346 (56.8%, 95% range 53.3%-60.1%)
v5 (= v4 + ...): unpaired-open charge 1.0 -> 1.5 per point above 3. Why: v3 took unpaired open 4s 63% vs trained 34%.
  margin +1.28 a round, rounds won 1043/2000 (52.1%), Yaniv calls 958 of which caught 31 (3.2%); matches 466-334 (58.2%, 95% range 54.8%-61.6%)  -> neutral, kept
v6 (= v5 + ...): Yaniv threshold early part 0.04 -> 0.03. Why: v2 (calling more) lost margin; test the other direction.
  margin +1.14 a round, rounds won 1006/2000 (50.3%), Yaniv calls 897 of which caught 20 (2.2%); matches 454-346 (56.8%, 95% range 53.3%-60.1%)  -> worse, reverted to 0.04
v7 (= v5 + ...): Yaniv threshold flatter in ply: T = 0.015 + relu(0.025 - 0.00125*ply) (+ same gap/6-7pt terms); v5 was relu(0.04 - 0.002*ply). Same 0.04 at ply 0, but 0.015 instead of 0 from ply 20. Why: on my own positions the trained net calls 51% at ply 14+ with assaf 0.005-0.03 where v5 called 16%, while early (ply<=9, assaf<0.03) v5 already called more than it (99% vs 91%).
  margin +1.32 a round, rounds won 1062/2000 (53.1%), Yaniv calls 995 of which caught 40 (4.0%); matches 469-331 (58.6%, 95% range 55.2%-62.0%)  -> kept
v8 (= v7 + ...): gap term 0.002 -> 0.004 per point of (opp_est - my pts) around 5, cap 15 -> 10. Why: trained net's early call rate at assaf<0.03 rises from .69 (gap 4-6) to .96 (gap 8-10); v7 called .90-1.0 regardless.
  margin +1.30 a round, rounds won 1058/2000 (52.9%), Yaniv calls 989 of which caught 38 (3.8%); matches 467-333 (58.4%, 95% range 54.9%-61.7%)  -> neutral, reverted to 0.002 / 15 (simpler, slightly better)
v9 (= v7 + ...): -1 for leaving a card worth <= 4 face up (low_top). Why: the counter takes unpaired open cards <= 4 95% of the time, so a low top card feeds it; small so low pairs are still thrown.
  margin +1.33 a round, rounds won 1060/2000 (53.0%), Yaniv calls 991 of which caught 40 (4.0%); matches 468-332 (58.5%, 95% range 55.1%-61.9%)  -> neutral, kept (FINAL)
info: v9 head-to-head vs the trained net (2,000 rounds seed 41000): margin -0.10 a round, rounds won 948/2000 (47.4%).
