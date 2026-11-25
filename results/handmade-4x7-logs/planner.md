# planner log (official test: 2000 rounds seed 41000, 800 matches seed 41001, vs counter)

v1 base design: score = -(0.25 H + 0.75 P) + danger shift (4 steps of 0.15 on opp_est <= 13/11/9/7) + call bonus B (3 at 7, +0.8 per point under) + gifts (0.3/pt low top, 3 known-rank match) - 2/card kept; Yaniv: call if assaf < 0.30 - 0.035*pts.
  -> margin +0.26 a round, rounds won 1057/2000 (52.8%), Yaniv calls 1079 of which caught 96 (8.9%); matches 426-374 (53.2%, 95% range 49.8%-56.7%)
v2 call threshold 0.28 - 0.035*pts (0.035 at 7): v1 was caught 8.9%, uniform assaf understates real risk (opponents keep low cards)
  -> margin +0.46 a round, rounds won 1042/2000 (52.1%), Yaniv calls 1033 of which caught 73 (7.1%); matches 437-363 (54.6%, 95% range 51.2%-58.0%)
v3 CARD_COST 2 -> 1 (thought: low pairs can be shed later, so a kept card costs less) -- worse, reverted
  -> margin +0.36 a round, rounds won 1031/2000 (51.5%), Yaniv calls 1022 of which caught 75 (7.3%); matches 435-365 (54.4%, 95% range 50.9%-57.8%)
v4 convex call threshold (0.04 at 7 ... 0.45 at 0): the tolerated risk should grow faster as points fall (cheaper catch, more opponent hands beat); rollouts showed my linear rule losing on low-point non-calls
  -> margin +0.55 a round, rounds won 1046/2000 (52.3%), Yaniv calls 1034 of which caught 70 (6.8%); matches 441-359 (55.1%, 95% range 51.7%-58.5%)
v5 DUP_VALUE 1.5 per card kept beyond the first of its rank (+ deck-draw pairing chance * (1.5 + 0.5*pts)): rollouts showed the net refusing to take an open card that makes a pair/triple; the pair shrinks the hand a turn later, which n_kept does not see
  -> margin +0.79 a round, rounds won 1058/2000 (52.9%), Yaniv calls 1043 of which caught 73 (7.0%); matches 451-349 (56.4%, 95% range 52.9%-59.8%)
v6 known-rank gift scaled by rank: 0.5 + 0.25*pts instead of flat 3 (a low pair is little help to an opponent near the end; rollout showed net keeping a 4 over a 3 to avoid feeding a 4)
  -> margin +0.83 a round, rounds won 1064/2000 (53.2%), Yaniv calls 1050 of which caught 73 (7.0%); matches 464-336 (58.0%, 95% range 54.5%-61.4%)
v7 last danger step 0.15 -> 0.25 (opp est <= 7 calls almost surely, the plan barely matters) -- no change, reverted
  -> margin +0.82 a round, rounds won 1065/2000 (53.2%), Yaniv calls 1052 of which caught 73 (6.9%); matches 463-337 (57.9%, 95% range 54.4%-61.3%)
v8 + catch bonus: when opp est <= 7 the call bonus B counts twice (a low hand also catches the opponent's call: Assaf swing ~40) -- within noise, kept as principled
  -> margin +0.85 a round, rounds won 1065/2000 (53.2%), Yaniv calls 1050 of which caught 72 (6.9%); matches 466-334 (58.2%, 95% range 54.8%-61.6%)
v9 stricter call at exactly 7 (thr .04 -> .025; 7-point calls were caught 11.6%) -- worse, reverted
  -> margin +0.72 a round, rounds won 1047/2000 (52.3%), Yaniv calls 1023 of which caught 71 (6.9%); matches 445-355 (55.6%, 95% range 52.2%-59.0%)
(note: official 2000 rounds is noisy, ~+-0.3; from v8 on I also report a 4000-round check at seed 52000 (rounds only): v8 = +0.29 there, trained net = +1.14 there)
v10 CARD_COST 2 -> 3 and DUP_VALUE flat 1.5 -> 0.15*rank points: rollouts against the trained net's choices showed it throwing LOW pairs early (3 3 before a 9): a low pair clogs two of four slots, a card shed now is worth more than one shed later; high sets still valued
  -> margin +0.89 a round, rounds won 1076/2000 (53.8%), Yaniv calls 1064 of which caught 68 (6.4%); matches 464-336 (58.0%, 95% range 54.5%-61.4%)   [seed 52000 x4000: +0.45, 52.4%]
v11 TAKE_INFO 1.5 flat cost on taking the open card (it becomes known to the opponent; trained net drew from the deck where I took a 4/5) -- worse, reverted
  -> margin +0.64 a round, rounds won 1056/2000 (52.8%), Yaniv calls 1047 of which caught 72 (6.9%); matches 442-358 (55.2%, 95% range 51.8%-58.7%)   [seed 52000 x4000: +0.29]
v12 danger steps moved down to opp_est <= 11/9/7/5 (was 13/11/9/7): an opponent estimated at 10 calls next turn only ~20% of the time, so the present was over-weighted mid-round (trained net took K pairs where I drew)
  -> margin +1.06 a round, rounds won 1088/2000 (54.4%), Yaniv calls 1073 of which caught 66 (6.2%); matches 457-343 (57.1%, 95% range 53.7%-60.5%)   [seed 52000 x4000: +0.40]
(from here also a 12000-round check at seed 53000, rounds only; v12 = +0.66 there, 53.0%)
v13 card cost 4.5 with no danger, falling 1.0 per danger step (trained net dumps low pairs early; near the end hand size stops mattering) -- worse on the big check, reverted (CARD_RELIEF = 0)
  -> margin +0.88 a round, rounds won 1077/2000 (53.8%), Yaniv calls 1063 of which caught 65 (6.1%); matches 461-339 (57.6%, 95% range 54.2%-61.0%)   [seed 52000 x4000: +0.42; seed 53000 x12000: +0.55]
v14 info cost: 0.3 per point of a taken open card when opp_est <= 9 (rollouts: an opponent near a call that sees my card calls safely; the counter's sure-call uses my known cards) -- neutral/positive, kept
  -> margin +1.03 a round, rounds won 1087/2000 (54.4%), Yaniv calls 1073 of which caught 68 (6.3%); matches 444-356 (55.5%, 95% range 52.0%-58.9%)   [seed 53000 x12000: +0.70]
FINAL = v14 (training/yaniv/handmade_designs/planner.py, net at SCRATCH/nets/planner.json, dims [125,126,112,124,118,115,113,133,112,96,1]). Reference at seed 53000 x12000: trained net +1.16 (caught 5.0%), planner +0.70 (caught 6.3%).
Diagnostics used (not fitting): rollout judge (SCRATCH/planner/judge.py, judge2.py) at disagreements vs counter / vs trained net, counter continuation, hidden cards resampled uniformly (biased toward calling, so call verdicts were ignored).
