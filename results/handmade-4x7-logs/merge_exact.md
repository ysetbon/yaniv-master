# merge_exact: variant log
Dev test = 12,000 rounds seed 52000 + 3,000 matches seed 52001 (--workers 3). Official = 2,000 rounds seed 41000 + 800 matches seed 41001.
Base = clone.py BEST copied and reorganised (checked: 1236/1236 identical moves to nets/clone.json).

References (dev):
- trained net: margin +1.39 a round, rounds won 6513/12000 (54.3%), Yaniv calls 6173 of which caught 307 (5.0%); matches 1816-1184 (60.5%, 95% range 58.8%-62.3%)
- clone.py BEST (= merge_exact v0): margin +0.89 a round, rounds won 6491/12000 (54.1%), Yaniv calls 6367 of which caught 396 (6.2%); matches 1754-1246 (58.5%, 95% range 56.7%-60.2%)

Variants:
1. y_reader: Yaniv rule = reader.py's (assaf < 0.015 + relu(0.025 - 0.00125 ply) + 0.002 (gap-5, gap capped 15) + 0.005 relu(pts-5), or sure), throws = clone. Why: reader's rule was caught 4.0% vs clone's 6.7% on the official test.
   dev: margin +0.97 a round, rounds won 6251/12000 (52.1%), Yaniv calls 5879 of which caught 256 (4.4%); matches 1759-1241 (58.6%, 95% range 56.9%-60.4%)
2. y_c1: Yaniv rule = caller.py's C1 (opp_est - my_pts > 0.4 pts + 1.3 + 10 assaf/40.., or sure), throws = clone. Why: opp_est - my_pts separates good and bad calls better than assaf (caller's diagnostics).
   dev: margin +0.94 a round, rounds won 6370/12000 (53.1%), Yaniv calls 6099 of which caught 303 (5.0%); matches 1773-1227 (59.1%, 95% range 57.3%-60.8%)
3. h3: throws + heavy-card charge 3 x (1 - w_b) per heavy card held after the move (A/joker 0, a 2 half; deck card heavy with chance ~1.07 - unseen_low; an open card taken onto a kept card of its rank not counted, since it leaves with that set next turn); Yaniv = clone's. Why (reader): a hand with fewer cards stays smaller for the rest of the round, so throw pairs of 3+ rather than a higher single; weighted by (1 - w) because it pays only on future turns. 3 = about the points of a low card kept to the end (off-policy check on reader's trained-net positions: pair-of-3+ throw rates now 0.9-1.0 like the trained net, clone 0.07 for 33 vs a 6+; the pairing-take exemption keeps triple/pair takes near the trained net's).
   dev: margin +1.18 a round, rounds won 6628/12000 (55.2%), Yaniv calls 6490 of which caught 397 (6.1%); matches 1768-1232 (58.9%, 95% range 57.2%-60.7%)
(resumed by a second agent after a container restart; nets/merge_exact_h3p.json is byte-identical to h3.json, so not a separate variant. BEST set to h3 in the file.)
4. h3_reader: h3 throws + reader's Yaniv rule. Why: compare the call rules again now that the throws include the heavy-card charge (off-policy: h3 with clone's rule calls far more than the trained net at assaf 0.01-0.05 late in the round).
   dev: margin +1.26 a round, rounds won 6400/12000 (53.3%), Yaniv calls 6017 of which caught 260 (4.3%); matches 1812-1188 (60.4%, 95% range 58.6%-62.1%)
5. h3_c1: h3 throws + caller's C1. Same reason.
   dev: margin +1.26 a round, rounds won 6515/12000 (54.3%), Yaniv calls 6227 of which caught 306 (4.9%); matches 1801-1199 (60.0%, 95% range 58.3%-61.8%)
   Diagnostic before 6 (SCRATCH/merge_exact/rules.py on 25k Yaniv-legal non-sure turns of h3_c1 and of h3_reader, true hidden hands): the two rules agree on 93% of spots. Only-C1 calls (6%): true caught 13%, round margin +4.1 when called (h3_c1) vs +4.8 when not (h3_reader); only-reader calls (1.5%, all at 6-7 pts): caught 21%, +1.6 vs +1.4.
6. h3_blend: call only if BOTH reader's rule and C1 agree (or sure). Why: each screens out the other's riskiest calls (above); playing on keeps the Assaf-trap value of the catch bonus.
   dev: margin +1.24 a round, rounds won 6371/12000 (53.1%), Yaniv calls 5966 of which caught 246 (4.1%); matches 1825-1175 (60.8%, 95% range 59.1%-62.6%)
   -> reader / C1 / blend all +1.24..+1.26 on h3 (all ~+0.07 over clone's rule): the disagreement spots are near break-even. Kept C1 (simplest, most rounds won).
7. c1_cv4: h3 + C1 + 3 for a move that makes Yaniv sure next turn while the opponent holds 4 cards (reader: +0.08 official on its base). Why: a call vs a 4-card opponent is almost never caught.
   dev: margin +1.16 a round, rounds won 6505/12000 (54.2%), Yaniv calls 6262 of which caught 340 (5.4%); matches 1823-1177 (60.8%, 95% range 59.0%-62.5%)
   -> worse (-0.10): more caught calls; clone's exact E[next throw] + catch bonus already value low hands, the bonus double-counts. Dropped.
   Diagnostic before 8 (h3_c1's own calls, true hidden hands): caught rate by C1 margin (opp_est - 1.4 pts - 10 assaf - 1.3) is 14% at 0-1 points of margin with the round worth ~+4 when called, ~10% at 1-2 (+6), and it does NOT rise with the ply (opp_est already leans low), so no ply term; playing on in such spots averaged ~+4.8 (h3_reader data).
8. c1_tight: C1 bias 1.3 -> 2.3 (one point more margin needed). Why: the margin-0-1 calls are break-even or worse once the catch bonus gives waiting its Assaf-trap value.
   dev: margin +1.30 a round, rounds won 6371/12000 (53.1%), Yaniv calls 5932 of which caught 214 (3.6%); matches 1808-1192 (60.3%, 95% range 58.5%-62.0%)
9. c1_score: h3 + C1 (bias 1.3) + scorer.add_score_units copied (open-card cut with threat, deck cut, assaf cut +400 on the Yaniv row: row is -200 + 400 ycall here, throws within ~-60..+20, so +400 forces the call as in scorer). Rounds identical (0-0 starts), matches only.
   dev: margin +1.26 a round, rounds won 6515/12000 (54.3%), Yaniv calls 6227 of which caught 306 (4.9%); matches 2048-952 (68.3%, 95% range 66.6%-69.9%)
   -> matches 60.0% -> 68.3% (trained net 60.5%). Kept.
10. FINAL = BEST: h3 throws + C1 at bias 2.3 + scorer's score units (python -m training.yaniv.handmade_designs.merge_exact -> nets/merge_exact.json, dims [125,296,227,221,348,197,195,192,165,115,58,1]).
   dev: margin +1.30 a round, rounds won 6371/12000 (53.1%), Yaniv calls 5932 of which caught 214 (3.6%); matches 2062-938 (68.7%, 95% range 67.1%-70.4%)
   official: margin +1.48 a round, rounds won 1076/2000 (53.8%), Yaniv calls 997 of which caught 36 (3.6%); matches 562-238 (70.2%, 95% range 67.0%-73.3%)
Not tested, with reasons: open-card tell and planner's info cost (h3's unpaired-open take rates already match the trained net's: 0.80/0.38/0.10 at 3/4/5 vs 0.90/0.35/0.07); pairing/triple take bonuses (pairing takes 0.90-0.96 vs trained 0.85-0.95 already); a ply term in the call rule (C1's true catch rate does not rise with ply); match-point / desperate / bust score terms (scorer measured no effect).
