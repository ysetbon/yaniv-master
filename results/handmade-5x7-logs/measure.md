# Real-game measurements (5 cards, Yaniv at 7)

**Status: all parts (a-g) final.** One script: `training/yaniv/handmade5_measure.py` (`part1`, `e`, `f`, `g`); own seeds 900000-905999.

## Summary for designers (details and sample sizes in the sections)

| what | real game, measured | 4-card experiment said |
|---|---|---|
| P(counter calls next turn) by `opp_est` cuts 5/8/11/14/17, counter on 1-4 cards | 0.55 / 0.32 / 0.18 / 0.10 / 0.05 / 0.01 | 0.55 / 0.30 / 0.18 / 0.08 / 0.04 / 0.01 |
| ... by its card count at equal est | est 8-11: 0.11 (4 cards) 0.18 (3) 0.29 (2); est 5-8: 0.21 (3) 0.34 (2) 0.70 (1) | no dependence |
| ... counter on 5 cards | 0.006 overall; 0.08 at est 11-14, 0.03 at 14-17, 0 above | - |
| real catch risk of my call by `assaf` bin (0 / 0-.01 / .01-.02 / .02-.04 / .04-.08 / .08-.15 / .15+) | 0 / 0.05 / 0.12 / 0.16 / 0.21-0.27 / 0.31 / 0.60 | 3-8x assaf at the low end |
| counter's total when it calls, by its cards 5/4/3/2/1 | 6.3 / 6.0 / 5.5 / 4.7 / 3.0-3.7 (23% at 7, 25% at 6, 20% at 5) | - |
| P(hm4's hand <= counter's total when the counter calls) | 10.1% (counter in the same seat: 7.0%) | - |
| rounds reaching both <= 4 / my decisions after it | 81% / 48% | - |
| calls made while a hand still holds 5 cards | 15.8% of all calls | - |
| low pair thrown with a much higher single in hand (counter / hm4 / trained) | 2s: 3% / 9% / 68%; 3s: 5% / 85% / 96%; 4s: 9% / 98% / 98% | - |
| unpaired open card taken (counter / hm4 / trained) | a 3: 99 / 85 / 93%; a 4: 79 / 1 / 23%; a 5: 41 / 0 / 7% | - |
| one card fewer for me, in result points at equal held points | 3.7 at ply 4, 2.7 at ply 8, 2.0 at ply 12 (rollout: +2.6) | - |
| the same in held points, against 5 / 4 / 3 / 2 opponent cards (ply 8) | about 7.5 / 5.5 / 4 / 2.5 | card charge 5, 3 against 1-2 cards |
| match-score units | 70.0% of matches with, 62.0% without (3,000 matches each); 0.48 cuts a match against 0.10 | carried the 4-card net's matches |


Script: `python -m training.yaniv.handmade5_measure part1 --rounds 20000` (seeds 900000+). Rounds start from 0-0, seats and the lead alternate. hm4 = the 4-card hand-written net playing the whole real game unchanged. `opp_est` and `assaf` are the net's inputs 94 and 95 (x40 for opp_est) at my decision, before my throw. Games: hm4 vs counter 20,000 rounds, counter vs counter 20,000 (both seats counted as 'me'), trained vs counter 10,000.

Marks: `*` fewer than 100 samples (rough: +-0.1 or worse on a rate), `**` fewer than 30 (do not use).

## a. P(the counter calls Yaniv on its next turn) by its card count and `opp_est`

**Reading (a).** Pooled over the counter holding 1-4 cards the curve is the 4-card one again: **0.55 / 0.32 / 0.18 / 0.10 / 0.05 / 0.01** at cuts 5/8/11/14/17 (me = hm4; 0.67 / 0.37 / 0.20 / 0.10 / 0.05 / 0.01 when the counter is "me"). **But there IS a card-count effect at equal `opp_est`**, about x1.5 per card fewer: est 8-11 -> 0.11 (4 cards), 0.18 (3), 0.29 (2); est 5-8 -> 0.21 (3), 0.34 (2), 0.70 (1, n=83). The pooled curve works only because card count and est move together. A counter holding 5 cards calls next turn 0.6% of the time (0.08 at est 11-14, 0.03 at 14-17, 0.00 above 17, which is 93% of the 5-card cases): treat it as zero unless est < 14.
Late in the round (ply 20+) the same est means fewer calls (0.22 against 0.32-0.37 in the 5-8 bin). The counter CAN call (holds <= 7) about 1.5x as often as it does call when est < 11.

**me = hm4, opponent = counter** (126,566 of my throws that the opponent answered). Cell: P(opponent calls on its next turn) (samples); `*` n<100, `**` n<30.

| opp cards \ opp_est | <5 | 5-8 | 8-11 | 11-14 | 14-17 | 17+ | any |
|---|---|---|---|---|---|---|---|
| 5 | - | - | - | 0.08 (418) | 0.03 (2997) | 0.00 (43345) | 0.01 (46760) |
| 4 | - | 0.00 (1)** | 0.11 (1020) | 0.09 (10226) | 0.05 (13653) | 0.01 (19364) | 0.04 (44264) |
| 3 | 0.33 (3)** | 0.21 (1215) | 0.18 (13081) | 0.11 (8710) | 0.04 (1927) | 0.00 (527) | 0.14 (25463) |
| 2 | 0.42 (463) | 0.34 (7255) | 0.29 (1217) | 0.01 (86)* | 0.00 (15)** | - | 0.33 (9036) |
| 1 | 0.61 (960) | 0.70 (83)* | - | - | - | - | 0.61 (1043) |
| 1-4 | 0.55 (1426) | 0.32 (8554) | 0.18 (15318) | 0.10 (19022) | 0.05 (15595) | 0.01 (19891) | 0.11 (79806) |
| all | 0.55 (1426) | 0.32 (8554) | 0.18 (15318) | 0.10 (19440) | 0.04 (18592) | 0.01 (63236) | 0.07 (126566) |

Same cells, P(the opponent CAN call on its next turn: it holds <= 7 then):

| opp cards \ opp_est | <5 | 5-8 | 8-11 | 11-14 | 14-17 | 17+ | any |
|---|---|---|---|---|---|---|---|
| 5 | - | - | - | 0.15 (418) | 0.05 (2997) | 0.00 (43345) | 0.01 (46760) |
| 4 | - | 1.00 (1)** | 0.24 (1020) | 0.13 (10226) | 0.05 (13653) | 0.01 (19364) | 0.06 (44264) |
| 3 | 0.67 (3)** | 0.52 (1215) | 0.28 (13081) | 0.12 (8710) | 0.04 (1927) | 0.00 (527) | 0.21 (25463) |
| 2 | 0.75 (463) | 0.54 (7255) | 0.35 (1217) | 0.01 (86)* | 0.00 (15)** | - | 0.52 (9036) |
| 1 | 0.75 (960) | 0.99 (83)* | - | - | - | - | 0.77 (1043) |
| 1-4 | 0.75 (1426) | 0.54 (8554) | 0.28 (15318) | 0.13 (19022) | 0.05 (15595) | 0.01 (19891) | 0.17 (79806) |

Finer, by opp_est rounded down to a whole point (P calls next turn):

| opp cards \ est | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15+ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | - | - | - | - | - | - | - | 0.00 (1)** | 0.14 (36)* | 0.13 (209) | 0.10 (775) | 0.10 (2480) | 0.09 (3659) | 0.07 (4087) | 0.06 (4454) | 0.02 (28563) |
| 3 | - | - | - | - | 0.33 (3)** | 0.31 (36)* | 0.24 (263) | 0.20 (916) | 0.19 (3508) | 0.18 (4930) | 0.16 (4643) | 0.13 (3960) | 0.10 (2857) | 0.08 (1893) | 0.05 (995) | 0.02 (1459) |
| 2 | - | 1.00 (1)** | 0.60 (5)** | 0.62 (56)* | 0.39 (401) | 0.35 (2151) | 0.34 (3122) | 0.31 (1982) | 0.29 (889) | 0.30 (256) | 0.12 (72)* | 0.03 (38)* | 0.00 (20)** | 0.00 (28)** | 0.00 (14)** | 0.00 (1)** |
| 1 | - | 0.88 (16)** | 0.63 (298) | 0.57 (497) | 0.65 (149) | 0.69 (75)* | 0.75 (8)** | - | - | - | - | - | - | - | - | - |
| 1-4 | - | 0.88 (17)** | 0.63 (303) | 0.58 (553) | 0.46 (553) | 0.36 (2262) | 0.33 (3393) | 0.28 (2899) | 0.21 (4433) | 0.19 (5395) | 0.15 (5490) | 0.12 (6478) | 0.10 (6536) | 0.07 (6008) | 0.05 (5463) | 0.02 (30023) |

By ply band (opponent holding 1-4 cards), P calls next turn:

| ply \ opp_est | <5 | 5-8 | 8-11 | 11-14 | 14-17 | 17+ |
|---|---|---|---|---|---|---|
| ply 0-5 | 0.44 (9)** | 0.48 (25)** | 0.27 (512) | 0.09 (2289) | 0.05 (3287) | 0.01 (17338) |
| 6-9 | 0.65 (132) | 0.35 (1427) | 0.19 (3190) | 0.11 (7590) | 0.05 (10614) | 0.01 (2414) |
| 10-13 | 0.60 (292) | 0.37 (2791) | 0.18 (6474) | 0.09 (6457) | 0.03 (1544) | 0.00 (128) |
| 14-19 | 0.57 (476) | 0.31 (2885) | 0.18 (4375) | 0.09 (2527) | 0.01 (149) | 0.00 (11)** |
| 20+ | 0.47 (517) | 0.22 (1426) | 0.14 (767) | 0.11 (159) | 0.00 (1)** | - |

**me = counter, opponent = counter** (250,209 of my throws that the opponent answered). Cell: P(opponent calls on its next turn) (samples); `*` n<100, `**` n<30.

| opp cards \ opp_est | <5 | 5-8 | 8-11 | 11-14 | 14-17 | 17+ | any |
|---|---|---|---|---|---|---|---|
| 5 | - | - | 0.00 (3)** | 0.06 (656) | 0.04 (5464) | 0.00 (86285) | 0.01 (92408) |
| 4 | - | 0.14 (7)** | 0.15 (1731) | 0.09 (19140) | 0.05 (26629) | 0.02 (39703) | 0.04 (87210) |
| 3 | 1.00 (2)** | 0.26 (1623) | 0.20 (24904) | 0.11 (18908) | 0.04 (4146) | 0.00 (1172) | 0.15 (50755) |
| 2 | 0.52 (415) | 0.37 (14267) | 0.27 (2974) | 0.02 (184) | 0.00 (27)** | - | 0.35 (17867) |
| 1 | 0.70 (1627) | 0.88 (342) | - | - | - | - | 0.73 (1969) |
| 1-4 | 0.67 (2044) | 0.37 (16239) | 0.20 (29609) | 0.10 (38232) | 0.05 (30802) | 0.01 (40875) | 0.12 (157801) |
| all | 0.67 (2044) | 0.37 (16239) | 0.20 (29612) | 0.10 (38888) | 0.04 (36266) | 0.01 (127160) | 0.08 (250209) |

Same cells, P(the opponent CAN call on its next turn: it holds <= 7 then):

| opp cards \ opp_est | <5 | 5-8 | 8-11 | 11-14 | 14-17 | 17+ | any |
|---|---|---|---|---|---|---|---|
| 5 | - | - | 0.33 (3)** | 0.13 (656) | 0.05 (5464) | 0.00 (86285) | 0.01 (92408) |
| 4 | - | 0.43 (7)** | 0.26 (1731) | 0.13 (19140) | 0.05 (26629) | 0.02 (39703) | 0.06 (87210) |
| 3 | 1.00 (2)** | 0.47 (1623) | 0.28 (24904) | 0.13 (18908) | 0.04 (4146) | 0.00 (1172) | 0.20 (50755) |
| 2 | 0.70 (415) | 0.53 (14267) | 0.34 (2974) | 0.02 (184) | 0.00 (27)** | - | 0.49 (17867) |
| 1 | 0.76 (1627) | 1.00 (342) | - | - | - | - | 0.81 (1969) |
| 1-4 | 0.75 (2044) | 0.53 (16239) | 0.28 (29609) | 0.13 (38232) | 0.05 (30802) | 0.02 (40875) | 0.16 (157801) |

Finer, by opp_est rounded down to a whole point (P calls next turn):

| opp cards \ est | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15+ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | - | - | - | - | - | - | 0.00 (1)** | 0.17 (6)** | 0.20 (41)* | 0.18 (305) | 0.14 (1385) | 0.11 (4394) | 0.10 (6916) | 0.07 (7830) | 0.06 (8542) | 0.02 (57790) |
| 3 | - | - | - | - | 1.00 (2)** | 0.27 (45)* | 0.28 (337) | 0.26 (1241) | 0.23 (5587) | 0.21 (9657) | 0.17 (9660) | 0.13 (8641) | 0.10 (6072) | 0.08 (4195) | 0.05 (2032) | 0.02 (3286) |
| 2 | - | - | 1.00 (4)** | 0.69 (65)* | 0.48 (346) | 0.42 (3065) | 0.36 (6488) | 0.36 (4714) | 0.29 (2211) | 0.27 (560) | 0.14 (203) | 0.04 (67)* | 0.00 (44)* | 0.00 (73)* | 0.00 (24)** | 0.00 (3)** |
| 1 | - | 1.00 (5)** | 0.63 (341) | 0.69 (855) | 0.79 (426) | 0.87 (298) | 0.95 (44)* | - | - | - | - | - | - | - | - | - |
| 1-4 | - | 1.00 (5)** | 0.63 (345) | 0.69 (920) | 0.65 (774) | 0.46 (3408) | 0.36 (6870) | 0.34 (5961) | 0.25 (7839) | 0.21 (10522) | 0.17 (11248) | 0.13 (13102) | 0.10 (13032) | 0.08 (12098) | 0.06 (10598) | 0.02 (61079) |

By ply band (opponent holding 1-4 cards), P calls next turn:

| ply \ opp_est | <5 | 5-8 | 8-11 | 11-14 | 14-17 | 17+ |
|---|---|---|---|---|---|---|
| ply 0-5 | 0.46 (28)** | 0.48 (63)* | 0.23 (1071) | 0.09 (4607) | 0.05 (6221) | 0.02 (35225) |
| 6-9 | 0.61 (289) | 0.36 (2937) | 0.20 (6246) | 0.11 (15757) | 0.05 (20963) | 0.01 (5281) |
| 10-13 | 0.71 (645) | 0.39 (5770) | 0.20 (12777) | 0.09 (12427) | 0.03 (3290) | 0.00 (339) |
| 14-19 | 0.69 (700) | 0.40 (5229) | 0.21 (8068) | 0.10 (5110) | 0.02 (324) | 0.00 (27)** |
| 20+ | 0.59 (382) | 0.26 (2240) | 0.21 (1447) | 0.12 (331) | 0.00 (4)** | 0.00 (3)** |

## b. Real catch rate of a Yaniv call against the `assaf` input

**Reading (b).** The 4-card finding holds and is the same for all three players: real risk / `assaf` = **18x at 0-0.01 (real 4.8%), 8x at 0.01-0.02 (12%), 6x at 0.02-0.04 (16-17%), 4-5x at 0.04-0.08 (21-27%), 2.7x at 0.08-0.15 (31%), 1.5x above (60%)**. `assaf` = 0 exactly was never caught (3,450 + 5,517 + 2,029 positions). At equal `assaf`, MORE opponent cards means MORE real risk (0.01-0.02: 16-22% with 4 cards, 11% with 3; 0.04-0.08: 26-30% with 3 cards, 11-13% with 2; 0.08-0.15: 41-44% with 3, 26-28% with 2), and risk grows with the ply (bottom bin: 1.6% before ply 6, 3% at 6-9, 5% at 10-13, 7% at 14-19, 9% from ply 20).
A cleaner yardstick than `assaf` is `opp_est` minus my total (last table of each block): under 10% risk needs a gap of about 6 with the opponent on 4 cards, 5-6 on 3 cards, 3-4 on 2 cards. With `assaf` <= 0.04 the real risk by my total is 0 / 0 / 0 / 1% / 3% / 6% / 7.5% / 10% for totals 0-7. Calls actually made: hm4 caught 4.9%, trained 4.7%, counter 7.0%.
Samples: cells with opp 5 or opp 1 above the first two rows are empty or tiny; opp 4 above assaf 0.04 is too thin to use.

**me = hm4, opponent = counter**: 14,889 positions where I hold <= 7 at my turn; I called in 10,534 of them, caught 0.049; not calling: the opponent was at or under me in 0.400 of 4,355.
Cell: real share with the opponent's hand <= mine (samples); `*` n<100, `**` n<30.

| assaf | mean assaf | opp 5 | opp 4 | opp 3 | opp 2 | opp 1 | all | real / assaf |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.000 | 0.000 (732) | 0.000 (1134) | 0.000 (989) | 0.000 (515) | 0.000 (80)* | 0.000 (3450) | - |
| 0-0.01 | 0.003 | 0.032 (881) | 0.055 (2979) | 0.040 (1018) | 0.000 (13)** | - | 0.048 (4891) | 17.9x |
| 0.01-0.02 | 0.014 | - | 0.158 (241) | 0.108 (854) | 0.000 (28)** | - | 0.116 (1123) | 8.0x |
| 0.02-0.04 | 0.029 | - | 0.175 (97)* | 0.177 (1137) | 0.045 (133) | 0.000 (3)** | 0.164 (1370) | 5.7x |
| 0.04-0.08 | 0.058 | - | 0.185 (27)** | 0.261 (801) | 0.108 (445) | - | 0.206 (1273) | 3.6x |
| 0.08-0.15 | 0.114 | - | 0.500 (2)** | 0.440 (266) | 0.258 (617) | 0.000 (5)** | 0.311 (890) | 2.7x |
| 0.15+ | 0.401 | - | 1.000 (1)** | 0.620 (108) | 0.559 (1350) | 0.714 (433) | 0.598 (1892) | 1.5x |

By ply band:

| assaf | ply 0-5 | 6-9 | 10-13 | 14-19 | 20+ |
|---|---|---|---|---|---|
| 0 | 0.000 (148) | 0.000 (733) | 0.000 (1014) | 0.000 (1122) | 0.000 (433) |
| 0-0.01 | 0.016 (248) | 0.029 (1367) | 0.046 (1688) | 0.068 (1324) | 0.087 (264) |
| 0.01-0.02 | 0.000 (13)** | 0.066 (167) | 0.116 (378) | 0.136 (434) | 0.122 (131) |
| 0.02-0.04 | 0.125 (16)** | 0.135 (222) | 0.177 (441) | 0.169 (532) | 0.151 (159) |
| 0.04-0.08 | 0.000 (2)** | 0.194 (108) | 0.212 (306) | 0.226 (544) | 0.169 (313) |
| 0.08-0.15 | 0.000 (2)** | 0.270 (89)* | 0.269 (212) | 0.320 (353) | 0.355 (234) |
| 0.15+ | 0.000 (4)** | 0.425 (87)* | 0.485 (336) | 0.598 (758) | 0.678 (707) |

By my own total (all positions where I may call), and with assaf <= 0.04 only:

| my total | mean assaf | real caught | mean assaf (<=0.04) | real caught (<=0.04) |
|---|---|---|---|---|
| 0 | 0.001 | 0.000 (73)* | 0.001 | 0.000 (73)* |
| 1 | 0.002 | 0.000 (263) | 0.000 | 0.000 (259) |
| 2 | 0.012 | 0.011 (741) | 0.002 | 0.001 (702) |
| 3 | 0.055 | 0.082 (1628) | 0.003 | 0.013 (1327) |
| 4 | 0.064 | 0.109 (2510) | 0.004 | 0.033 (1855) |
| 5 | 0.087 | 0.183 (3357) | 0.007 | 0.060 (2256) |
| 6 | 0.063 | 0.164 (3000) | 0.008 | 0.075 (2214) |
| 7 | 0.078 | 0.222 (3317) | 0.009 | 0.097 (2148) |

By the opponent's estimated total `opp_est` minus my total (all assaf values), opponent 1-4 cards:

| opp cards \ est - mine | <0 | 0-2 | 2-4 | 4-6 | 6-8 | 8-11 | 11+ |
|---|---|---|---|---|---|---|---|
| 4 | - | 0.400 (5)** | 0.239 (109) | 0.146 (639) | 0.061 (1176) | 0.020 (1662) | 0.001 (890) |
| 3 | 0.750 (28)** | 0.415 (390) | 0.227 (1433) | 0.106 (1744) | 0.027 (1082) | 0.009 (456) | 0.025 (40)* |
| 2 | 0.602 (495) | 0.429 (1295) | 0.117 (952) | 0.010 (310) | 0.000 (48)* | 0.000 (1)** | - |
| 1 | 0.696 (336) | 0.532 (141) | 0.000 (43)* | 0.000 (1)** | - | - | - |

**me = counter, opponent = counter**: 26,480 positions where I hold <= 7 at my turn; I called in 19,982 of them, caught 0.070; not calling: the opponent was at or under me in 0.493 of 6,498.
Cell: real share with the opponent's hand <= mine (samples); `*` n<100, `**` n<30.

| assaf | mean assaf | opp 5 | opp 4 | opp 3 | opp 2 | opp 1 | all | real / assaf |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.000 | 0.000 (1269) | 0.000 (1985) | 0.000 (1509) | 0.000 (681) | 0.000 (73)* | 0.000 (5517) | - |
| 0-0.01 | 0.003 | 0.021 (1739) | 0.057 (5693) | 0.038 (1488) | 0.000 (8)** | - | 0.047 (8928) | 17.9x |
| 0.01-0.02 | 0.015 | - | 0.220 (422) | 0.110 (1470) | 0.000 (21)** | - | 0.133 (1913) | 9.1x |
| 0.02-0.04 | 0.029 | - | 0.219 (137) | 0.176 (2338) | 0.069 (131) | - | 0.173 (2606) | 5.9x |
| 0.04-0.08 | 0.057 | - | 0.429 (21)** | 0.302 (1811) | 0.134 (418) | - | 0.272 (2250) | 4.8x |
| 0.08-0.15 | 0.115 | - | 0.000 (1)** | 0.413 (482) | 0.276 (917) | - | 0.323 (1400) | 2.8x |
| 0.15+ | 0.387 | - | - | 0.623 (215) | 0.587 (3073) | 0.832 (578) | 0.626 (3866) | 1.6x |

By ply band:

| assaf | ply 0-5 | 6-9 | 10-13 | 14-19 | 20+ |
|---|---|---|---|---|---|
| 0 | 0.000 (278) | 0.000 (1266) | 0.000 (1895) | 0.000 (1623) | 0.000 (455) |
| 0-0.01 | 0.010 (497) | 0.022 (2531) | 0.054 (3232) | 0.067 (2239) | 0.070 (429) |
| 0.01-0.02 | 0.000 (37)* | 0.091 (397) | 0.122 (679) | 0.160 (645) | 0.213 (155) |
| 0.02-0.04 | 0.094 (32)* | 0.102 (481) | 0.196 (965) | 0.174 (883) | 0.229 (245) |
| 0.04-0.08 | 0.200 (5)** | 0.201 (189) | 0.233 (677) | 0.303 (970) | 0.296 (409) |
| 0.08-0.15 | 0.167 (12)** | 0.228 (145) | 0.284 (451) | 0.357 (526) | 0.380 (266) |
| 0.15+ | 0.143 (7)** | 0.550 (191) | 0.606 (803) | 0.655 (1598) | 0.616 (1267) |

By my own total (all positions where I may call), and with assaf <= 0.04 only:

| my total | mean assaf | real caught | mean assaf (<=0.04) | real caught (<=0.04) |
|---|---|---|---|---|
| 0 | 0.000 | 0.000 (49)* | 0.000 | 0.000 (49)* |
| 1 | 0.000 | 0.000 (238) | 0.000 | 0.000 (238) |
| 2 | 0.003 | 0.003 (714) | 0.001 | 0.003 (703) |
| 3 | 0.019 | 0.029 (1668) | 0.002 | 0.008 (1535) |
| 4 | 0.035 | 0.068 (3177) | 0.003 | 0.023 (2692) |
| 5 | 0.062 | 0.130 (5244) | 0.006 | 0.049 (3996) |
| 6 | 0.079 | 0.195 (7093) | 0.009 | 0.075 (4995) |
| 7 | 0.107 | 0.275 (8297) | 0.010 | 0.100 (4756) |

By the opponent's estimated total `opp_est` minus my total (all assaf values), opponent 1-4 cards:

| opp cards \ est - mine | <0 | 0-2 | 2-4 | 4-6 | 6-8 | 8-11 | 11+ |
|---|---|---|---|---|---|---|---|
| 4 | - | 0.750 (4)** | 0.279 (140) | 0.167 (1198) | 0.066 (2421) | 0.015 (2933) | 0.004 (1563) |
| 3 | 0.618 (55)* | 0.425 (824) | 0.253 (2949) | 0.104 (3091) | 0.030 (1706) | 0.011 (635) | 0.000 (53)* |
| 2 | 0.669 (1451) | 0.425 (2244) | 0.169 (1155) | 0.011 (353) | 0.000 (41)* | 0.000 (5)** | - |
| 1 | 0.831 (516) | 0.491 (106) | 0.000 (28)** | 0.000 (1)** | - | - | - |

**me = trained, opponent = counter**: 8,274 positions where I hold <= 7 at my turn; I called in 5,278 of them, caught 0.047; not calling: the opponent was at or under me in 0.343 of 2,996.
Cell: real share with the opponent's hand <= mine (samples); `*` n<100, `**` n<30.

| assaf | mean assaf | opp 5 | opp 4 | opp 3 | opp 2 | opp 1 | all | real / assaf |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.000 | 0.000 (387) | 0.000 (654) | 0.000 (640) | 0.000 (305) | 0.000 (43)* | 0.000 (2029) | - |
| 0-0.01 | 0.003 | 0.020 (509) | 0.060 (1540) | 0.041 (532) | 0.000 (12)** | - | 0.048 (2593) | 17.4x |
| 0.01-0.02 | 0.015 | 0.000 (1)** | 0.145 (159) | 0.119 (480) | 0.000 (6)** | - | 0.124 (646) | 8.5x |
| 0.02-0.04 | 0.029 | 1.000 (1)** | 0.227 (44)* | 0.177 (693) | 0.056 (89)* | - | 0.168 (827) | 5.8x |
| 0.04-0.08 | 0.057 | - | 0.143 (7)** | 0.291 (495) | 0.213 (183) | - | 0.269 (685) | 4.7x |
| 0.08-0.15 | 0.114 | - | 0.500 (2)** | 0.364 (165) | 0.253 (300) | 1.000 (1)** | 0.295 (468) | 2.6x |
| 0.15+ | 0.386 | - | - | 0.704 (54)* | 0.565 (744) | 0.680 (228) | 0.597 (1026) | 1.5x |

By ply band:

| assaf | ply 0-5 | 6-9 | 10-13 | 14-19 | 20+ |
|---|---|---|---|---|---|
| 0 | 0.000 (90)* | 0.000 (402) | 0.000 (610) | 0.000 (636) | 0.000 (291) |
| 0-0.01 | 0.016 (124) | 0.020 (711) | 0.056 (927) | 0.066 (667) | 0.079 (164) |
| 0.01-0.02 | 0.062 (16)** | 0.121 (91)* | 0.085 (199) | 0.139 (267) | 0.192 (73)* |
| 0.02-0.04 | 0.000 (3)** | 0.161 (137) | 0.183 (289) | 0.162 (271) | 0.157 (127) |
| 0.04-0.08 | 0.500 (2)** | 0.132 (53)* | 0.241 (174) | 0.322 (295) | 0.242 (161) |
| 0.08-0.15 | 0.000 (2)** | 0.071 (28)** | 0.233 (146) | 0.329 (173) | 0.378 (119) |
| 0.15+ | - | 0.429 (56)* | 0.520 (204) | 0.608 (380) | 0.653 (386) |

By my own total (all positions where I may call), and with assaf <= 0.04 only:

| my total | mean assaf | real caught | mean assaf (<=0.04) | real caught (<=0.04) |
|---|---|---|---|---|
| 0 | 0.000 | 0.000 (64)* | 0.000 | 0.000 (64)* |
| 1 | 0.009 | 0.013 (228) | 0.001 | 0.000 (222) |
| 2 | 0.022 | 0.038 (498) | 0.002 | 0.011 (443) |
| 3 | 0.045 | 0.081 (853) | 0.003 | 0.021 (678) |
| 4 | 0.066 | 0.121 (1159) | 0.005 | 0.031 (901) |
| 5 | 0.060 | 0.144 (1462) | 0.007 | 0.052 (1116) |
| 6 | 0.068 | 0.181 (1847) | 0.009 | 0.073 (1308) |
| 7 | 0.087 | 0.233 (2163) | 0.010 | 0.106 (1363) |

By the opponent's estimated total `opp_est` minus my total (all assaf values), opponent 1-4 cards:

| opp cards \ est - mine | <0 | 0-2 | 2-4 | 4-6 | 6-8 | 8-11 | 11+ |
|---|---|---|---|---|---|---|---|
| 4 | - | 1.000 (1)** | 0.222 (54)* | 0.140 (365) | 0.073 (669) | 0.014 (851) | 0.006 (466) |
| 3 | 0.846 (13)** | 0.403 (231) | 0.247 (839) | 0.112 (1062) | 0.019 (646) | 0.008 (255) | 0.000 (13)** |
| 2 | 0.563 (350) | 0.435 (604) | 0.176 (450) | 0.005 (210) | 0.000 (24)** | 0.000 (1)** | - |
| 1 | 0.724 (181) | 0.338 (71)* | 0.056 (18)** | 0.000 (2)** | - | - | - |

## c. The counter's total when it calls

**Reading (c).** The counter's calling total falls with its card count: mean **6.3 (5 cards), 6.0 (4), 5.5 (3), 4.6-4.8 (2), 3.0-3.7 (1)**; overall 5.1-5.3, with 23% of calls at 7, 25% at 6, 20% at 5, 15% at 4, 17% at 3 or less. So holding <= 6 can catch 23% of its calls at best, <= 5 48%, <= 4 68%.
hm4 catches **10.1%** of the counter's calls now (the counter catches 7.0% of its twin's and 4.9% of hm4's); when the counter calls, hm4 holds <= 7 in 21% of cases (41% when the counter is down to 1 card, 31% at 2 cards, 12% at 4 cards). The catch rate is 11-12% for counter totals 5-7 and 14% when the counter holds 2 cards.

**counter calls, hm4 is called on** (9,456 calls). The caller's total by its card count; `caught` = P(the other hand <= the caller's total).

| caller's cards | calls (share) | mean total | =0 | =1 | =2 | =3 | =4 | =5 | =6 | =7 | caught | other hand's mean total | P(other hand <= 7) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 311 (3.3%) | 6.34 | 0% | 0% | 0% | 1% | 3% | 12% | 32% | 53% | 0.048 (311) | 16.4 | 0.07 |
| 4 | 1911 (20.2%) | 5.96 | 0% | 0% | 1% | 2% | 9% | 18% | 32% | 40% | 0.073 (1911) | 14.6 | 0.12 |
| 3 | 3609 (38.2%) | 5.45 | 0% | 0% | 2% | 7% | 15% | 20% | 29% | 26% | 0.091 (3609) | 12.9 | 0.16 |
| 2 | 2984 (31.6%) | 4.55 | 0% | 2% | 7% | 17% | 20% | 24% | 19% | 11% | 0.137 (2984) | 10.6 | 0.31 |
| 1 | 641 (6.8%) | 3.04 | 4% | 11% | 24% | 26% | 18% | 12% | 4% | 2% | 0.094 (641) | 8.9 | 0.41 |
| all | 9456 (100.0%) | 5.13 | 0% | 2% | 5% | 10% | 15% | 20% | 25% | 23% | 0.101 (9456) | 12.4 | 0.21 |

Caught rate by the caller's total (all card counts), and the share of calls with the other hand <= 7:

| caller's total | caught |
|---|---|
| 0 | 0.000 (38)* |
| 1 | 0.007 (149) |
| 2 | 0.028 (464) |
| 3 | 0.062 (954) |
| 4 | 0.085 (1406) |
| 5 | 0.119 (1889) |
| 6 | 0.111 (2346) |
| 7 | 0.124 (2210) |

**counter calls, counter is called on** (19,982 calls). The caller's total by its card count; `caught` = P(the other hand <= the caller's total).

| caller's cards | calls (share) | mean total | =0 | =1 | =2 | =3 | =4 | =5 | =6 | =7 | caught | other hand's mean total | P(other hand <= 7) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 633 (3.2%) | 6.32 | 0% | 0% | 0% | 0% | 3% | 15% | 30% | 52% | 0.052 (633) | 16.2 | 0.07 |
| 4 | 3893 (19.5%) | 5.97 | 0% | 0% | 0% | 2% | 9% | 16% | 33% | 40% | 0.063 (3893) | 14.2 | 0.11 |
| 3 | 7707 (38.6%) | 5.52 | 0% | 0% | 2% | 6% | 13% | 20% | 29% | 28% | 0.074 (7707) | 12.4 | 0.16 |
| 2 | 6304 (31.5%) | 4.79 | 0% | 2% | 6% | 12% | 19% | 26% | 23% | 12% | 0.081 (6304) | 10.7 | 0.27 |
| 1 | 1445 (7.2%) | 3.70 | 2% | 6% | 13% | 20% | 24% | 23% | 7% | 3% | 0.033 (1445) | 9.7 | 0.34 |
| all | 19982 (100.0%) | 5.27 | 0% | 1% | 4% | 8% | 15% | 21% | 26% | 24% | 0.070 (19982) | 12.2 | 0.19 |

Caught rate by the caller's total (all card counts), and the share of calls with the other hand <= 7:

| caller's total | caught |
|---|---|
| 0 | 0.000 (49)* |
| 1 | 0.000 (238) |
| 2 | 0.003 (714) |
| 3 | 0.015 (1634) |
| 4 | 0.034 (2946) |
| 5 | 0.056 (4247) |
| 6 | 0.091 (5286) |
| 7 | 0.115 (4868) |

**hm4 calls, counter is called on** (10,534 calls). The caller's total by its card count; `caught` = P(the other hand <= the caller's total).

| caller's cards | calls (share) | mean total | =0 | =1 | =2 | =3 | =4 | =5 | =6 | =7 | caught | other hand's mean total | P(other hand <= 7) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 289 (2.7%) | 6.33 | 0% | 0% | 0% | 0% | 4% | 12% | 30% | 53% | 0.024 (289) | 16.4 | 0.06 |
| 4 | 1775 (16.9%) | 5.81 | 0% | 0% | 1% | 3% | 12% | 19% | 30% | 36% | 0.056 (1775) | 14.5 | 0.10 |
| 3 | 3779 (35.9%) | 5.14 | 0% | 0% | 4% | 11% | 17% | 23% | 24% | 21% | 0.052 (3779) | 12.8 | 0.15 |
| 2 | 3611 (34.3%) | 4.29 | 0% | 4% | 10% | 18% | 21% | 24% | 13% | 10% | 0.047 (3611) | 10.9 | 0.27 |
| 1 | 1080 (10.3%) | 3.11 | 5% | 9% | 22% | 24% | 24% | 9% | 4% | 3% | 0.040 (1080) | 9.5 | 0.39 |
| all | 10534 (100.0%) | 4.79 | 1% | 2% | 7% | 13% | 18% | 21% | 19% | 19% | 0.049 (10534) | 12.2 | 0.21 |

Caught rate by the caller's total (all card counts), and the share of calls with the other hand <= 7:

| caller's total | caught |
|---|---|
| 0 | 0.000 (73)* |
| 1 | 0.000 (259) |
| 2 | 0.004 (723) |
| 3 | 0.021 (1401) |
| 4 | 0.033 (1872) |
| 5 | 0.057 (2191) |
| 6 | 0.063 (2025) |
| 7 | 0.086 (1990) |

## d. Hand sizes over a round

**Reading (d).** Both hands <= 4 is reached (I get to decide there) in **81% of rounds**, at ply 6 on the median (quartiles 4/6/9), and **48-50% of my decisions** come after it; so a 5-card net with the "both" handover makes about half the decisions: (5,5) 20.6%, (5,4) 11.2%, (4,5) 9.1%, (5,3) 4.1%, (3,5) 4.2%, the rest of the 5-rows/columns 3%. With "mine <= 4" or "opp <= 4" the endgame net takes 63-65%, with "either" 79%.
Rounds last 12.7 plies on average (10-90%: 7 to 19-20); showdowns do not happen. **15.8% of Yaniv calls are made while one of the two hands still holds 5 cards** (3.1% by a 5-card caller, 12.7% against a 5-card hand): that is the share of rounds the 5-card net finishes itself. The typical call is 3 or 2 cards against 4 or 3. hm4 and the trained net call in 53% of rounds and are caught 4.9% / 4.7%; margins here +1.45 +-0.23 (hm4) and +1.79 +-0.32 (trained), the same within noise.

**me = hm4 vs counter** (20,000 rounds from 0-0, 137,104 of my decisions, 6.86 a round). A round 'reaches' a point when I make at least one decision there.

| point | rounds reaching it | my decisions after it | mean ply of my first decision there | quartiles of that ply |
|---|---|---|---|---|
| both <= 4 | 81.3% | 48.4% | 6.6 | 4 / 6 / 9 |
| mine <= 4 | 90.4% | 63.0% | 5.0 | 3 / 4 / 6 |
| opp <= 4 | 90.4% | 64.7% | 4.6 | 2 / 4 / 6 |
| either <= 4 | 99.4% | 79.4% | 3.3 | 2 / 3 / 4 |

Share of my decisions by (my cards, opponent's cards):

| my cards \ opp cards | 5 | 4 | 3 | 2 | 1 | any |
|---|---|---|---|---|---|---|
| 5 | 20.6% | 11.2% | 4.1% | 1.0% | 0.1% | 37.0% |
| 4 | 9.1% | 13.0% | 7.5% | 2.1% | 0.2% | 31.9% |
| 3 | 4.2% | 8.2% | 6.2% | 2.2% | 0.2% | 21.0% |
| 2 | 1.2% | 2.8% | 2.9% | 1.6% | 0.2% | 8.8% |
| 1 | 0.2% | 0.4% | 0.4% | 0.3% | 0.1% | 1.3% |
| any | 35.3% | 35.5% | 21.2% | 7.2% | 0.8% | 100% |

Round length in plies (throws by both): mean 12.7, 10/25/50/75/90% = 7 / 9 / 12 / 16 / 19; showdowns (48 plies): 0.05%.
Result: I add 6.55 a round, the counter 8.00 (margin +1.45, +-0.23).

| caller | share of rounds | caught | mean total | caller has 5 | caller has 4 | caller has 3 | caller has 2 | caller has 1 | other has 5 | other has 4 | other has 3 | other has 2 | other has 1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hm4 | 52.7% | 4.9% | 4.79 | 2.7% | 16.9% | 35.9% | 34.3% | 10.3% | 15.3% | 41.8% | 34.5% | 7.5% | 0.8% |
| counter | 47.3% | 10.1% | 5.13 | 3.3% | 20.2% | 38.2% | 31.6% | 6.8% | 10.9% | 36.8% | 40.7% | 10.7% | 1.0% |

Share of all calls by (caller's cards, other's cards), both callers together:

| caller's cards \ other's | 5 | 4 | 3 | 2 | 1 |
|---|---|---|---|---|---|
| 5 | 0.4% | 1.3% | 1.2% | 0.1% | 0.0% |
| 4 | 2.7% | 7.6% | 7.4% | 0.8% | 0.0% |
| 3 | 5.1% | 16.0% | 13.3% | 2.4% | 0.1% |
| 2 | 4.0% | 12.1% | 12.4% | 4.0% | 0.4% |
| 1 | 1.0% | 2.4% | 3.1% | 1.7% | 0.4% |

**me = trained vs counter** (10,000 rounds from 0-0, 69,117 of my decisions, 6.91 a round). A round 'reaches' a point when I make at least one decision there.

| point | rounds reaching it | my decisions after it | mean ply of my first decision there | quartiles of that ply |
|---|---|---|---|---|
| both <= 4 | 81.2% | 49.5% | 6.5 | 4 / 6 / 8 |
| mine <= 4 | 91.0% | 64.9% | 4.8 | 3 / 4 / 6 |
| opp <= 4 | 89.4% | 64.3% | 4.7 | 2 / 4 / 6 |
| either <= 4 | 99.2% | 79.6% | 3.3 | 2 / 3 / 4 |

Share of my decisions by (my cards, opponent's cards):

| my cards \ opp cards | 5 | 4 | 3 | 2 | 1 | any |
|---|---|---|---|---|---|---|
| 5 | 20.4% | 10.2% | 3.6% | 0.9% | 0.1% | 35.1% |
| 4 | 9.3% | 12.7% | 7.4% | 2.0% | 0.2% | 31.6% |
| 3 | 4.5% | 8.5% | 6.5% | 2.3% | 0.2% | 22.1% |
| 2 | 1.3% | 3.1% | 3.2% | 1.5% | 0.3% | 9.4% |
| 1 | 0.2% | 0.4% | 0.7% | 0.5% | 0.1% | 1.8% |
| any | 35.7% | 34.9% | 21.4% | 7.2% | 0.9% | 100% |

Round length in plies (throws by both): mean 12.8, 10/25/50/75/90% = 7 / 9 / 12 / 16 / 20; showdowns (48 plies): 0.00%.
Result: I add 6.40 a round, the counter 8.19 (margin +1.79, +-0.32).

| caller | share of rounds | caught | mean total | caller has 5 | caller has 4 | caller has 3 | caller has 2 | caller has 1 | other has 5 | other has 4 | other has 3 | other has 2 | other has 1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| trained | 52.8% | 4.7% | 4.83 | 2.3% | 16.1% | 36.5% | 34.7% | 10.5% | 17.0% | 43.1% | 31.9% | 7.1% | 0.8% |
| counter | 47.2% | 11.1% | 5.10 | 3.4% | 19.8% | 38.1% | 31.4% | 7.2% | 10.1% | 35.8% | 40.9% | 12.2% | 1.0% |

Share of all calls by (caller's cards, other's cards), both callers together:

| caller's cards \ other's | 5 | 4 | 3 | 2 | 1 |
|---|---|---|---|---|---|
| 5 | 0.6% | 1.0% | 1.1% | 0.1% | 0.0% |
| 4 | 2.5% | 7.5% | 7.0% | 0.8% | 0.0% |
| 3 | 5.2% | 15.8% | 13.4% | 2.8% | 0.1% |
| 2 | 4.3% | 12.9% | 11.5% | 4.0% | 0.5% |
| 1 | 1.2% | 2.5% | 3.1% | 1.8% | 0.4% |
## e. Behaviour while someone holds 5 cards

**Reading (e).** While someone holds 5 cards the three players throw sets of 6+ points alike (93-98%) and never break a pair. They differ on LOW pairs when a much higher single is in hand (H >= 2X): **pair of 2s: counter 3%, hm4 9%, trained 68%; pair of 3s: 5% / 85% / 96%; pair of 4s: 9% / 98% / 98%; pair of 5s: 56% / 99% / 99%**; aces and jokers are never thrown as a pair by anyone. The trained net throws a pair of 2s more when the opponent still has 5 cards (35% against 20%).
Open card, unpaired: everyone takes 0-2; a 3: counter 99%, hm4 85%, trained 93%; **a 4: 79% / 1% / 23%; a 5: 41% / 0% / 7%**; 6+ never. A card whose rank I hold is taken 96-100% of the time by all three, whatever its value (and the matching card is kept).
hm4 and the trained net differ on 12.3% of these positions (16.7% at 5-5, 7-11% elsewhere with enough samples; the (5,1) and (1,5) cells have ~100 positions). **42% of those differences are only which ten-point rank (T/J/Q/K) is thrown** - same points, so the real disagreement rate is about 7%. The real kinds, in order: (1) the trained net throws a low pair (mostly 2s) where hm4 throws its highest single, 23%; (2) the trained net takes an unpaired open 4-6 that hm4 leaves, 8%; (3) the trained net throws a lower single and keeps a high one (sometimes to take an open card that pairs it), 7%; (4) both hold two sets and throw a different one / three of a kind against two, 6%; (5) with a hand of low cards hm4 throws a lower single than its highest, 5%. Nothing here says which of the two is right: f and the designers' tests must.

`python -m training.yaniv.handmade5_measure e --rounds 6000`: 41,752 positions (my decisions with 5 cards in my hand or the opponent's), from 6,000 rounds of hm4 vs counter and 6,000 of trained vs counter (seeds 903000+). The three players choose on the same positions. Cells: counter / hm4 / trained (samples).

### e1. A set is available (exactly one rank held twice or more; Yaniv not legal)

P(throws the set), by the rank's points X and the highest other card H:

| pair of X points each | H >= 2X | X < H < 2X | H <= X | any H |
|---|---|---|---|---|
| 0 (jokers) | 0.00 / 0.00 / 0.00 (453) | - | - | 0.00 / 0.00 / 0.00 (453) |
| 1 | 0.00 / 0.00 / 0.01 (2470) | - | - | 0.00 / 0.00 / 0.01 (2470) |
| 2 | 0.03 / 0.09 / 0.68 (1790) | 0.97 / 0.97 / 0.97 (75)* | 1.00 / 1.00 / 1.00 (1)** | 0.07 / 0.12 / 0.69 (1867) |
| 3 | 0.05 / 0.85 / 0.96 (794) | 0.95 / 0.99 / 0.99 (188) | 0.97 / 1.00 / 0.97 (73)* | 0.28 / 0.89 / 0.97 (1057) |
| 4 | 0.09 / 0.98 / 0.98 (562) | 0.94 / 0.97 / 0.97 (316) | 0.97 / 0.97 / 0.99 (231) | 0.52 / 0.98 / 0.98 (1124) |
| 5 | 0.56 / 0.99 / 0.99 (410) | 0.91 / 0.98 / 0.96 (451) | 0.96 / 0.98 / 0.98 (355) | 0.81 / 0.98 / 0.98 (1238) |
| 6 | - | 0.97 / 0.97 / 0.95 (789) | 0.97 / 0.98 / 0.98 (424) | 0.97 / 0.97 / 0.96 (1226) |
| 7 | - | 0.97 / 0.97 / 0.95 (662) | 0.96 / 0.98 / 0.96 (593) | 0.97 / 0.98 / 0.96 (1268) |
| 8 | - | 0.96 / 0.96 / 0.96 (510) | 0.96 / 0.97 / 0.96 (644) | 0.96 / 0.97 / 0.96 (1166) |
| 9 | - | 0.95 / 0.95 / 0.92 (480) | 0.96 / 0.96 / 0.94 (736) | 0.96 / 0.96 / 0.93 (1227) |
| 10 | - | - | 0.94 / 0.94 / 0.93 (4531) | 0.94 / 0.94 / 0.93 (4558) |
| any | 0.06 / 0.28 / 0.46 (6479) | 0.96 / 0.97 / 0.95 (3471) | 0.95 / 0.96 / 0.94 (7588) | 0.62 / 0.71 / 0.77 (17654) |

What is done instead: P(keeps the pair and throws another card) and P(breaks the pair):

| pair | throws the set | keeps the pair, throws another card | breaks the pair | throws the set when the opp has 5 | throws the set when the opp has <= 4 |
|---|---|---|---|---|---|
| X 0-2 | 0.03 / 0.05 / 0.27 (4790) | 0.97 / 0.95 / 0.73 (4790) | 0.00 / 0.00 / 0.00 (4790) | 0.03 / 0.05 / 0.35 (2315) | 0.03 / 0.05 / 0.20 (2475) |
| X 3-5 | 0.55 / 0.95 / 0.97 (3419) | 0.45 / 0.05 / 0.03 (3419) | 0.00 / 0.00 / 0.00 (3419) | 0.50 / 0.95 / 0.98 (2225) | 0.64 / 0.97 / 0.96 (1194) |
| X 6-9 | 0.96 / 0.97 / 0.95 (4887) | 0.04 / 0.03 / 0.05 (4887) | 0.00 / 0.00 / 0.00 (4887) | 0.96 / 0.96 / 0.94 (3363) | 0.98 / 0.98 / 0.97 (1524) |
| X 10 | 0.94 / 0.94 / 0.93 (4558) | 0.06 / 0.06 / 0.07 (4558) | 0.00 / 0.00 / 0.00 (4558) | 0.93 / 0.93 / 0.91 (3385) | 0.99 / 0.98 / 0.98 (1173) |

Three of a kind held (or four): P(throws the whole set): 0.67 / 0.73 / 0.77 (991)

### e2. Taking the open card (Yaniv not legal)

| open card's points | unpaired: takes it | unpaired, opp has 5 | unpaired, opp has <= 4 | its rank is in my hand: takes it | ... takes it and keeps the matching card (builds a set) |
|---|---|---|---|---|---|
| 0 | 1.00 / 1.00 / 1.00 (207) | 1.00 / 1.00 / 1.00 (207) | - | 1.00 / 1.00 / 1.00 (22)** | 1.00 / 1.00 / 1.00 (22)** |
| 1 | 1.00 / 1.00 / 1.00 (331) | 1.00 / 1.00 / 1.00 (331) | - | 1.00 / 1.00 / 1.00 (104) | 1.00 / 1.00 / 1.00 (104) |
| 2 | 1.00 / 1.00 / 1.00 (432) | 1.00 / 1.00 / 1.00 (357) | 1.00 / 1.00 / 0.99 (75)* | 1.00 / 1.00 / 1.00 (159) | 1.00 / 1.00 / 1.00 (159) |
| 3 | 0.99 / 0.85 / 0.93 (584) | 0.99 / 0.87 / 0.94 (391) | 1.00 / 0.80 / 0.92 (193) | 1.00 / 0.99 / 1.00 (306) | 1.00 / 0.99 / 1.00 (306) |
| 4 | 0.79 / 0.01 / 0.23 (1234) | 0.69 / 0.00 / 0.21 (768) | 0.96 / 0.02 / 0.27 (466) | 1.00 / 0.99 / 0.99 (568) | 0.98 / 0.98 / 0.98 (568) |
| 5 | 0.41 / 0.00 / 0.07 (1782) | 0.32 / 0.00 / 0.10 (1067) | 0.55 / 0.00 / 0.02 (715) | 0.97 / 0.97 / 0.98 (731) | 0.96 / 0.97 / 0.97 (731) |
| 6 | 0.02 / 0.00 / 0.02 (2166) | 0.01 / 0.00 / 0.03 (1282) | 0.03 / 0.00 / 0.00 (884) | 0.98 / 0.98 / 0.97 (780) | 0.98 / 0.98 / 0.97 (780) |
| 7 | 0.00 / 0.00 / 0.01 (2668) | 0.00 / 0.00 / 0.01 (1639) | 0.00 / 0.00 / 0.00 (1029) | 0.96 / 0.97 / 0.97 (873) | 0.96 / 0.97 / 0.97 (873) |
| 8 | 0.00 / 0.00 / 0.00 (3119) | 0.00 / 0.00 / 0.00 (2029) | 0.00 / 0.00 / 0.00 (1090) | 0.94 / 0.97 / 0.97 (865) | 0.94 / 0.97 / 0.97 (865) |
| 9 | 0.00 / 0.00 / 0.00 (3345) | 0.00 / 0.00 / 0.00 (2223) | 0.00 / 0.00 / 0.00 (1122) | 0.96 / 0.98 / 0.98 (831) | 0.96 / 0.98 / 0.98 (831) |
| 10 | 0.00 / 0.00 / 0.00 (16063) | 0.00 / 0.00 / 0.00 (11221) | 0.00 / 0.00 / 0.00 (4842) | 0.96 / 0.98 / 0.99 (3393) | 0.96 / 0.98 / 0.99 (3393) |

### e3. How often the players differ, by (my cards, opponent's cards)

| my cards, opp cards | positions | hm4 differs from trained | hm4 differs from counter | trained differs from counter |
|---|---|---|---|---|
| 5, 5 | 16898 | 16.7% | 22.3% | 21.9% |
| 5, 4 | 9008 | 10.9% | 15.7% | 16.8% |
| 5, 3 | 3143 | 8.1% | 18.4% | 15.8% |
| 5, 2 | 757 | 12.0% | 21.4% | 14.9% |
| 5, 1 | 97* | 20.6% | 29.9% | 15.5% |
| 4, 5 | 7496 | 7.3% | 9.0% | 9.7% |
| 3, 5 | 3271 | 8.4% | 7.1% | 8.2% |
| 2, 5 | 978 | 13.3% | 6.9% | 10.4% |
| 1, 5 | 104 | 13.5% | 13.5% | 5.8% |
| all | 41752 | 12.3% | 16.6% | 16.6% |

### e4. The most common kinds of hm4-vs-trained disagreement (5,146 disagreements)

- **both throw a ten-point single but a different rank of the same points (which of T/J/Q/K goes)**: 2147 (41.7%)
  - hand 2♠ 3♥ 6♦ T♥ Q♣ (31 pts), open K♠, opponent 5 cards, ply 1: hm4 throws Q, draws from the deck; trained throws T, draws from the deck; counter throws T, draws from the deck
  - hand 2♣ 6♦ T♣ J♠ K♥ (38 pts), open Q♥, opponent 5 cards, ply 1: hm4 throws T, draws from the deck; trained throws K, draws from the deck; counter throws T, draws from the deck
  - hand 4♠ 5♣ 8♦ Q♣ K♠ (37 pts), open 7♥, opponent 5 cards, ply 0: hm4 throws Q, draws from the deck; trained throws K, draws from the deck; counter throws K, draws from the deck
- **hm4 throws its highest single, the trained net throws a set**: 1166 (22.7%)
  - hand ★ A♦ 2♦ 2♠ 4♠ (9 pts), open 8♥, opponent 5 cards, ply 0: hm4 throws 4, draws from the deck; trained throws 2 x 2, draws from the deck; counter throws 4, draws from the deck
  - hand 2♦ 2♥ 5♦ T♦ (19 pts), open K♥, opponent 5 cards, ply 3: hm4 throws T, draws from the deck; trained throws 2 x 2, draws from the deck; counter throws T, draws from the deck
  - hand A♣ 2♦ 2♠ 4♥ 6♠ (15 pts), open K♥, opponent 5 cards, ply 1: hm4 throws 6, draws from the deck; trained throws 2 x 2, draws from the deck; counter throws 6, draws from the deck
- **same throw; only the trained net takes the open card (worth 4-6, unpaired)**: 434 (8.4%)
  - hand 2♦ 5♣ 6♠ 7♥ T♠ (30 pts), open 4♠, opponent 2 cards, ply 6: hm4 throws T, draws from the deck; trained throws T, takes the open card; counter throws T, takes the open card
  - hand ★ A♦ A♠ 2♠ 7♥ (11 pts), open 5♦, opponent 1 cards, ply 9: hm4 throws 7, draws from the deck; trained throws 7, takes the open card; counter throws 7, takes the open card
  - hand A♣ 2♥ 9♠ (12 pts), open 4♦, opponent 5 cards, ply 19: hm4 throws 9, draws from the deck; trained throws 9, takes the open card; counter throws 9, draws from the deck
- **hm4 throws its highest single, the trained net throws a lower single**: 349 (6.8%)
  - hand 2♦ 5♣ 6♠ 7♥ 9♠ (29 pts), open 6♣, opponent 2 cards, ply 8: hm4 throws 9, takes the open card; trained throws 7, takes the open card; counter throws 9, takes the open card
  - hand ★ 2♣ 8♠ T♥ K♦ (30 pts), open T♣, opponent 5 cards, ply 0: hm4 throws K, takes the open card; trained throws 8, takes the open card; counter throws K, takes the open card
  - hand 2♦ 4♥ T♥ J♠ Q♣ (36 pts), open 8♥, opponent 5 cards, ply 1: hm4 throws Q, draws from the deck; trained throws 2, draws from the deck; counter throws J, draws from the deck
- **both throw a set but of a different rank and value**: 291 (5.7%)
  - hand 3♥ 4♣ 8♣ 8♥ 8♠ (31 pts), open 8♦, opponent 5 cards, ply 5: hm4 throws 3 x 8, draws from the deck; trained throws 2 x 8, takes the open card; counter throws 4, takes the open card
  - hand 3♥ 3♠ 4♣ J♦ J♥ (30 pts), open 8♥, opponent 4 cards, ply 1: hm4 throws 2 x J, draws from the deck; trained throws 2 x 3, draws from the deck; counter throws 2 x J, draws from the deck
  - hand 2♦ 2♥ 5♦ T♣ T♠ (29 pts), open J♠, opponent 5 cards, ply 1: hm4 throws 2 x T, draws from the deck; trained throws 2 x 2, draws from the deck; counter throws 2 x T, draws from the deck
- **hm4 throws a lower single, the trained net throws the highest single**: 236 (4.6%)
  - hand A♥ 2♣ 3♥ 4♦ (10 pts), open 5♥, opponent 5 cards, ply 10: hm4 throws 3, draws from the deck; trained throws 4, draws from the deck; counter throws 4, draws from the deck
  - hand ★ A♥ A♠ 3♣ 5♦ (10 pts), open J♠, opponent 5 cards, ply 1: hm4 throws 3, draws from the deck; trained throws 5, draws from the deck; counter throws 5, draws from the deck
  - hand 3♥ 5♣ 6♦ 8♦ (22 pts), open K♣, opponent 5 cards, ply 4: hm4 throws 6, draws from the deck; trained throws 8, draws from the deck; counter throws 8, draws from the deck
- **hm4 throws a lower single, the trained net throws a set**: 95 (1.8%)
  - hand 2♣ 2♠ 6♣ 7♣ 8♠ (25 pts), open 8♥, opponent 3 cards, ply 5: hm4 throws 7, takes the open card; trained throws 2 x 2, takes the open card; counter throws 7, takes the open card
  - hand 2♦ 2♠ 3♦ 6♦ K♠ (23 pts), open K♥, opponent 3 cards, ply 6: hm4 throws 6, takes the open card; trained throws 2 x 2, takes the open card; counter throws K, draws from the deck
  - hand 2♦ 2♥ 6♥ Q♥ (20 pts), open Q♠, opponent 5 cards, ply 2: hm4 throws 6, takes the open card; trained throws 2 x 2, takes the open card; counter throws 6, takes the open card
- **hm4 throws a set, the trained net throws a lower single**: 93 (1.8%)
  - hand A♣ 2♦ 3♦ 8♣ 8♠ (22 pts), open 8♥, opponent 4 cards, ply 6: hm4 throws 2 x 8, draws from the deck; trained throws 3, takes the open card; counter throws 2 x 8, draws from the deck
  - hand ★ 3♠ 7♣ 7♠ (17 pts), open 7♦, opponent 5 cards, ply 7: hm4 throws 2 x 7, draws from the deck; trained throws 3, takes the open card; counter throws 3, takes the open card
  - hand 6♠ T♠ J♥ Q♥ Q♠ (46 pts), open ★, opponent 5 cards, ply 0: hm4 throws 2 x Q, takes the open card; trained throws 6, takes the open card; counter throws 2 x Q, takes the open card
- **hm4 throws a set, the trained net throws the highest single**: 92 (1.8%)
  - hand 3♣ 3♠ 7♣ 8♦ 9♠ (30 pts), open T♥, opponent 3 cards, ply 5: hm4 throws 2 x 3, draws from the deck; trained throws 9, draws from the deck; counter throws 9, draws from the deck
  - hand 3♥ 6♣ 6♦ 7♥ 8♦ (30 pts), open K♦, opponent 5 cards, ply 5: hm4 throws 2 x 6, draws from the deck; trained throws 8, draws from the deck; counter throws 2 x 6, draws from the deck
  - hand ★ A♦ 2♥ 2♠ 5♦ (10 pts), open 6♦, opponent 5 cards, ply 6: hm4 throws 2 x 2, draws from the deck; trained throws 5, draws from the deck; counter throws 5, draws from the deck
## f. What a card is worth

**Reading (f).** In result points (counter's added minus mine; a round is worth +-16): by the cell means, one card fewer for me is worth about **+4 at plies 4-8 and +3 at ply 12**, one card fewer for the counter costs the same; equal card counts give about +1 (hm4's edge). Holding points equal (f2), a card costs **3.7 at ply 4, 2.7 at ply 8, 2.0 at ply 12**, and a held point costs 0.28 / 0.53 / 0.73. The card's cost falls slowly as the opponent's hand shrinks (ply 8: 2.9 / 2.9 / 2.6 / 2.0 for 5 / 4 / 3 / 2 opp cards) while a point's cost doubles (0.38 / 0.51 / 0.62 / 0.77). **So in units of held points one card is worth about 7.5 / 5.5 / 4 / 2.5 points against 5 / 4 / 3 / 2 opponent cards at ply 8** (at ply 4: ~20 / 11 / 6; at ply 12: 3.5 / 3.3 / 2.5 / 1.8). The 4-card net's charge of 5 (3 against 1-2 cards) is right for 3-4 opponent cards and too low while the opponent holds 5, early.
The rollouts agree in size: deleting a 0-2 point card of mine is worth +2.6 +-0.2 whatever the opponent's card count (2.3 at 2 cards), and the same to the counter (-2.7). Deleting a 3+ card is worth about +5.5 nearly whatever its value: a high card costs one turn, not its points - except late: a ten is worth +4.0 against 5 opponent cards and +7 to +9 against 3-2. Being the one to move is worth +3.
Not reliable: any cell with the opponent or me on 1 card, ply-4 cells with 2 cards, and the "opp has 2" row at ply 4.

`python -m training.yaniv.handmade5_measure f --rounds 30000`: 30,000 rounds of hm4 vs counter from 0-0 (seeds 904000+). Result = points the counter adds minus points I add, at the end of the round (overall mean +1.48). Cells: mean ±95% range (samples); `*` n<100, `**` n<30. A round's result has a standard deviation of about 16 points, so cells with under ~1,000 samples are only good to +-1.

### f1. Mean final result by (my cards, opp cards) at ply 4 (29,769 rounds still going)

| my cards \ opp cards | 5 | 4 | 3 | 2 | 1 |
|---|---|---|---|---|---|
| 5 | +0.9 ±0.5 (4937) | -2.7 ±0.4 (5186) | -7.6 ±0.8 (1361) | -11.7 ±2.8 (103) | -15.0 ±2.8 (2)** |
| 4 | +5.2 ±0.4 (6369) | +1.2 ±0.4 (5923) | -3.7 ±0.9 (1427) | -9.1 ±2.9 (97)* | -13.3 ±2.3 (3)** |
| 3 | +9.3 ±0.6 (2001) | +5.5 ±0.7 (1703) | +0.5 ±1.5 (423) | -10.1 ±4.6 (21)** | - |
| 2 | +14.0 ±1.8 (105) | +12.6 ±1.9 (92)* | -0.7 ±8.6 (14)** | - | - |
| 1 | +26.0 ±0.0 (1)** | - | +12.0 ±0.0 (1)** | - | - |

### f1. Mean final result by (my cards, opp cards) at ply 8 (25,989 rounds still going)

| my cards \ opp cards | 5 | 4 | 3 | 2 | 1 |
|---|---|---|---|---|---|
| 5 | +1.8 ±1.2 (752) | -2.5 ±0.7 (1807) | -5.6 ±0.8 (1422) | -10.1 ±1.4 (381) | -16.5 ±2.5 (23)** |
| 4 | +4.3 ±0.6 (2242) | +1.2 ±0.5 (4718) | -3.1 ±0.6 (3153) | -6.9 ±1.1 (681) | -11.6 ±3.3 (54)* |
| 3 | +8.4 ±0.6 (1996) | +5.2 ±0.5 (3771) | +1.0 ±0.6 (2107) | -3.3 ±1.5 (445) | -10.0 ±3.3 (28)** |
| 2 | +12.1 ±0.8 (622) | +8.3 ±0.8 (997) | +4.9 ±1.0 (552) | -0.4 ±2.5 (90)* | -8.2 ±4.2 (4)** |
| 1 | +14.3 ±2.5 (49)* | +11.6 ±2.7 (65)* | +8.5 ±3.8 (24)** | -6.2 ±12.9 (5)** | -4.0 ±0.0 (1)** |

### f1. Mean final result by (my cards, opp cards) at ply 12 (16,663 rounds still going)

| my cards \ opp cards | 5 | 4 | 3 | 2 | 1 |
|---|---|---|---|---|---|
| 5 | +1.2 ±2.5 (134) | -2.3 ±1.4 (509) | -4.4 ±1.3 (605) | -8.6 ±1.7 (255) | -9.2 ±4.8 (34)* |
| 4 | +5.2 ±1.2 (577) | +1.3 ±0.7 (1795) | -2.1 ±0.7 (1960) | -4.3 ±1.1 (783) | -10.4 ±2.0 (74)* |
| 3 | +6.8 ±0.9 (893) | +4.3 ±0.6 (2446) | +2.1 ±0.6 (2180) | -1.3 ±1.1 (829) | -4.0 ±3.2 (88)* |
| 2 | +9.7 ±1.0 (464) | +7.6 ±0.7 (1140) | +4.1 ±0.7 (1055) | +1.5 ±1.2 (389) | -2.4 ±3.8 (36)* |
| 1 | +11.1 ±1.8 (79)* | +8.5 ±1.5 (163) | +7.2 ±1.7 (116) | +4.1 ±1.4 (55)* | +0.8 ±6.1 (4)** |

### f2. Matched comparison: least-squares fit of the final result on cards and points held

result ~ const + a x my cards + b x my points + c x opp cards + d x opp points (+ I am to move). Coefficients ± 95% range. `a` is what one card MORE in my hand costs at equal points.

| ply | rounds | n | per card of mine | per point of mine | per opp card | per opp point | I am to move |
|---|---|---|---|---|---|---|---|
| 4 | all | 29769 | -3.72 ±0.26 | -0.28 ±0.03 | +3.41 ±0.27 | +0.36 ±0.03 | +3.03 ±0.35 |
| 4 | opp has 5 | 13413 | -4.01 ±0.38 | -0.19 ±0.04 | - | +0.32 ±0.04 | +2.62 ±0.51 |
| 4 | opp has 4 | 12904 | -3.62 ±0.40 | -0.32 ±0.04 | - | +0.36 ±0.04 | +3.19 ±0.54 |
| 4 | opp has 3 | 3226 | -3.12 ±0.78 | -0.50 ±0.08 | - | +0.50 ±0.09 | +3.83 ±1.06 |
| 4 | opp has 2 | 221 | -0.91 ±2.81 | -0.55 ±0.26 | - | +0.54 ±0.37 | +3.17 ±3.56 |
| 8 | all | 25989 | -2.74 ±0.22 | -0.53 ±0.03 | +2.50 ±0.22 | +0.62 ±0.04 | +2.93 ±0.35 |
| 8 | opp has 5 | 5661 | -2.90 ±0.44 | -0.38 ±0.07 | - | +0.58 ±0.07 | +2.62 ±0.71 |
| 8 | opp has 4 | 11358 | -2.90 ±0.34 | -0.51 ±0.05 | - | +0.62 ±0.05 | +2.87 ±0.54 |
| 8 | opp has 3 | 7258 | -2.59 ±0.42 | -0.62 ±0.06 | - | +0.66 ±0.07 | +3.20 ±0.67 |
| 8 | opp has 2 | 1602 | -2.04 ±0.85 | -0.77 ±0.13 | - | +0.67 ±0.16 | +3.29 ±1.38 |
| 12 | all | 16663 | -1.98 ±0.25 | -0.73 ±0.05 | +1.62 ±0.26 | +0.81 ±0.05 | +2.64 ±0.43 |
| 12 | opp has 5 | 2147 | -1.76 ±0.64 | -0.50 ±0.12 | - | +0.73 ±0.13 | +3.51 ±1.08 |
| 12 | opp has 4 | 6053 | -2.29 ±0.42 | -0.69 ±0.08 | - | +0.96 ±0.08 | +2.52 ±0.70 |
| 12 | opp has 3 | 5916 | -1.93 ±0.43 | -0.77 ±0.09 | - | +0.75 ±0.09 | +2.69 ±0.73 |
| 12 | opp has 2 | 2311 | -1.61 ±0.68 | -0.89 ±0.14 | - | +0.58 ±0.17 | +2.32 ±1.18 |

### f3. Rollouts: delete one card and play the round out again (same deck order, same players)

At plies 4, 8 and 12 one random card is deleted from my hand (or from the counter's), and the rest of the round is replayed; the cell is the mean change of my final result. A deleted 0-2 point card is close to the pure value of holding one card fewer.

**One of MY cards deleted: my gain, by the deleted card's points and the opponent's card count** (plies pooled; 71,858 rollouts)

| deleted card | opp cards 5 | opp cards 4 | opp cards 3 | opp cards 2 | opp cards 1 | any |
|---|---|---|---|---|---|---|
| 0-2 | +2.6 ±0.3 (6521) | +2.7 ±0.3 (10640) | +2.6 ±0.4 (6791) | +2.3 ±0.7 (1933) | +1.6 ±1.6 (188) | +2.6 ±0.2 (26073) |
| 3-5 | +5.8 ±0.4 (5962) | +5.5 ±0.3 (8852) | +5.7 ±0.5 (4860) | +6.4 ±0.9 (1198) | +4.7 ±2.0 (91)* | +5.7 ±0.2 (20963) |
| 6-9 | +5.1 ±0.4 (5464) | +5.0 ±0.4 (6991) | +5.9 ±0.6 (3104) | +6.3 ±1.3 (617) | +7.6 ±3.3 (41)* | +5.3 ±0.2 (16217) |
| 10 | +4.0 ±0.5 (3145) | +5.0 ±0.5 (3604) | +7.0 ±0.8 (1504) | +8.7 ±1.9 (326) | +7.9 ±7.1 (26)** | +5.2 ±0.3 (8605) |
| any | +4.4 ±0.2 (21092) | +4.3 ±0.2 (30087) | +4.6 ±0.2 (16259) | +4.6 ±0.5 (4074) | +3.6 ±1.2 (346) | +4.4 ±0.1 (71858) |

By ply and by the hand's own card count before the deletion (deleted card worth 0-2):

| ply | hand had 5 | hand had 4 | hand had 3 | hand had 2 | any |
|---|---|---|---|---|---|
| 4 | +3.2 ±0.5 (3791) | +2.5 ±0.4 (3977) | +2.3 ±0.8 (928) | +1.9 ±1.5 (45)* | +2.8 ±0.3 (8741) |
| 8 | +2.9 ±0.7 (2011) | +3.0 ±0.4 (4412) | +2.1 ±0.5 (2884) | +1.3 ±0.8 (636) | +2.6 ±0.3 (9943) |
| 12 | +3.3 ±1.0 (880) | +3.0 ±0.6 (2628) | +2.0 ±0.6 (2826) | +1.0 ±0.6 (1055) | +2.3 ±0.3 (7389) |

Straight-line fit over all deleted cards: change = +2.87 +0.35 x the card's points.

**One of the COUNTER's cards deleted: my change (negative = my loss), by the deleted card's points and MY card count** (plies pooled; 72,070 rollouts)

| deleted card | my cards 5 | my cards 4 | my cards 3 | my cards 2 | my cards 1 | any |
|---|---|---|---|---|---|---|
| 0-2 | -2.9 ±0.4 (5292) | -2.8 ±0.3 (10180) | -2.5 ±0.3 (7264) | -2.4 ±0.5 (2537) | -2.7 ±1.2 (291) | -2.7 ±0.2 (25564) |
| 3-5 | -6.3 ±0.5 (5605) | -6.1 ±0.3 (10607) | -6.2 ±0.4 (7192) | -5.2 ±0.6 (2072) | -4.3 ±1.0 (186) | -6.1 ±0.2 (25662) |
| 6-9 | -5.9 ±0.5 (4315) | -6.3 ±0.4 (5960) | -6.6 ±0.6 (2948) | -6.9 ±1.0 (613) | -6.8 ±3.1 (54)* | -6.3 ±0.3 (13890) |
| 10 | -5.1 ±0.7 (2240) | -6.5 ±0.6 (2978) | -8.3 ±0.9 (1411) | -8.8 ±1.6 (298) | -8.1 ±1.8 (27)** | -6.5 ±0.4 (6954) |
| any | -5.0 ±0.2 (17452) | -5.0 ±0.2 (29725) | -5.0 ±0.2 (18815) | -4.3 ±0.4 (5520) | -3.9 ±0.8 (558) | -5.0 ±0.1 (72070) |

By ply and by the hand's own card count before the deletion (deleted card worth 0-2):

| ply | hand had 5 | hand had 4 | hand had 3 | hand had 2 | any |
|---|---|---|---|---|---|
| 4 | -2.7 ±0.4 (4072) | -3.2 ±0.5 (3505) | -2.3 ±0.9 (757) | +0.1 ±1.7 (48)* | -2.9 ±0.3 (8382) |
| 8 | -2.3 ±0.6 (2427) | -3.1 ±0.4 (4456) | -2.8 ±0.6 (2514) | -0.9 ±1.0 (471) | -2.7 ±0.3 (9868) |
| 12 | -3.2 ±0.9 (1052) | -2.3 ±0.6 (2985) | -2.3 ±0.6 (2510) | -2.4 ±0.9 (767) | -2.4 ±0.3 (7314) |

Straight-line fit over all deleted cards: change = -2.91 -0.49 x the card's points.

## g. Do the match-score units pay in the real game?

**Reading (g).** Yes, clearly. On the same 3,000 match deals hm4 wins **70.0%** with the score units and **62.0%** without (+-1.7 each; difference 8 points, about 6 times the noise). With them hm4 lands on exactly 50 or 100 **0.48 times a match** (in 38% of matches; 0.33 at 50 and 0.15 at 100), against 0.10 without them, which is also the counter's rate (0.10-0.12). 94% of hm4's cuts come from being called on while holding the right total, 6% from its own caught call (score + hand + 30). The units cost nothing visible in rounds (see the working test below).

`python -m training.yaniv.handmade5_measure g --matches 3000 --scratch DIR` builds hm4 without `add_score_units` and plays 3,000 matches to 100 of each player against the counter on the same deals (seeds 905000+). A cut = a round after which a score lands exactly on 50 or 100 (and drops by 50).

| player | matches won-lost | rounds a match | my cuts at 50 a match | my cuts at 100 a match | my cuts a match | the counter's cuts a match | matches with a cut of mine | my cuts: called on / my own caught call / showdown |
|---|---|---|---|---|---|---|---|---|
| hm4 (with score units) | 2100-900 (70.0% ±1.6%) | 13.1 | 0.328 | 0.148 | 0.476 | 0.119 | 38.3% | 1342 / 85 / 1 |
| hm4 without score units | 1861-1139 (62.0% ±1.7%) | 12.3 | 0.066 | 0.029 | 0.095 | 0.116 | 9.2% | 253 / 31 / 0 |
| counter (vs counter) | 1490-1510 (49.7% ±1.8%) | 12.4 | 0.067 | 0.039 | 0.106 | 0.103 | 10.0% | 286 / 32 / 0 |

### g2. The working test (`handmade_eval --game 5x7 --workers 2 --work`, 6,000 rounds seed 61000, 1,500 matches seed 61001)

| player | margin | rounds won | calls caught | matches |
|---|---|---|---|---|
| hm4 (with score units) | +1.49 | 55.2% | 4.5% | 1029-471 (68.6%, 95% range 66.2-70.9%) |
| hm4 without score units (scratch net `hm4-no-score-units.json`) | +1.50 | 55.3% | 4.5% | 920-580 (61.3%, 95% range 58.8-63.8%) |

Rounds are the same within noise (the test's rounds start from 0-0, where the units are silent); matches fall by 7.3 points without the units (noise about +-2.5). Both tests agree: the score units are worth 7-8 points of matches in the real game, and hm4 on the working test is the bar: **+1.49 margin, 68.6% of matches**.
