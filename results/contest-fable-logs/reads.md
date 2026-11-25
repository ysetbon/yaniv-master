# Thread `reads`: can the net read the opponent's hand better, and do better reads make better throws?

Files: `training/yaniv/contest_fable_reads_measure.py` (logged games), `contest_fable_reads.py` (design + probes),
`contest_fable_reads_run.py`, `contest_fable_reads_run2.py` (paired runners). All runs with 2 workers. Rounds from 0-0.

## 1. Measurement (plain logged games, nothing fitted)

Commands (hero hm5; one row per hero decision; rows where the hero calls Yaniv are dropped in the tables):

    python -m training.yaniv.contest_fable_reads_measure hm5 {hm5,trained,counter} 30000 4101 m_<opp>.npz 2   # 211,248 / 216,797 / 206,425 decisions
    python -m training.yaniv.contest_fable_reads_measure hm5 {hm5,trained,counter} 30000 4102 n_<opp>.npz 2   # + my state after the move

Truth columns: e1 = the opponent calls Yaniv on its next turn; e2 = it calls on its next turn or the one after
(= it ends the round before my second next turn, the ALPHA measurement); opp_true = its real total.
"regime" = the opponent's card count (12 = 1 or 2 cards); est buckets are contest_fable's EST_CUTS.

### Table A. P(opponent calls on its next turn) by (its cards, opp_est bucket); cell = rate (n). hm5 hero, 30,000 rounds, seed 4101

vs counter

| reg \ eb | <5 | 5-8 | 8-11 | 11-14 | 14-17 | 17+ |
|---|---|---|---|---|---|---|
| 5 |  |  |  | 0.051 (727) | 0.034 (5362) | 0.005 (66656) |
| 4 |  |  | 0.116 (1699) | 0.084 (16184) | 0.044 (20574) | 0.016 (27638) |
| 3 |  | 0.224 (1801) | 0.176 (19614) | 0.107 (12493) | 0.054 (2735) | 0.006 (676) |
| 12 | 0.556 (1948) | 0.342 (10202) | 0.231 (1734) | 0.018 (109) |  |  |

vs hm5

| reg \ eb | <5 | 5-8 | 8-11 | 11-14 | 14-17 | 17+ |
|---|---|---|---|---|---|---|
| 5 |  |  |  | 0.049 (446) | 0.038 (2924) | 0.005 (53770) |
| 4 |  |  | 0.086 (1677) | 0.080 (13163) | 0.048 (17893) | 0.012 (35169) |
| 3 |  | 0.157 (2843) | 0.151 (22420) | 0.094 (14804) | 0.033 (4731) | 0.003 (1347) |
| 12 | 0.406 (5019) | 0.246 (17493) | 0.153 (2204) | 0.011 (283) | 0.000 (54) |  |

vs trained

| reg \ eb | <5 | 5-8 | 8-11 | 11-14 | 14-17 | 17+ |
|---|---|---|---|---|---|---|
| 5 |  |  |  | 0.035 (491) | 0.034 (3434) | 0.005 (55156) |
| 4 |  |  | 0.075 (1706) | 0.078 (14031) | 0.044 (18177) | 0.012 (33330) |
| 3 |  | 0.114 (3021) | 0.133 (23169) | 0.099 (14468) | 0.039 (4456) | 0.002 (1291) |
| 12 | 0.268 (8329) | 0.209 (17464) | 0.212 (2208) | 0.025 (204) | 0.000 (48) |  |

### Table A2. mean TRUE total of the opponent, same cells

vs counter

| reg | <5 | 5-8 | 8-11 | 11-14 | 14-17 | 17+ |
|---|---|---|---|---|---|---|
| 5 |  |  | 8.4 | 12.2 | 14.1 | 24.7 |
| 4 |  | 7.4 | 10.6 | 12.3 | 15.0 | 20.7 |
| 3 | 3.0 | 8.3 | 10.4 | 13.0 | 16.5 | 22.9 |
| 12 | 5.0 | 7.9 | 10.0 | 17.0 | 19.9 |  |

vs hm5

| reg | <5 | 5-8 | 8-11 | 11-14 | 14-17 | 17+ |
|---|---|---|---|---|---|---|
| 5 |  |  | 5.5 | 11.9 | 14.2 | 26.1 |
| 4 |  | 10.7 | 10.4 | 12.7 | 15.7 | 22.0 |
| 3 | 3.7 | 8.1 | 10.6 | 14.1 | 17.9 | 23.3 |
| 12 | 4.8 | 7.8 | 11.7 | 17.6 | 20.0 |  |

vs trained

| reg | <5 | 5-8 | 8-11 | 11-14 | 14-17 | 17+ |
|---|---|---|---|---|---|---|
| 5 |  |  |  | 11.7 | 14.1 | 25.9 |
| 4 |  | 7.9 | 10.3 | 12.5 | 15.5 | 22.1 |
| 3 | 3.1 | 8.0 | 10.4 | 13.8 | 17.7 | 23.4 |
| 12 | 5.0 | 7.8 | 11.0 | 17.1 | 20.0 |  |

### Table B. the same by MY cards after the move (k = kept + 1). seed 4102, 30,000 rounds each

vs counter, opponent regime 12

| k_after \ eb | <5 | 5-8 | 8-11 | 11-14 | 14-17 | 17+ |
|---|---|---|---|---|---|---|
| 1 | 0.282 (262) | 0.118 (440) |  |  |  |  |
| 2 | 0.481 (761) | 0.183 (2519) | 0.033 (153) |  |  |  |
| 3 | 0.634 (569) | 0.411 (3656) | 0.285 (520) |  |  |  |
| 4 | 0.651 (335) | 0.423 (2603) | 0.307 (748) | 0.021 (47) |  |  |
| 5 | 0.758 (95) | 0.400 (658) | 0.254 (382) | 0.028 (36) |  |  |

vs counter, opponent regime 3

| k_after \ eb | <5 | 5-8 | 8-11 | 11-14 | 14-17 | 17+ |
|---|---|---|---|---|---|---|
| 1 |  | 0.018 (223) | 0.015 (583) | 0.000 (50) |  |  |
| 2 |  | 0.117 (830) | 0.064 (4329) | 0.034 (886) | 0.000 (41) |  |
| 3 |  | 0.317 (561) | 0.202 (8242) | 0.108 (3861) | 0.046 (414) | 0.000 (92) |
| 4 |  | 0.403 (176) | 0.233 (5561) | 0.112 (5628) | 0.053 (1365) | 0.006 (332) |
| 5 |  | 0.161 (31) | 0.198 (978) | 0.110 (2014) | 0.039 (898) | 0.013 (317) |

vs counter, opponent regime 4

| k_after \ eb | <5 | 5-8 | 8-11 | 11-14 | 14-17 | 17+ |
|---|---|---|---|---|---|---|
| 1 |  |  | 0.009 (114) | 0.002 (525) | 0.000 (169) |  |
| 2 |  |  | 0.049 (777) | 0.032 (3903) | 0.013 (2052) | 0.007 (442) |
| 3 |  |  | 0.188 (702) | 0.104 (7113) | 0.047 (7769) | 0.020 (4448) |
| 4 |  |  | 0.267 (150) | 0.110 (3983) | 0.051 (8464) | 0.017 (13302) |
| 5 |  |  |  | 0.098 (594) | 0.047 (2226) | 0.013 (9286) |

vs hm5, opponent regime 12

| k_after \ eb | <5 | 5-8 | 8-11 | 11-14 | 14-17 | 17+ |
|---|---|---|---|---|---|---|
| 1 | 0.213 (748) | 0.082 (832) |  |  |  |  |
| 2 | 0.262 (2167) | 0.107 (5053) | 0.078 (77) |  |  |  |
| 3 | 0.545 (1485) | 0.287 (6440) | 0.146 (501) | 0.000 (69) |  |  |
| 4 | 0.630 (689) | 0.391 (4033) | 0.215 (1122) | 0.000 (110) |  |  |
| 5 | 0.639 (191) | 0.332 (921) | 0.196 (581) | 0.000 (60) |  |  |

vs hm5, opponent regime 3

| k_after \ eb | <5 | 5-8 | 8-11 | 11-14 | 14-17 | 17+ |
|---|---|---|---|---|---|---|
| 1 |  | 0.012 (339) | 0.013 (627) | 0.000 (34) |  |  |
| 2 |  | 0.074 (1331) | 0.052 (4882) | 0.018 (819) | 0.000 (131) |  |
| 3 |  | 0.227 (872) | 0.159 (9257) | 0.097 (4402) | 0.027 (849) | 0.000 (247) |
| 4 |  | 0.416 (214) | 0.224 (6024) | 0.092 (6879) | 0.029 (2366) | 0.002 (650) |
| 5 |  |  | 0.191 (1069) | 0.096 (2620) | 0.030 (1497) | 0.009 (549) |

vs hm5, opponent regime 4

| k_after \ eb | <5 | 5-8 | 8-11 | 11-14 | 14-17 | 17+ |
|---|---|---|---|---|---|---|
| 1 |  |  | 0.000 (138) | 0.000 (417) | 0.000 (97) |  |
| 2 |  |  | 0.033 (659) | 0.024 (3042) | 0.014 (1621) | 0.003 (690) |
| 3 |  |  | 0.136 (610) | 0.101 (5695) | 0.053 (6843) | 0.015 (5857) |
| 4 |  |  | 0.236 (123) | 0.124 (3254) | 0.050 (7468) | 0.013 (16398) |
| 5 |  |  |  | 0.116 (518) | 0.048 (2083) | 0.010 (11762) |

vs trained, opponent regime 12

| k_after \ eb | <5 | 5-8 | 8-11 | 11-14 | 14-17 | 17+ |
|---|---|---|---|---|---|---|
| 1 | 0.173 (2000) | 0.094 (902) |  |  |  |  |
| 2 | 0.196 (3600) | 0.083 (4999) | 0.051 (59) |  |  |  |
| 3 | 0.315 (2596) | 0.196 (7154) | 0.102 (508) | 0.000 (46) |  |  |
| 4 | 0.534 (743) | 0.374 (3682) | 0.301 (1095) | 0.041 (98) |  |  |
| 5 | 0.707 (147) | 0.360 (851) | 0.273 (587) | 0.093 (54) |  |  |

vs trained, opponent regime 3

| k_after \ eb | <5 | 5-8 | 8-11 | 11-14 | 14-17 | 17+ |
|---|---|---|---|---|---|---|
| 1 |  | 0.020 (346) | 0.013 (640) | 0.000 (33) |  |  |
| 2 |  | 0.040 (1497) | 0.028 (5204) | 0.013 (785) | 0.000 (80) |  |
| 3 |  | 0.187 (987) | 0.135 (9792) | 0.083 (4356) | 0.023 (744) | 0.000 (238) |
| 4 |  | 0.353 (221) | 0.219 (5865) | 0.098 (6684) | 0.033 (2227) | 0.003 (637) |
| 5 |  | 0.200 (30) | 0.195 (1010) | 0.111 (2468) | 0.036 (1472) | 0.009 (530) |

vs trained, opponent regime 4

| k_after \ eb | <5 | 5-8 | 8-11 | 11-14 | 14-17 | 17+ |
|---|---|---|---|---|---|---|
| 1 |  |  | 0.007 (143) | 0.000 (437) | 0.000 (104) |  |
| 2 |  |  | 0.024 (757) | 0.014 (3416) | 0.014 (1615) | 0.003 (581) |
| 3 |  |  | 0.113 (671) | 0.095 (6265) | 0.047 (6941) | 0.014 (5583) |
| 4 |  |  | 0.203 (158) | 0.115 (3444) | 0.052 (7572) | 0.013 (15741) |
| 5 |  |  |  | 0.107 (524) | 0.052 (2034) | 0.010 (11224) |

### Table B2. baseline for PI: my cards after >= 4

vs counter

| reg \ eb | <5 | 5-8 | 8-11 | 11-14 | 14-17 | 17+ |
|---|---|---|---|---|---|---|
| 5 |  |  |  | 0.111 (63) | 0.040 (1064) | 0.003 (52696) |
| 4 |  |  | 0.250 (164) | 0.108 (4577) | 0.051 (10690) | 0.015 (22588) |
| 3 |  | 0.367 (207) | 0.228 (6539) | 0.111 (7642) | 0.047 (2263) | 0.009 (649) |
| 12 | 0.674 (430) | 0.419 (3261) | 0.289 (1130) | 0.024 (83) |  |  |

vs hm5

| reg \ eb | <5 | 5-8 | 8-11 | 11-14 | 14-17 | 17+ |
|---|---|---|---|---|---|---|
| 5 |  |  |  | 0.111 (36) | 0.055 (659) | 0.003 (44186) |
| 4 |  |  | 0.231 (130) | 0.123 (3772) | 0.050 (9551) | 0.012 (28160) |
| 3 |  | 0.391 (243) | 0.219 (7093) | 0.093 (9499) | 0.029 (3863) | 0.005 (1199) |
| 12 | 0.632 (880) | 0.380 (4954) | 0.208 (1703) | 0.000 (170) | 0.000 (51) |  |

vs trained

| reg \ eb | <5 | 5-8 | 8-11 | 11-14 | 14-17 | 17+ |
|---|---|---|---|---|---|---|
| 5 |  |  |  | 0.098 (41) | 0.046 (720) | 0.004 (45033) |
| 4 |  |  | 0.193 (166) | 0.114 (3968) | 0.052 (9606) | 0.012 (26965) |
| 3 |  | 0.335 (251) | 0.215 (6875) | 0.102 (9152) | 0.034 (3699) | 0.006 (1167) |
| 12 | 0.563 (890) | 0.371 (4533) | 0.291 (1682) | 0.059 (152) | 0.000 (43) |  |

### Table C. P(the opponent ends the round before my second next turn) (the ALPHA measurement) by (its cards, my cards after)

vs counter

| reg \ k_after | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| 5 | 0.000 (519) | 0.021 (4076) | 0.038 (13745) | 0.024 (30193) | 0.012 (23630) |
| 4 | 0.006 (824) | 0.046 (7177) | 0.133 (20032) | 0.121 (25899) | 0.082 (12120) |
| 3 | 0.030 (857) | 0.127 (6089) | 0.310 (13170) | 0.332 (13062) | 0.271 (4238) |
| 12 | 0.270 (714) | 0.413 (3436) | 0.654 (4757) | 0.679 (3746) | 0.620 (1177) |

vs hm5

| reg \ k_after | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| 5 | 0.000 (305) | 0.012 (2496) | 0.044 (9274) | 0.025 (24361) | 0.010 (20520) |
| 4 | 0.004 (684) | 0.038 (6021) | 0.117 (19010) | 0.108 (27243) | 0.067 (14370) |
| 3 | 0.030 (1011) | 0.099 (7196) | 0.251 (15630) | 0.289 (16133) | 0.229 (5764) |
| 12 | 0.214 (1582) | 0.274 (7311) | 0.526 (8505) | 0.627 (5983) | 0.544 (1775) |

vs trained

| reg \ k_after | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| 5 | 0.000 (333) | 0.010 (2705) | 0.040 (9807) | 0.025 (24905) | 0.011 (20889) |
| 4 | 0.006 (708) | 0.027 (6374) | 0.109 (19460) | 0.106 (26915) | 0.071 (13790) |
| 3 | 0.031 (1024) | 0.062 (7599) | 0.212 (16119) | 0.280 (15634) | 0.244 (5510) |
| 12 | 0.247 (2904) | 0.242 (8665) | 0.388 (10314) | 0.588 (5642) | 0.608 (1658) |

### Table C1. P(opponent calls next) by (its cards, my cards after), all est

vs counter

| reg \ k_after | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| 5 | 0.000 (519) | 0.009 (4076) | 0.015 (13745) | 0.006 (30193) | 0.002 (23630) |
| 4 | 0.002 (824) | 0.027 (7177) | 0.066 (20032) | 0.044 (25899) | 0.024 (12120) |
| 3 | 0.016 (857) | 0.067 (6089) | 0.173 (13170) | 0.159 (13062) | 0.108 (4238) |
| 12 | 0.176 (714) | 0.242 (3436) | 0.423 (4757) | 0.414 (3746) | 0.368 (1177) |

vs hm5

| reg \ k_after | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| 5 | 0.000 (305) | 0.005 (2496) | 0.018 (9274) | 0.006 (24361) | 0.002 (20520) |
| 4 | 0.000 (684) | 0.020 (6021) | 0.058 (19010) | 0.038 (27243) | 0.019 (14370) |
| 3 | 0.014 (1011) | 0.051 (7196) | 0.136 (15630) | 0.132 (16133) | 0.089 (5764) |
| 12 | 0.143 (1582) | 0.152 (7311) | 0.321 (8505) | 0.376 (5983) | 0.305 (1775) |

vs trained

| reg \ k_after | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| 5 | 0.000 (333) | 0.005 (2705) | 0.017 (9807) | 0.006 (24905) | 0.002 (20889) |
| 4 | 0.001 (708) | 0.014 (6374) | 0.056 (19460) | 0.038 (26915) | 0.020 (13790) |
| 3 | 0.015 (1024) | 0.028 (7599) | 0.117 (16119) | 0.134 (15634) | 0.097 (5510) |
| 12 | 0.148 (2904) | 0.130 (8665) | 0.220 (10314) | 0.373 (5642) | 0.347 (1658) |

### Table D. open card (the opponent's last throw; first turns excluded): P(call next) observed / expected from the (regime, est) cell, mean (true - est), n. seed 4101

vs counter

| reg | 0-2 | 3-4 | 5-7 | 8-9 | 10 |
|---|---|---|---|---|---|
| 5 |  | 0.055 / 0.019, -5.3, 2217 | 0.019 / 0.012, -3.9, 11389 | 0.006 / 0.009, -2.2, 12170 | 0.004 / 0.008, +0.0, 31975 |
| 4 | 0.078 / 0.058, -3.7, 551 | 0.080 / 0.056, -2.0, 7610 | 0.054 / 0.047, -0.5, 19862 | 0.033 / 0.041, +0.4, 13310 | 0.029 / 0.038, +1.0, 24769 |
| 3 | 0.071 / 0.142, -0.6, 1035 | 0.150 / 0.151, +0.2, 7020 | 0.155 / 0.147, +0.8, 11783 | 0.142 / 0.140, +1.1, 6465 | 0.135 / 0.137, +1.4, 11017 |
| 12 | 0.103 / 0.346, +1.3, 698 | 0.326 / 0.368, +1.2, 3748 | 0.392 / 0.355, +1.5, 4348 | 0.383 / 0.349, +1.5, 1984 | 0.375 / 0.345, +1.5, 3235 |

vs hm5

| reg | 0-2 | 3-4 | 5-7 | 8-9 | 10 |
|---|---|---|---|---|---|
| 5 |  | 0.050 / 0.017, -4.7, 1353 | 0.018 / 0.011, -3.3, 7502 | 0.008 / 0.009, -1.7, 8386 | 0.004 / 0.008, +0.5, 24864 |
| 4 | 0.011 / 0.025, +1.3, 2392 | 0.044 / 0.038, +0.4, 7706 | 0.051 / 0.041, +0.1, 17882 | 0.035 / 0.037, +0.8, 13336 | 0.028 / 0.034, +1.6, 26595 |
| 3 | 0.037 / 0.101, +1.8, 1950 | 0.112 / 0.119, +1.3, 6756 | 0.137 / 0.121, +1.3, 13276 | 0.126 / 0.118, +1.6, 8739 | 0.106 / 0.114, +1.9, 15430 |
| 12 | 0.074 / 0.270, +1.1, 1723 | 0.249 / 0.283, +1.1, 6386 | 0.303 / 0.264, +1.8, 6628 | 0.299 / 0.260, +1.9, 3899 | 0.279 / 0.257, +1.9, 6417 |

vs trained

| reg | 0-2 | 3-4 | 5-7 | 8-9 | 10 |
|---|---|---|---|---|---|
| 5 |  | 0.052 / 0.016, -5.0, 1467 | 0.018 / 0.011, -3.6, 7812 | 0.006 / 0.009, -1.8, 8862 | 0.004 / 0.008, +0.6, 25888 |
| 4 | 0.006 / 0.023, +1.4, 2383 | 0.050 / 0.038, +0.3, 8006 | 0.049 / 0.040, -0.1, 18114 | 0.033 / 0.036, +0.8, 13210 | 0.026 / 0.033, +1.6, 25542 |
| 3 | 0.034 / 0.101, +0.9, 1744 | 0.104 / 0.110, +0.8, 7308 | 0.131 / 0.112, +1.0, 13421 | 0.114 / 0.109, +1.4, 8630 | 0.096 / 0.105, +1.9, 15313 |
| 12 | 0.118 / 0.237, +1.6, 3071 | 0.255 / 0.227, +1.7, 6471 | 0.253 / 0.222, +1.7, 7254 | 0.231 / 0.222, +1.7, 4150 | 0.211 / 0.222, +1.8, 7307 |

### What the tables say (step 1)

1. **(count, opp_est) fit.** contest_fable's PI rows are right for the counter and a little high for hm5 / trained in
   the 1-2 card regime (PI 0.50/0.35/0.27 for est <5 / 5-8 / 8-11; measured counter 0.56/0.34/0.23, hm5
   0.41/0.25/0.15, trained 0.27/0.21/0.21). In the 1-2 card regime with est >= 11 the opponent calls next 1-2% of
   the time (PI says 0.15 / 0.08): a 1-2 card hand estimated that high holds a known heavy card.
   ALPHA (e2, table C; all my card counts pooled): counter 0.025/0.107/0.284/0.570 for 5/4/3/1-2 cards, hm5
   0.022/0.093/0.237/0.450, trained 0.021/0.091/0.216/0.385 (contest_fable: 0.15/0.35/0.6/0.85, deliberately a
   step above).
2. **MY card count after the move is the strongest missing signal, for all three opponents** (table B / C1 / C).
   P(it calls next) with 2 cards in my hand is a third of the value with 4-5 cards, with 1 card about a tenth
   (except est < 5 against 1-2 cards), with 3 cards 0.7-0.9 (trained 0.55). E.g. regime 1-2, est 5-8: my cards
   1/2/3/4/5 -> hm5 0.08/0.11/0.29/0.39/0.33, trained 0.09/0.08/0.20/0.37/0.36, counter 0.12/0.18/0.41/0.42/0.40.
   Reason: every opponent's call rule reads its own chance of being caught, which is high against a short hand.
   As an est shift: 3 cards ~ +1.5 points, 2 cards ~ +5, 1 card ~ +9 (maps the k-card rows onto the 4-5 card row).
   e2 (ALPHA) at 2 cards is 0.33-0.45 of its value at 4 cards; at 1 card 0.1-0.4.
3. **Open card A / 2 / joker** (the opponent just threw a low card): P(call next) is 0.4-0.5 of the cell's value
   against hm5 / trained in every regime below 5 cards (e.g. 1-2 cards: 0.074 against 0.270 expected, n 1,723;
   after the my-cards correction still 0.42-0.53). Against the counter 0.3-0.5 for 1-3 cards. n is small: about
   0.2 decisions a round.
4. **Against 5 cards, a low open card** (it has thrown singles only, highest first, so its other cards are at or
   under the open card): open 3-4 -> P(call next) 0.050-0.055 against 0.017 from the cell, true total 4.7-5.3
   UNDER est; open 5-7 -> 0.018 / 0.011, 3.3-3.9 under; open 8-9: 1.7-2.2 under; open 10: est is right.
   Strong and identical for all three opponents, but the base rate is tiny (0.05 at most) and the class is rare
   (open 3-4 against 5 cards: 0.045 decisions a round).
5. Weak or nothing beyond these: ply (ratios 0.9-1.2 within a cell), unseen-low share, known card count
   (known_n >= 1 against 1-2 cards: 1.27 x; against 0: 0.88 x), my own low cards (large raw effect, but it is my
   card count in disguise: ratio 0.95-1.15 after the my-cards correction), gone count of the open rank.
6. The est error itself: against 1-2 cards with no known card est is 2.4 points too LOW (hm5: est 6.3, true 8.7);
   with one known card it is right. (For the Yaniv thread, not used here.)
7. **The catch bonus pays for something that almost never happens** (table E). The net adds CATCH x PI x
   P(my hand <= 5 after the draw), PI ~ 0.4 against 1-2 cards. Measured P(it calls next AND I catch it), my hand
   really <= 5: after TAKING THE OPEN CARD to 1-2 cards 0.000-0.010 for every opponent (it knows the card: when
   it does call, 16-22% of the time, it is because it is sure); after a deck draw to 2 cards 0.04 (hm5), 0.013
   (trained), 0.155 (counter); to 3 cards 0.175 / 0.076 / 0.304; to 4 cards 0.295 / 0.240 / 0.352.

### Table E. My hand after the move is <= 5 points (the catch bonus's condition), seed 4102: what the net assumes (mean PI of the cell) against what happens. cell = PI assumed / P(opp calls next) / P(it calls AND I catch it) / n

| opp | its cards | my cards after | deck draw | took the open card |
|---|---|---|---|---|
| counter | 12 | 1 | 0.41 / 0.103 / 0.019 / 107 | 0.40 / 0.218 / 0.000 / 501 |
| counter | 12 | 2 | 0.38 / 0.255 / 0.155 / 679 | 0.38 / 0.205 / 0.010 / 1263 |
| counter | 12 | 3 | 0.37 / 0.428 / 0.304 / 717 | 0.36 / 0.307 / 0.165 / 375 |
| counter | 12 | 4 | 0.35 / 0.454 / 0.352 / 227 |  |
| counter | 3 | 1 | 0.19 / 0.009 / 0.005 / 213 | 0.20 / 0.031 / 0.000 / 384 |
| counter | 3 | 2 | 0.18 / 0.063 / 0.040 / 1177 | 0.19 / 0.075 / 0.005 / 986 |
| counter | 3 | 3 | 0.17 / 0.151 / 0.139 / 1334 | 0.16 / 0.137 / 0.117 / 342 |
| counter | 3 | 4 | 0.15 / 0.191 / 0.179 / 341 |  |
| counter | 4 | 1 | 0.08 / 0.000 / 0.000 / 229 | 0.09 / 0.006 / 0.000 / 326 |
| counter | 4 | 2 | 0.08 / 0.017 / 0.010 / 1212 | 0.08 / 0.018 / 0.004 / 553 |
| counter | 4 | 3 | 0.06 / 0.057 / 0.052 / 1219 | 0.06 / 0.069 / 0.069 / 246 |
| counter | 4 | 4 | 0.04 / 0.019 / 0.016 / 320 |  |
| hm5 | 12 | 1 | 0.42 / 0.004 / 0.000 / 271 | 0.42 / 0.219 / 0.000 / 1021 |
| hm5 | 12 | 2 | 0.39 / 0.119 / 0.040 / 1284 | 0.40 / 0.156 / 0.004 / 3058 |
| hm5 | 12 | 3 | 0.38 / 0.339 / 0.175 / 1322 | 0.38 / 0.238 / 0.058 / 702 |
| hm5 | 12 | 4 | 0.36 / 0.428 / 0.295 / 383 |  |
| hm5 | 3 | 1 | 0.20 / 0.000 / 0.000 / 304 | 0.22 / 0.037 / 0.000 / 376 |
| hm5 | 3 | 2 | 0.19 / 0.048 / 0.013 / 1427 | 0.20 / 0.073 / 0.001 / 1147 |
| hm5 | 3 | 3 | 0.17 / 0.105 / 0.083 / 1533 | 0.16 / 0.075 / 0.043 / 414 |
| hm5 | 3 | 4 | 0.14 / 0.135 / 0.122 / 394 |  |
| hm5 | 4 | 1 | 0.09 / 0.000 / 0.000 / 212 | 0.09 / 0.000 / 0.000 / 206 |
| hm5 | 4 | 2 | 0.08 / 0.007 / 0.003 / 988 | 0.08 / 0.016 / 0.002 / 433 |
| hm5 | 4 | 3 | 0.06 / 0.039 / 0.038 / 1005 | 0.05 / 0.013 / 0.013 / 234 |
| hm5 | 4 | 4 | 0.04 / 0.028 / 0.028 / 281 |  |
| trained | 12 | 1 | 0.46 / 0.051 / 0.004 / 552 | 0.45 / 0.218 / 0.000 / 1666 |
| trained | 12 | 2 | 0.41 / 0.070 / 0.013 / 1738 | 0.42 / 0.200 / 0.001 / 3483 |
| trained | 12 | 3 | 0.39 / 0.227 / 0.076 / 1672 | 0.41 / 0.229 / 0.021 / 1339 |
| trained | 12 | 4 | 0.37 / 0.398 / 0.240 / 362 |  |
| trained | 3 | 1 | 0.21 / 0.000 / 0.000 / 273 | 0.22 / 0.034 / 0.000 / 439 |
| trained | 3 | 2 | 0.19 / 0.023 / 0.009 / 1457 | 0.20 / 0.047 / 0.001 / 1418 |
| trained | 3 | 3 | 0.17 / 0.081 / 0.062 / 1584 | 0.17 / 0.090 / 0.041 / 458 |
| trained | 3 | 4 | 0.14 / 0.117 / 0.098 / 386 |  |
| trained | 4 | 1 | 0.09 / 0.000 / 0.000 / 218 | 0.10 / 0.004 / 0.000 / 228 |
| trained | 4 | 2 | 0.08 / 0.004 / 0.003 / 1062 | 0.08 / 0.017 / 0.006 / 473 |
| trained | 4 | 3 | 0.06 / 0.042 / 0.041 / 1085 | 0.05 / 0.020 / 0.020 / 197 |
| trained | 4 | 4 | 0.04 / 0.028 / 0.028 / 288 |  |

### Table F. P(the opponent takes the card I left on top) by its points, seed 4102

| opp | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| counter | 1.00 | 1.00 | 1.00 | 0.92 | 0.60 | 0.20 | 0.14 | 0.12 | 0.10 | 0.09 |
| hm5 | 1.00 | 1.00 | 0.77 | 0.36 | 0.22 | 0.19 | 0.17 | 0.15 | 0.14 | 0.11 |
| trained | 0.50 | 0.96 | 0.80 | 0.43 | 0.23 | 0.17 | 0.16 | 0.13 | 0.13 | 0.10 |

## 2. Designs and paired tests

Module `training/yaniv/contest_fable_reads.py`, `build(**kw)`. Paired runner: `contest_fable_reads_run.py`
(hero = the design, alternative = hm5's move where it differs; **gain NEGATIVE = the design's move is better**).
Kit.trace check (300 rounds, 4,016 decisions, scratch chk.py): `build(KSLOPE=0, OPEN_LOW=0)` makes hm5's move
every time; the est_eff steps match `est + 3.75 relu(2.4 - kept) + 4.5 [open A/2/joker]` to 2e-4.

### 2a. Design K: P(call) read through an effective estimate (FAILED)

est_eff = opp_est + 3.75 x relu(2.4 - cards kept) + 4.5 x [open card A / 2 / joker], read by BOTH the plan term
and the catch term (551 units, 7 fewer than hm5). `python -m training.yaniv.contest_fable_reads_run
'py:contest_fable_reads.build' OPP 20000 71`:

| opp | moves changed a round | each (hm5's move minus mine) | a round |
|---|---|---|---|
| hm5 | 0.046 | +0.63 +-0.53 | +0.029 (worse) |
| trained | 0.051 | +0.48 +-0.52 | +0.025 (worse) |

By kind against hm5: I keep a pair where hm5 throws it (86 cases) +3.99 +-0.67 (bad: a lower P(call) raises the
plan's weight on G); "same throw, I draw from the deck" (514) -1.12 +-0.70 (good); other throw with lower points
(165) +2.82 +-1.25 (bad). Against trained everything within +-1 se. Verdict: the better P(call) read in the
plan term does not help; it changes 1 move in 20 rounds and those are neutral to bad.

### 2b. Where is hm5's valuation wrong? Its own second choice, played out

`python -m training.yaniv.contest_fable_reads_run2 {hm5,trained} 12000 81` (hm5 hero; alternative = its second
choice at every throw; 78,143 / 79,689 records). Second choice minus first, by score gap: <1: -0.42 +-0.10 /
-0.42 +-0.10; 1-3: -1.52 / -1.59; 3+: -4.22 / -4.10. hm5's ranking is right in every class I grouped by
(kind of change x regime x my cards after x open low x open <= 4 against 5 cards) EXCEPT one family:
"hm5 takes the open card, the same throw with a deck draw is second, and I end with <= 2 cards":

| class (all score gaps) | vs hm5: n, deck minus open | vs trained |
|---|---|---|
| 2 cards after, the open card pairs nothing | 1,716, +1.19 +-0.38 | 1,780, +0.69 +-0.37 |
| 1 card after | 580, +0.59 +-0.54 | 805, +0.95 +-0.45 |
| 2 cards after, the open card pairs my kept card | 425, -1.69 +-0.88 | 426, -0.71 +-0.86 |
| by my total after the take: 5 | 898, +2.37 +-0.53 | 846, +1.92 +-0.54 |
| ... 4 | 673, +1.46 +-0.53 | 686, +0.36 +-0.59 |
| ... 3 | 445, -0.69 +-0.67 | 594, +0.81 +-0.54 |
| ... 2 | 183, -1.50 +-1.03 | 350, -0.53 +-0.65 |
| by regime: 1-2 cards | 1,386, +0.84 +-0.29 | 1,577, +0.96 +-0.29 |
| ... 3 cards | 756, +1.03 +-0.67 | 840, +0.72 +-0.63 |
| ... 4 cards | 514, -0.31 +-0.90 | 533, -0.71 +-0.83 |

This is the catch bonus (table E): an open take that leaves me at <= 5 gets CATCH x PI ~ +10, and the event it
pays for has probability 0.00.

### 2c. Catch bonus variants (paired, seed 72, 20,000 rounds; KSLOPE=0, OPEN_LOW=0)

| variant | opp | changed a round | each | a round |
|---|---|---|---|---|
| CATCH_MODE=off (513 units) | hm5 | 0.119 | -0.38 +-0.35 | -0.045 (better) |
| | trained | 0.142 | -0.44 +-0.32 | -0.062 |
| CATCH_MODE=k4 (bonus only if I keep >= 3 cards; 562 units) | hm5 | 0.113 | -0.38 +-0.36 | -0.043 |
| | trained | 0.136 | -0.63 +-0.32 | -0.085 |

By my cards after (k4, hm5 / trained): 1 card -1.67 +-0.80 / -1.88 +-0.66; 2 cards -1.00 +-0.47 / -0.78 +-0.48;
**3 cards +1.79 +-0.80 / +0.63 +-0.59 (worse)**. By kind: same throw, I draw from the deck -1.07 +-0.47 / -1.28
+-0.40 (0.07-0.09 a round); same throw, I take the open card +1.86 +-1.05 / +1.13 +-0.84 (0.02 a round).
So removing the bonus at 1-2 cards is right and removing it at 3 cards is not (table E: at 3 cards after a deck
draw the catch happens 0.08-0.30 of the time). Next variant: "k3" = bonus only if I keep >= 2 cards.

### 2d. CATCH_MODE=k3: catch bonus only when I keep >= 2 cards (3-5 after the draw). FRESH seed 73, 20,000 rounds

`python -m training.yaniv.contest_fable_reads_run 'py:contest_fable_reads.build@{"KSLOPE":0,"OPEN_LOW":0,"CATCH_MODE":"k3"}' OPP 20000 73`
(562 units: hm5's 558 + the [cards kept >= 2] step and its carries; the 24 catch units get one more gate input.)

| opp | changed a round | each (hm5's move minus mine) | a round |
|---|---|---|---|
| hm5 | 0.089 | **-1.18 +-0.38** | -0.105 (better) |
| trained | 0.107 | **-1.02 +-0.35** | -0.109 (better) |
| counter | 0.083 | -0.43 +-0.40 | -0.036 |

By my cards after (hm5 / trained / counter): 1 card -1.02 +-0.63 / -1.38 +-0.63 / +0.15 +-0.59; 2 cards -1.58
+-0.50 / -1.34 +-0.48 / -0.64 +-0.54; 3 cards (few: 136 / 234 / 63) +1.39 +-1.18 / +1.54 +-0.82 / -1.78 +-2.18.
By kind: same throw, I draw from the deck where hm5 takes the open card (1,286 / 1,536 / 1,262) -1.29 +-0.46 /
-1.26 +-0.42 / -0.29 +-0.46; same throw, I take the open card (106 / 127 / 185) -5.02 +-2.00 / -3.77 +-1.98 /
-2.29 +-1.57; I throw one card where hm5 throws a pair (122 / 202 / 57) +1.95 +-1.12 / +1.30 +-0.90 / -2.09
+-2.40 (the one class that goes the wrong way, 0.006-0.010 a round).

Kit.trace check of k3 (scratch chk2.py, 300 rounds, 4,200 decisions, 32,826 throw rows, 2,268 with kept <= 1):
every catch unit is 0 on kept <= 1 rows and equal to hm5's on kept >= 2 rows (0 violations; output difference
on kept >= 2 rows and net-against-trace < 3e-4); the best move differs from hm5's at 45 decisions.

## 3. Duels of the best design (k3), seed 31, 20,000 rounds and 6,000 matches each (results in reads-duels.txt)

`python -m training.yaniv.contest_fable_lab duel 'py:contest_fable_reads.build@{"KSLOPE":0,"OPEN_LOW":0,"CATCH_MODE":"k3"}' hm5 trained counter --rounds 20000 --matches 6000 --seed 31 --workers 2`
(the same net: `py:contest_fable_reads.best`, 562 hidden units, layers [125, 220, 100, 66, 35, 33, 53, 29, 26, 1])

| opp | margin a round | rounds won | caught | matches |
|---|---|---|---|---|
| hm5 (head to head) | +0.043 +-0.029 | 50.5% | 6.0% | 3020-2980 (50.3%, 95% 49.1-51.6) |
| trained | +0.223 +-0.082 | 52.1% | 6.8% | 3741-2259 (62.4%, 95% 61.1-63.6) |
| counter | +1.699 +-0.097 | 56.2% | 4.6% | 4223-1777 (70.4%, 95% 69.2-71.5) |

hm5's own baselines (seed 1, 20,000 rounds / 4,000 matches, baselines.txt): trained -0.145 +-0.081, 61.5%;
counter +1.616 +-0.097, 71.7%. Different seeds, so only the head-to-head line is a paired comparison.

Reading: the paired test says +0.10 a round against hm5 and trained (2.9-3.1 se each on a fresh seed) and +0.04
against the counter; the head-to-head duel shows +0.04 +-0.03 and no measurable match gain (50.3%). The gain is
real but small: about 0.1 point a round at most, under 1 point of matches.

## 4. Summary

- Reads that exist (all three opponents): my own card count after the move (strongest), a low open card
  (A/2/joker) just thrown, a low open card against 5 cards, and "it knows the card I took".
- Feeding the better P(call) into the plan term does NOT improve throws (design K: neutral to worse).
- The one place the read pays is the catch bonus, which hm5 pays in situations where the catch cannot happen
  (I hold 1-2 cards, especially after taking the open card). Gating it on "I keep >= 2 cards" gains ~0.1 a round
  in paired tests against hm5 and trained. Removing it altogether (saves 45 units) gains about half of that
  (it is still worth something with 3+ cards).
- Not built / not tested: the 5-card low-open-card signal (base rate too small to move a throw), per-opponent PI
  rows (PI_TABLE="avg" exists in the module but was never run), FEW_ALPHA (exists, never run), CATCH_K3 (exists,
  never run). Usage limit: stopped here on the lead's instruction.

## 5. Note on the base net (added at the end)

contest_fable.py changed while I worked (commit 4ee183e: new Yaniv rule, triple take, 574 units). Everything above
was measured on the hm5-identical base. `contest_fable_reads.build` / `best` now take `yaniv="hm5"`,
`triple=False` as defaults so that `py:contest_fable_reads.best` is still the measured net (562 units; checked:
the neutral build makes hm5's move at 2,624 of 2,624 decisions). `best(yaniv="new", triple=True)` puts the catch
gate on the lead's current net (578 units, +4 on 574); that combination has NOT been played.
To merge by hand: call `contest_fable_reads.fix_catch(kit, "k3")` right after `add_throw_units(kit, cfg)`.
