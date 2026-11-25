# Topic `match`: playing for the match (log)

Helper: `training/yaniv/contest_opus_match.py`. Scratch: `/tmp/claude-0/-home-user-yaniv-master/3c7d4d81-ef83-553b-adc8-6a5382f1abf0/scratchpad/match/`.

## Instruments

- **Match-win table** (`build_table`): P(win | my score, opp score) between equal players, value iteration
  (match_value.win_table) over 20,000 round outcomes of hm5 against itself (seed 4242). Spot values:
  (0,0) 0.50, (50,0) 0.354, (90,60) 0.189, (60,90) 0.811. Used only to convert the end of a round into
  match-win units; it is not fitted to anything and no weight is read from it.
- **Match play-out** (`match_playout`): whole matches hero vs opponent (mirrored pairs). At every hero decision where a
  rule proposes B != the hero's move A, the round is played to its end both ways on the same hidden cards (hero's
  own policy afterwards) and the record keeps dwin = P(win | scores after B) - P(win | scores after A) and the
  round-margin difference. "dwin a match x100" = points of match-win rate the rule would add per match if applied
  alone (one decision at a time; whole-match tests confirm).

## Probes (hero hm5)

### P1. Flip the Yaniv decision (alt_flip), 2,000 matches vs hm5, seed 7101

Idea: are hm5's call / no-call choices right in MATCH terms at special scores? Regions: Cdie = a caught call ends
the match against me (my score + pts + 30 > 100); Ccut = a caught call lands me on 50/100; Okill = opp score +
opp_est > 100 (a won call likely ends the match for me); Onear = opp score >= 80. Assaf bucket = uniform catch chance.

| tag | n | per match | dwin each x100 (+-se) | dwin a match x100 | dmargin each (+-se) |
|---|---|---|---|---|---|
| call->throw C-/O- a<.05 | 8240 | 4.120 | -3.42 (+-0.16) | -14.104 | -4.51 (+-0.21) |
| call->throw C-/O- a<.15 | 116 | 0.058 | -0.68 (+-1.24) | -0.040 | -0.06 (+-1.70) |
| call->throw C-/Okill a<.05 | 425 | 0.212 | -3.35 (+-1.03) | -0.711 | -3.64 (+-0.88) |
| call->throw C-/Okill a<.15 | 5 | 0.003 | -22.17 (+-9.62) | -0.055 | -6.80 (+-1.83) |
| call->throw C-/Onear a<.05 | 318 | 0.159 | -9.07 (+-1.41) | -1.442 | -2.73 (+-0.97) |
| call->throw C-/Onear a<.15 | 8 | 0.004 | +0.07 (+-4.48) | +0.000 | -1.62 (+-6.12) |
| call->throw Ccut/O- a<.05 | 263 | 0.132 | -5.60 (+-0.78) | -0.736 | -4.24 (+-1.19) |
| call->throw Ccut/O- a<.15 | 81 | 0.041 | -5.00 (+-1.27) | -0.202 | +0.49 (+-2.94) |
| call->throw Ccut/O- a<.3 | 50 | 0.025 | -9.28 (+-1.62) | -0.232 | +4.90 (+-3.96) |
| call->throw Ccut/O- a>=.3 | 22 | 0.011 | -6.25 (+-3.28) | -0.069 | +12.95 (+-6.40) |
| call->throw Ccut/Okill a<.05 | 11 | 0.005 | -9.10 (+-5.48) | -0.050 | -3.09 (+-5.35) |
| call->throw Ccut/Okill a<.15 | 4 | 0.002 | -2.13 (+-2.13) | -0.004 | -10.50 (+-10.17) |
| call->throw Ccut/Onear a<.05 | 16 | 0.008 | -7.74 (+-6.42) | -0.062 | -0.25 (+-3.79) |
| call->throw Ccut/Onear a<.15 | 6 | 0.003 | +10.49 (+-8.94) | +0.031 | -9.33 (+-5.85) |
| call->throw Ccut/Onear a<.3 | 5 | 0.003 | -11.12 (+-9.06) | -0.028 | +15.80 (+-16.24) |
| call->throw Cdie/O- a<.05 | 1868 | 0.934 | -1.73 (+-0.38) | -1.619 | -5.02 (+-0.42) |
| call->throw Cdie/O- a<.15 | 16 | 0.008 | -1.94 (+-4.57) | -0.016 | -8.88 (+-3.62) |
| call->throw Cdie/Okill a<.05 | 190 | 0.095 | -15.06 (+-2.85) | -1.430 | -6.41 (+-1.13) |
| call->throw Cdie/Okill a<.15 | 3 | 0.002 | -3.10 (+-3.10) | -0.005 | -12.67 (+-7.86) |
| call->throw Cdie/Onear a<.05 | 120 | 0.060 | -16.87 (+-3.62) | -1.012 | -6.06 (+-1.34) |
| call->throw Cdie/Onear a<.15 | 4 | 0.002 | -14.67 (+-2.42) | -0.029 | -10.75 (+-1.80) |
| throw->call C-/O- a<.05 | 1083 | 0.541 | -0.56 (+-0.52) | -0.301 | -0.99 (+-0.71) |
| throw->call C-/O- a<.15 | 2877 | 1.438 | -2.40 (+-0.31) | -3.447 | -3.24 (+-0.43) |
| throw->call C-/O- a<.3 | 1896 | 0.948 | -9.37 (+-0.40) | -8.883 | -13.50 (+-0.56) |
| throw->call C-/O- a>=.3 | 1499 | 0.750 | -15.88 (+-0.40) | -11.903 | -24.09 (+-0.55) |
| throw->call C-/Okill a<.05 | 35 | 0.018 | +1.31 (+-3.05) | +0.023 | -1.80 (+-4.38) |
| throw->call C-/Okill a<.15 | 67 | 0.034 | +3.32 (+-2.64) | +0.111 | -2.51 (+-2.75) |
| throw->call C-/Okill a<.3 | 32 | 0.016 | -12.53 (+-3.12) | -0.201 | -19.06 (+-4.06) |
| throw->call C-/Okill a>=.3 | 19 | 0.009 | -6.11 (+-3.32) | -0.058 | -20.26 (+-4.90) |
| throw->call C-/Onear a<.05 | 46 | 0.023 | +2.58 (+-3.50) | +0.059 | -3.54 (+-3.15) |
| throw->call C-/Onear a<.15 | 130 | 0.065 | +0.50 (+-2.08) | +0.032 | -1.22 (+-1.84) |
| throw->call C-/Onear a<.3 | 72 | 0.036 | -1.66 (+-2.27) | -0.060 | -8.21 (+-2.87) |
| throw->call C-/Onear a>=.3 | 85 | 0.043 | -9.76 (+-1.90) | -0.415 | -19.99 (+-2.40) |
| throw->call Cdie/O- a<.05 | 241 | 0.120 | -1.61 (+-1.22) | -0.194 | -0.35 (+-1.67) |
| throw->call Cdie/O- a<.15 | 573 | 0.286 | -4.08 (+-0.79) | -1.168 | -2.88 (+-0.96) |
| throw->call Cdie/O- a<.3 | 362 | 0.181 | -7.69 (+-0.98) | -1.392 | -13.00 (+-1.33) |
| throw->call Cdie/O- a>=.3 | 289 | 0.144 | -12.68 (+-1.20) | -1.832 | -21.20 (+-1.32) |
| throw->call Cdie/Okill a<.05 | 21 | 0.011 | -3.42 (+-8.30) | -0.036 | -0.76 (+-3.72) |
| throw->call Cdie/Okill a<.15 | 36 | 0.018 | -7.08 (+-8.61) | -0.128 | -4.83 (+-3.80) |
| throw->call Cdie/Okill a<.3 | 19 | 0.009 | -8.54 (+-16.25) | -0.081 | -8.79 (+-5.26) |
| throw->call Cdie/Okill a>=.3 | 5 | 0.003 | -45.87 (+-23.64) | -0.115 | -24.60 (+-11.38) |
| throw->call Cdie/Onear a<.05 | 14 | 0.007 | -31.82 (+-8.93) | -0.223 | -2.14 (+-7.66) |
| throw->call Cdie/Onear a<.15 | 51 | 0.025 | -4.11 (+-6.82) | -0.105 | -1.06 (+-2.77) |
| throw->call Cdie/Onear a<.3 | 33 | 0.017 | -15.15 (+-9.61) | -0.250 | -18.03 (+-4.24) |
| throw->call Cdie/Onear a>=.3 | 41 | 0.021 | -35.36 (+-7.82) | -0.725 | -23.83 (+-3.94) |

Reading: nowhere does flipping pay beyond noise. Calling when hm5 calls is right even in "Cdie" (a catch kills me):
withholding the call costs -1.7 (common) to -15/-17 (Okill/Onear) points of match win each time. Calling more
(throw->call) is negative everywhere except tiny cells (C-/Okill a<.15: +3.3 +-2.6, 0.03 a match). **No Yaniv-
threshold change by score is supported.** (Idea "call more aggressively at match point / less when a catch kills
me": rejected by measurement.)

### P2. Throw to fewest kept points (alt_greedy), 1,000 matches vs hm5, seed 7103

Idea: near 100 (my score + hand > 100: Mdie; within 10: Mnear) or when a caught opponent call would push it over
100 (Ocdie), shed points harder than hm5 does. d = kept-points difference bucket.

| tag | n | per match | dwin each x100 (+-se) | dwin a match x100 | dmargin each (+-se) |
|---|---|---|---|---|---|
| M-/O- d1 | 5485 | 5.485 | -1.43 (+-0.19) | -7.870 | -1.50 (+-0.24) |
| M-/O- d2 | 2723 | 2.723 | -1.29 (+-0.29) | -3.513 | -1.37 (+-0.35) |
| M-/O- d3 | 379 | 0.379 | -3.17 (+-0.86) | -1.200 | +4.94 (+-0.95) |
| M-/Ocdie d1 | 890 | 0.890 | -0.29 (+-0.66) | -0.256 | -1.85 (+-0.59) |
| M-/Ocdie d2 | 418 | 0.418 | +0.49 (+-0.97) | +0.206 | +0.01 (+-0.92) |
| M-/Ocdie d3 | 61 | 0.061 | -3.11 (+-3.11) | -0.190 | +6.20 (+-2.87) |
| Mdie/O- d1 | 551 | 0.551 | -0.97 (+-0.83) | -0.534 | -1.27 (+-0.79) |
| Mdie/O- d2 | 241 | 0.241 | -0.16 (+-1.42) | -0.038 | -2.09 (+-1.18) |
| Mdie/O- d3 | 18 | 0.018 | -7.99 (+-6.00) | -0.144 | -2.06 (+-3.66) |
| Mdie/Ocdie d1 | 184 | 0.184 | -5.03 (+-3.12) | -0.926 | -3.73 (+-1.30) |
| Mdie/Ocdie d2 | 75 | 0.075 | +0.25 (+-4.57) | +0.019 | -1.65 (+-2.38) |
| Mdie/Ocdie d3 | 8 | 0.008 | -6.83 (+-13.39) | -0.055 | +2.25 (+-8.81) |
| Mnear/O- d1 | 501 | 0.501 | -3.72 (+-1.02) | -1.861 | +2.02 (+-0.80) |
| Mnear/O- d2 | 333 | 0.333 | -5.33 (+-1.35) | -1.775 | +1.05 (+-1.01) |
| Mnear/O- d3 | 109 | 0.109 | -14.50 (+-2.64) | -1.580 | +13.52 (+-1.50) |
| Mnear/Ocdie d1 | 134 | 0.134 | -4.45 (+-2.69) | -0.597 | +0.19 (+-1.53) |
| Mnear/Ocdie d2 | 82 | 0.082 | +3.05 (+-4.00) | +0.250 | +4.84 (+-2.41) |
| Mnear/Ocdie d3 | 43 | 0.043 | -10.62 (+-4.78) | -0.457 | +6.58 (+-2.97) |

Reading: no region where shedding harder pays (Mnear is clearly negative: -3.7 to -14.5). hm5's throws are fine in
match terms. Rejected.

### P3. Landing the opponent on a cut (upper bound with the truth, alt_cutopp_truth), 2,000 matches vs hm5, seed 7102

Idea: my won call adds the opponent's points; if its score + points == 50/100 it drops to 0/50. Upper bound: when
the (true) landing happens, throw instead of calling.

| tag | n | per match | dwin each x100 (+-se) | dwin a match x100 | dmargin each (+-se) |
|---|---|---|---|---|---|
| cut100 | 561 | 0.281 | +26.09 (+-1.00) | +7.318 | -5.33 (+-0.42) |
| cut50 | 990 | 0.495 | +13.95 (+-0.43) | +6.905 | -5.13 (+-0.31) |

0.78 such calls a match (hm5 steers its own score+hand onto the cuts, so it sits there often): with perfect
information +14 points of match win a match. Hidden-card count at those calls: 0: 70, 1: 309, 2: 559, 3: 434,
4: 159, 5: 20 (of 1551).

### P4. Same, from the view: uniform chance p of the exact landing (alt_cutopp_p), 2,000 matches vs hm5, seed 7104

| tag | n | per match | dwin each x100 (+-se) | dwin a match x100 | dmargin each (+-se) |
|---|---|---|---|---|---|
| cut100 h0 p>=.5 | 29 | 0.015 | +34.01 (+-3.82) | +0.493 | -4.90 (+-1.55) |
| cut100 h1 p<.05 | 7 | 0.004 | +12.37 (+-13.49) | +0.043 | -5.57 (+-1.96) |
| cut100 h1 p<.1 | 66 | 0.033 | +8.62 (+-3.93) | +0.284 | -5.79 (+-1.26) |
| cut100 h1 p<.2 | 53 | 0.026 | +20.87 (+-3.88) | +0.553 | -7.13 (+-1.69) |
| cut100 h1 p<.5 | 31 | 0.015 | +7.04 (+-3.50) | +0.109 | -8.39 (+-1.83) |
| cut100 h2 p<.05 | 134 | 0.067 | +5.45 (+-2.79) | +0.365 | -5.67 (+-1.41) |
| cut100 h2 p<.1 | 210 | 0.105 | +1.99 (+-1.92) | +0.209 | -7.19 (+-1.08) |
| cut100 h2 p<.2 | 16 | 0.008 | -3.70 (+-6.62) | -0.030 | -12.06 (+-5.36) |
| cut100 h3 p<.05 | 718 | 0.359 | -5.00 (+-0.94) | -1.793 | -5.19 (+-0.66) |
| cut100 h3 p<.1 | 578 | 0.289 | -5.31 (+-0.98) | -1.534 | -5.94 (+-0.68) |
| cut50 h0 p>=.5 | 59 | 0.029 | +19.79 (+-1.77) | +0.584 | -6.81 (+-1.11) |
| cut50 h1 p<.05 | 15 | 0.007 | +0.28 (+-5.77) | +0.002 | -11.73 (+-3.04) |
| cut50 h1 p<.1 | 81 | 0.041 | +7.81 (+-2.10) | +0.316 | -6.63 (+-1.27) |
| cut50 h1 p<.2 | 98 | 0.049 | +13.05 (+-1.57) | +0.639 | -6.20 (+-0.83) |
| cut50 h1 p<.5 | 38 | 0.019 | -0.44 (+-2.98) | -0.008 | -5.18 (+-1.80) |
| cut50 h2 p<.05 | 298 | 0.149 | +2.30 (+-1.10) | +0.342 | -4.80 (+-0.81) |
| cut50 h2 p<.1 | 321 | 0.161 | +1.01 (+-0.93) | +0.161 | -6.02 (+-0.81) |
| cut50 h2 p<.2 | 46 | 0.023 | -0.10 (+-2.27) | -0.002 | -7.96 (+-2.04) |
| cut50 h3 p<.05 | 1304 | 0.652 | -3.44 (+-0.42) | -2.241 | -4.85 (+-0.50) |
| cut50 h3 p<.1 | 967 | 0.483 | -2.82 (+-0.50) | -1.364 | -5.25 (+-0.57) |

Reading: with 0-1 hidden cards it pays at every p; with 2 hidden it pays a little; with 3+ it loses (the chance is
too low; a withheld call costs ~3-5 points of match win).

### P5. By hidden count h and d = cut - opp_score - opp known points (alt_cutopp_d), 4,000 matches vs hm5, seed 7105

| tag | n | per match | dwin each x100 (+-se) | dwin a match x100 | dmargin each (+-se) |
|---|---|---|---|---|---|
| h0 d00 | 149 | 0.037 | +21.26 (+-1.51) | +0.792 | -6.05 (+-0.78) |
| h1 d00 | 65 | 0.016 | +10.18 (+-2.67) | +0.165 | -7.40 (+-1.43) |
| h1 d01 | 129 | 0.032 | +18.80 (+-1.91) | +0.606 | -4.19 (+-0.88) |
| h1 d02 | 71 | 0.018 | +17.60 (+-2.84) | +0.312 | -7.11 (+-1.58) |
| h1 d03 | 87 | 0.022 | +14.19 (+-2.07) | +0.309 | -3.77 (+-0.94) |
| h1 d04 | 78 | 0.019 | +11.34 (+-2.95) | +0.221 | -4.92 (+-1.28) |
| h1 d05 | 62 | 0.015 | +13.94 (+-2.73) | +0.216 | -7.94 (+-1.65) |
| h1 d06 | 59 | 0.015 | +7.89 (+-2.65) | +0.116 | -7.44 (+-1.66) |
| h1 d07 | 60 | 0.015 | +2.11 (+-3.59) | +0.032 | -7.67 (+-1.16) |
| h1 d08 | 54 | 0.013 | +7.84 (+-3.40) | +0.106 | -8.44 (+-1.61) |
| h1 d09 | 44 | 0.011 | +10.25 (+-3.45) | +0.113 | -6.32 (+-1.77) |
| h1 d10 | 133 | 0.033 | +5.24 (+-2.00) | +0.174 | -7.16 (+-1.18) |
| h2 d00 | 296 | 0.074 | +5.85 (+-1.51) | +0.433 | -6.42 (+-0.82) |
| h2 d04 | 557 | 0.139 | +5.07 (+-1.05) | +0.706 | -4.69 (+-0.52) |
| h2 d08 | 660 | 0.165 | +3.70 (+-0.95) | +0.611 | -5.02 (+-0.48) |
| h2 d12 | 491 | 0.123 | +1.23 (+-0.98) | +0.151 | -6.10 (+-0.65) |
| h2 d16 | 347 | 0.087 | -2.80 (+-0.93) | -0.243 | -8.96 (+-0.94) |
| h2 d20 | 81 | 0.020 | -0.71 (+-1.72) | -0.014 | -8.32 (+-1.51) |

Reading: h=0 & d=0 (certain): +21 each. h=1, any d in 0..10: +2 to +19 each, all positive. h=2: d 0..11 pays
(+3.7 to +5.9), 12..15 about zero, >=16 negative. Reasoned rule: h<=1 whenever the hidden card can make d; h=2 when
the two hidden cards would be low (d <= 11, average <= 5.5: the cards a steering opponent keeps). Expected:
+0.8 (h0) +2.4 (h1) +1.75 (h2) points of match win a match against hm5, if the gains add.

Same probe against the other opponents (they do not steer onto cuts, so the landings are chance only):

vs trained, 3,000 matches seed 7106:
| tag | n | per match | dwin each x100 (+-se) | dwin a match x100 | dmargin each (+-se) |
|---|---|---|---|---|---|
| h0 d00 | 10 | 0.003 | +37.57 (+-3.63) | +0.125 | -4.30 (+-1.57) |
| h1 d00 | 15 | 0.005 | +5.95 (+-5.06) | +0.030 | -8.73 (+-2.76) |
| h1 d01 | 22 | 0.007 | -1.28 (+-5.40) | -0.009 | -10.95 (+-3.02) |
| h1 d02 | 28 | 0.009 | -4.16 (+-2.65) | -0.039 | -6.07 (+-2.48) |
| h1 d03 | 17 | 0.006 | -3.92 (+-5.39) | -0.022 | -10.65 (+-2.09) |
| h1 d04 | 22 | 0.007 | +2.43 (+-4.42) | +0.018 | -3.14 (+-4.35) |
| h1 d05 | 20 | 0.007 | -0.66 (+-3.43) | -0.004 | -0.05 (+-5.08) |
| h1 d06 | 20 | 0.007 | -4.16 (+-4.53) | -0.028 | -3.05 (+-3.77) |
| h1 d07 | 28 | 0.009 | -2.50 (+-2.51) | -0.023 | -4.11 (+-3.00) |
| h1 d08 | 19 | 0.006 | -1.36 (+-3.05) | -0.009 | -4.95 (+-2.21) |
| h1 d09 | 32 | 0.011 | -1.95 (+-3.59) | -0.021 | +2.34 (+-4.02) |
| h1 d10 | 26 | 0.009 | -0.81 (+-2.53) | -0.007 | -5.77 (+-3.03) |
| h2 d00 | 137 | 0.046 | -4.44 (+-1.88) | -0.203 | -9.27 (+-1.53) |
| h2 d04 | 294 | 0.098 | +0.36 (+-1.35) | +0.035 | -2.44 (+-1.19) |
| h2 d08 | 326 | 0.109 | -1.85 (+-1.30) | -0.201 | -4.41 (+-1.18) |
| h2 d12 | 325 | 0.108 | -1.55 (+-1.02) | -0.168 | -4.39 (+-1.04) |
| h2 d16 | 297 | 0.099 | -0.95 (+-0.94) | -0.094 | -1.45 (+-1.19) |
| h2 d20 | 81 | 0.027 | -4.67 (+-1.63) | -0.126 | -6.32 (+-1.88) |

vs counter, 3,000 matches seed 7107:
| tag | n | per match | dwin each x100 (+-se) | dwin a match x100 | dmargin each (+-se) |
|---|---|---|---|---|---|
| h0 d00 | 7 | 0.002 | +17.67 (+-9.36) | +0.041 | +8.00 (+-5.30) |
| h1 d00 | 20 | 0.007 | -7.41 (+-5.19) | -0.049 | -7.80 (+-3.11) |
| h1 d01 | 35 | 0.012 | +2.53 (+-3.18) | +0.030 | -1.14 (+-2.92) |
| h1 d02 | 27 | 0.009 | +1.80 (+-6.33) | +0.016 | -0.78 (+-2.75) |
| h1 d03 | 33 | 0.011 | -2.23 (+-3.51) | -0.025 | -4.39 (+-1.47) |
| h1 d04 | 23 | 0.008 | -0.49 (+-3.10) | -0.004 | -4.39 (+-1.99) |
| h1 d05 | 37 | 0.012 | +1.28 (+-1.81) | +0.016 | -1.57 (+-2.01) |
| h1 d06 | 35 | 0.012 | +3.89 (+-2.40) | +0.045 | -0.37 (+-2.29) |
| h1 d07 | 26 | 0.009 | +1.56 (+-2.33) | +0.014 | +0.38 (+-3.70) |
| h1 d08 | 23 | 0.008 | +0.32 (+-2.05) | +0.002 | +2.74 (+-3.33) |
| h1 d09 | 37 | 0.012 | -3.31 (+-2.22) | -0.041 | -4.19 (+-2.55) |
| h1 d10 | 35 | 0.012 | +1.81 (+-2.35) | +0.021 | -1.74 (+-2.95) |
| h2 d00 | 201 | 0.067 | -2.09 (+-1.73) | -0.140 | -5.01 (+-0.96) |
| h2 d04 | 327 | 0.109 | -3.73 (+-1.15) | -0.406 | -3.72 (+-0.91) |
| h2 d08 | 327 | 0.109 | -0.11 (+-1.07) | -0.012 | -4.34 (+-0.97) |
| h2 d12 | 300 | 0.100 | -0.83 (+-0.89) | -0.083 | -3.35 (+-1.01) |
| h2 d16 | 314 | 0.105 | -0.86 (+-0.82) | -0.090 | -2.87 (+-0.98) |
| h2 d20 | 70 | 0.023 | -2.34 (+-1.15) | -0.055 | -5.43 (+-1.61) |

Reading: h0/h1 about neutral against both (sum -0.1 / +0.06 a match); h2 d 0..11 costs about -0.4 (trained) /
-0.56 (counter) points a match. The gain is an answer to hm5's own cut steering.

### P6. h = 3 (alt_cutopp_d3), 2,000 matches vs hm5, seed 7110

| tag | n | per match | dwin each x100 (+-se) | dwin a match x100 | dmargin each (+-se) |
|---|---|---|---|---|---|
| h3 d00 | 51 | 0.025 | -8.85 (+-3.06) | -0.226 | -11.90 (+-2.10) |
| h3 d04 | 258 | 0.129 | -0.77 (+-1.72) | -0.099 | -4.68 (+-1.01) |
| h3 d08 | 405 | 0.203 | -4.30 (+-1.33) | -0.872 | -4.62 (+-0.82) |
| h3 d12 | 419 | 0.209 | -4.20 (+-1.15) | -0.880 | -4.56 (+-0.68) |
| h3 d16 | 370 | 0.185 | -3.70 (+-1.22) | -0.684 | -4.92 (+-0.92) |
| h3 d20 | 372 | 0.186 | -3.78 (+-1.03) | -0.703 | -5.91 (+-1.00) |
| h3 d24 | 442 | 0.221 | -3.77 (+-0.67) | -0.833 | -7.89 (+-0.84) |
| h3 d28 | 377 | 0.189 | -3.12 (+-0.76) | -0.589 | -6.31 (+-1.03) |

Negative at every d: no veto with 3+ hidden cards.

### P7. h = 2 split by known opponent cards k (alt_cutopp_h2k), 2,000 matches each, seed 7111

vs hm5:
| tag | n | per match | dwin each x100 (+-se) | dwin a match x100 | dmargin each (+-se) |
|---|---|---|---|---|---|
| h2 k0 cut100 | 23 | 0.011 | -0.72 (+-4.23) | -0.008 | -7.26 (+-3.05) |
| h2 k0 cut50 | 34 | 0.017 | -8.68 (+-2.44) | -0.148 | -4.65 (+-3.20) |
| h2 k1 cut100 | 157 | 0.079 | +3.47 (+-2.63) | +0.272 | -8.54 (+-1.21) |
| h2 k1 cut50 | 297 | 0.148 | +4.15 (+-1.21) | +0.616 | -3.91 (+-0.83) |
| h2 k2 cut100 | 68 | 0.034 | +23.28 (+-3.46) | +0.792 | -4.90 (+-1.43) |
| h2 k2 cut50 | 134 | 0.067 | +5.61 (+-1.53) | +0.376 | -5.60 (+-1.01) |
| h2 k3 cut100 | 10 | 0.005 | +2.96 (+-4.07) | +0.015 | -3.70 (+-2.20) |
| h2 k3 cut50 | 16 | 0.008 | +5.00 (+-5.14) | +0.040 | -1.12 (+-1.07) |

vs counter:
| tag | n | per match | dwin each x100 (+-se) | dwin a match x100 | dmargin each (+-se) |
|---|---|---|---|---|---|
| h2 k0 cut100 | 17 | 0.009 | -5.74 (+-5.02) | -0.049 | -1.35 (+-5.04) |
| h2 k0 cut50 | 11 | 0.005 | -2.07 (+-5.64) | -0.011 | +2.82 (+-6.31) |
| h2 k1 cut100 | 153 | 0.076 | -5.46 (+-2.13) | -0.418 | -6.39 (+-1.23) |
| h2 k1 cut50 | 192 | 0.096 | +0.26 (+-1.24) | +0.025 | -2.67 (+-1.34) |
| h2 k2 cut100 | 66 | 0.033 | -2.05 (+-3.32) | -0.068 | -3.61 (+-1.42) |
| h2 k2 cut50 | 99 | 0.050 | -2.70 (+-1.40) | -0.134 | -4.54 (+-1.31) |
| h2 k3 cut100 | 2 | 0.001 | -6.20 (+-4.78) | -0.006 | -9.00 (+-5.00) |
| h2 k3 cut50 | 6 | 0.003 | -1.50 (+-1.23) | -0.005 | -1.67 (+-1.67) |

vs trained:
| tag | n | per match | dwin each x100 (+-se) | dwin a match x100 | dmargin each (+-se) |
|---|---|---|---|---|---|
| h2 k0 cut100 | 66 | 0.033 | -0.42 (+-3.83) | -0.014 | -2.20 (+-2.22) |
| h2 k0 cut50 | 76 | 0.038 | -0.08 (+-2.12) | -0.003 | -6.96 (+-1.96) |
| h2 k1 cut100 | 127 | 0.064 | -0.62 (+-2.60) | -0.039 | -4.89 (+-1.73) |
| h2 k1 cut50 | 181 | 0.090 | -3.56 (+-1.33) | -0.322 | -5.74 (+-1.39) |
| h2 k2 cut100 | 27 | 0.013 | -4.19 (+-3.87) | -0.057 | -7.52 (+-3.35) |
| h2 k2 cut50 | 32 | 0.016 | -4.90 (+-3.86) | -0.078 | -3.88 (+-3.37) |
| h2 k3 cut100 | 1 | 0.001 | -7.52 (+-0.00) | -0.004 | -6.00 (+-0.00) |

Reading: requiring k >= 1 would drop small negative cells (about +0.2 a match total, within noise). Not adopted
(not worth slicing further on one seed).

### P8. Is hm5's own cut steering (OPEN_CUT) well weighted? (alt_steer)

Idea: if the hero's move lands its score + hand on 50/100 via the open card, B = its best non-landing move ("drop");
if it does not but a landing take exists, B = the best landing take ("take"). t1 = opp_est <= 9, t0 >= 15, tm between.

hero hm5 vs hm5, 1,500 matches seed 7108:
| tag | n | per match | dwin each x100 (+-se) | dwin a match x100 | dmargin each (+-se) |
|---|---|---|---|---|---|
| drop cut100 t0 | 495 | 0.330 | -1.34 (+-1.09) | -0.442 | +0.64 (+-0.91) |
| drop cut100 t1 | 539 | 0.359 | -17.27 (+-1.37) | -6.205 | +4.30 (+-0.59) |
| drop cut100 tm | 709 | 0.473 | -8.61 (+-1.17) | -4.071 | +4.91 (+-0.67) |
| drop cut50 t0 | 918 | 0.612 | +0.35 (+-0.54) | +0.214 | +0.72 (+-0.68) |
| drop cut50 t1 | 867 | 0.578 | -8.27 (+-0.63) | -4.778 | +5.82 (+-0.44) |
| drop cut50 tm | 1210 | 0.807 | -3.22 (+-0.54) | -2.598 | +5.27 (+-0.47) |
| take cut100 t0 | 2073 | 1.382 | -0.33 (+-0.56) | -0.451 | -7.55 (+-0.47) |
| take cut100 t1 | 7 | 0.005 | +20.76 (+-13.77) | +0.097 | -19.00 (+-4.84) |
| take cut100 tm | 287 | 0.191 | +2.10 (+-1.72) | +0.401 | -13.32 (+-1.19) |
| take cut50 t0 | 3329 | 2.219 | -0.71 (+-0.27) | -1.585 | -6.91 (+-0.37) |
| take cut50 t1 | 7 | 0.005 | +7.76 (+-5.29) | +0.036 | -15.00 (+-2.81) |
| take cut50 tm | 454 | 0.303 | +0.49 (+-0.79) | +0.148 | -10.77 (+-0.91) |

hero v2 vs v2 (an opponent that vetoes, see below), 1,500 matches seed 7109:
| tag | n | per match | dwin each x100 (+-se) | dwin a match x100 | dmargin each (+-se) |
|---|---|---|---|---|---|
| drop cut100 t0 | 530 | 0.353 | -1.23 (+-1.09) | -0.434 | +1.69 (+-0.93) |
| drop cut100 t1 | 840 | 0.560 | -6.62 (+-0.91) | -3.708 | +1.84 (+-0.44) |
| drop cut100 tm | 756 | 0.504 | -3.56 (+-1.00) | -1.792 | +4.04 (+-0.63) |
| drop cut50 t0 | 876 | 0.584 | -0.62 (+-0.48) | -0.363 | -0.12 (+-0.65) |
| drop cut50 t1 | 1377 | 0.918 | -4.58 (+-0.43) | -4.207 | +1.84 (+-0.34) |
| drop cut50 tm | 1131 | 0.754 | -0.56 (+-0.49) | -0.424 | +3.43 (+-0.54) |
| take cut100 t0 | 1852 | 1.235 | -1.99 (+-0.53) | -2.457 | -5.84 (+-0.47) |
| take cut100 t1 | 16 | 0.011 | +0.83 (+-1.96) | +0.009 | -2.31 (+-1.42) |
| take cut100 tm | 245 | 0.163 | +1.56 (+-1.57) | +0.255 | -10.80 (+-1.12) |
| take cut50 t0 | 2823 | 1.882 | -1.64 (+-0.28) | -3.077 | -5.71 (+-0.39) |
| take cut50 t1 | 30 | 0.020 | -1.84 (+-0.80) | -0.037 | -2.83 (+-1.20) |
| take cut50 tm | 431 | 0.287 | -1.16 (+-0.76) | -0.332 | -9.80 (+-0.95) |

Reading: dropping a landing costs a lot (-17 / -8 each at t1 against hm5, still -6.6 / -4.6 against an opponent
with the veto); taking more is within noise (tm cut100 +2.1 +-1.7, 0.19 a match). hm5's steering weights are about
right and still pay against an opponent that has this veto. No change.

## The block: `add_match_units` (no Yaniv call that drops the opponent onto a cut)

Rule (reasoned from P5): veto the Yaniv row (-2000 = 2 YBIG, so a called row goes from +1000 to -1000, under every
throw) when, for the next cut (50 if opp_score < 50 else 100; the other cut is out of reach of 0-1 hidden cards
automatically), with h = opponent's hidden cards and d = cut - opp_score - known points:
h = 0 and d = 0; or h = 1 and 0 <= d <= 10; or (v2 only) h = 2 and 0 <= d <= 11;
and cut - opp_score > my points (otherwise the call is caught and the opponent adds nothing).
All inputs are exact integers after scaling (opp_score x120, known_pts x40, opp_cards / known_n x5, my_pts x40), so
every step is exact. Units: layers 1-2 only, reading inputs; v1 (add_match_units_h01) 20 units, v2 (add_match_units)
31 units. Checked against the Python rule on 32,440 decisions of hm5-vs-hm5 matches: the nets differ from hm5
exactly where the rule fires (59 times v1, 171 times v2), nowhere else.

### Whole matches (contest_opus_eval, mirrored pairs, same seeds for all nets; rounds from 0-0 are unchanged by
construction - the rule needs an opponent score within reach of a cut - 2,000 rounds seed 301 identical to hm5)

| net | vs hm5 s301 | vs hm5 s302 | vs trained s301 | vs trained s302 | vs counter s301 | vs counter s302 |
|---|---|---|---|---|---|---|
| hm5 | 800-800 (50.0%) | (50%) | 967-633 (60.4%) | 1000-600 (62.5%) | 1149-451 (71.8%) | 1169-431 (73.1%) |
| v1 (h0+h1, 20 units) | 873-727 (54.6%) | 843-757 (52.7%) | 962-638 (60.1%) | 1001-599 (62.6%) | 1148-452 (71.8%) | 1172-428 (73.2%) |
| v2 (h0+h1+h2, 31 units) | 916-684 (57.2%) | 902-698 (56.4%) | 952-648 (59.5%) | 978-622 (61.1%) | 1126-474 (70.4%) | 1157-443 (72.3%) |

Pooled 3,200 matches each, change against hm5 on the same deals: **v1: +3.6 (hm5 head to head), -0.1 (trained),
+0.1 (counter); v2: +6.8 (head to head), -1.2 (trained), -1.1 (counter).** Noise: 3,200 matches ~ +-1.7 points
(95%) for an absolute rate; the paired differences vs hm5 only move on the ~0.03-0.4 matches a game where the
rule fires, so they are tighter than that, but treat +-1 as noise. Raw: eval_v1v2_s301.txt, eval_v1v2_s302.txt
in the scratch dir.

Recommendation: v2 if the contest opponent is hm5-like (it steers onto cuts and has no veto: +6.8 points of matches);
v1 if we want no cost against non-steering opponents (+3.6 head to head, neutral elsewhere).

## Ideas rejected (measured above)
- Score-dependent Yaniv threshold (call more at opponent match point, less when a catch would kill me): P1, no cell
  pays beyond noise; withholding calls costs at every score region.
- Shed points harder near 100 / when a caught opponent call would kill it: P2, negative everywhere.
- Veto with 3+ hidden cards: P6, negative.
- Re-weighting hm5's cut steering: P8, already about right.
