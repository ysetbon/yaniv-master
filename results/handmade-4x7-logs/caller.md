# caller: variants log (official test unless noted: 2,000 rounds seed 41000, 800 matches seed 41001, --workers 2)

Throw policy T0 (all variants unless noted): score = -(40*pts_kept + 10*incoming) + 0.7*[open card's points if it pairs a kept card]
+ 1*keeps_pair + 1.5*callable_next - 1*low_top. Yaniv row: +-1000 via a gated 0/1 step.

1. caller_t0 (saved then as caller.json; T0 + call rule C1: call iff 40*opp_est - 40*my_pts - (0.4*my_pts_pts + 1.3) - 10*assaf > 0, or sure_yaniv).
   Why: counter self-play diagnostics show assaf understates catch risk 3-8x; opp_est - my_pts separates good/bad calls,
   break-even ~3 pts at 4-5 in hand and ~4 at 6-7.
   -> margin -0.78 a round, rounds won 894/2000 (44.7%), Yaniv calls 862 of which caught 40 (4.6%); matches 339-461 (42.4%, 95% range 39.0%-45.8%)
2. caller_ctr (T0 + the counter's own call rule as units: assaf <= _call_threshold(my_pts), or sure_yaniv). Baseline for the call rule.
   -> margin -1.00 a round, rounds won 920/2000 (46.0%), Yaniv calls 923 of which caught 64 (6.9%); matches 331-469 (41.4%, 95% range 38.0%-44.8%)
3. caller_n2 (T0 + C1 with margin shifted by the opponent's card count: +1.5 pts at 2 cards, -0.75 at 3 and 4).
   Why: in both diagnostic runs, at equal opp_est - my_pts the counter's 2-card hands were caught clearly less often
   (e.g. e-m in [2,3): 23% vs 32% at 3 cards), so break-even is ~3 at 2 cards and ~4 at 3 cards.
   -> margin -0.75 a round, rounds won 894/2000 (44.7%), Yaniv calls 853 of which caught 32 (3.8%); matches 342-458 (42.8%, 95% range 39.4%-46.2%)
4. caller_t1 (throw T1 + call rule of #3). T1 = T0 + take_open costs 1.5 points (a deck draw beats its average: high draws
   are just the next card thrown, and the opponent does not see it) + 0.5 more of a pairing open card's points while the
   opponent holds 4 cards (no call soon, so the hand after the next throw is what counts).
   Why: the biggest disagreement groups vs the counter were "net takes a 5/6 open card, counter draws" and "counter takes
   a high card that pairs, net does not".
   -> margin +0.63 a round, rounds won 1013/2000 (50.6%), Yaniv calls 964 of which caught 48 (5.0%); matches 431-369 (53.9%, 95% range 50.4%-57.3%)
5. caller_t1_ctr (throw T1 + the counter's call rule as units). Baseline for #4.
   -> margin +0.37 a round, rounds won 1036/2000 (51.8%), Yaniv calls 1031 of which caught 79 (7.7%); matches 430-370 (53.8%, 95% range 50.3%-57.2%)
6. caller_t2 (throw T2 = T1 but the open-card cost by opponent card count, 2.0 at 4 / 0.5 at 3 / 0 at <=2, and a graded
   face-up penalty 0.4*(4.5 - top points) instead of the flat low_top 1.0; call rule of #3).
   Why: remaining disagreements were "counter takes a 5 late, net draws" and "net keeps a low pair to avoid leaving a 4 up".
   -> margin +0.25 a round, rounds won 981/2000 (49.0%), Yaniv calls 943 of which caught 46 (4.9%); matches 394-406 (49.2%, 95% range 45.8%-52.7%)
   Worse than T1 (+0.63): matching the counter's draw choices more closely did not help; reverted to T1.
7. caller_c2 (T1 + call rule C2: threshold on opp_est - my_pts = 0.8*my_pts + 0.7, i.e. 3.9/4.7/5.5/6.3 at 4/5/6/7, no
   card-count shift). Why: re-running the call-vs-continue diagnostic on caller_t1's own games put break-even ~1.5-2 points
   above C1's line, uniformly over opponent card counts (the 2-card shift of #3 did not reproduce).
   -> margin +0.55 a round, rounds won 960/2000 (48.0%), Yaniv calls 869 of which caught 21 (2.4%); matches 422-378 (52.8%, 95% range 49.3%-56.2%)
   No gain over #4 (+0.63): the one-step "call now vs. play on" rollout overstates the value of waiting.
8. caller_t1_c1 (T1 + C1 exactly, no card-count shift). Why: the 2-card shift of #3 did not reproduce on T1's games.
   -> margin +0.70 a round, rounds won 1013/2000 (50.6%), Yaniv calls 964 of which caught 47 (4.9%); matches 416-384 (52.0%, 95% range 48.5%-55.4%)
   Same as #4 within noise; simpler, so preferred.
9. caller_t3 (T1 + C1 + throw the rank with fewer unseen copies, -0.3 per unseen copy of the thrown rank; 14 gate units).
   Why: the largest remaining disagreement group was ties among 10-point cards, where the counter keeps the rank a draw can pair.
   -> margin +0.64 a round, rounds won 1003/2000 (50.1%), Yaniv calls 953 of which caught 43 (4.5%); matches 424-376 (53.0%, 95% range 49.5%-56.4%)
   No gain; dropped.
10. caller_flat (T1 + one flat cut: call iff assaf < 0.02 or sure_yaniv). Why: test whether opp_est is needed, or whether
    the gain over the counter's rule is just "be stricter" (calibration table: assaf < 0.02 is caught ~5% in reality).
    -> margin +0.66 a round, rounds won 968/2000 (48.4%), Yaniv calls 882 of which caught 24 (2.7%); matches 421-379 (52.6%, 95% range 49.2%-56.1%)
    Same margin as C1 (+0.70) but fewer rounds won; most of the gain over the counter's rule is being stricter than its 0.04-0.08 cuts.

BEST: caller_t1_c1, saved as nets/caller.json (python -m training.yaniv.handmade_designs.caller).
