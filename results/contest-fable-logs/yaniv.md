# yaniv: when to call Yaniv and when to play on (rounds from 0-0)

Thread `yaniv` of team fable. Files: `training/yaniv/contest_fable_yaniv.py` (the design: `build`, `rule`,
`check`), `contest_fable_yaniv_probe.py` (the paired probe), `contest_fable_yaniv_tab.py` (tables),
`contest_fable_yaniv_rules.py` (candidate rules as functions of a position record, for the first-order check).
Scratch (pickles of records, ad hoc table scripts `zone.py two.py twoh.py thr.py swap.py swap2.py r1..r5.py`):
`/tmp/claude-0/-home-user-yaniv-master/ca258bc3-d453-5237-b2d9-a66c415800ab/scratchpad`.
Nothing here fits or searches weights: every number is a count over played rounds, and the rule's constants
are hand-picked from the tables below (every variant tried is listed in "Every measurement").

## Method

`python -m training.yaniv.contest_fable_yaniv_probe run OPP 60000 7101` = `paired("hm5", OPP,
"contest_fable_yaniv_probe:calls", 60000, 7101, workers=2)`. At every decision of the hero (hm5) where Yaniv is
legal:
- hm5 calls: the alternatives `net_throw` (hm5's own best throw row: what the net does if the Yaniv row is
  switched off), `hi_open` (throw the highest-points group, take the open card), `hi_deck` (same throw, deck),
  `single_open`;
- hm5 plays on: the alternative `call`.
Both moves are played to the end of the round on the same hidden cards, hm5's policy afterwards. `v` below =
points margin of CALLING minus that of hm5's best throw (one number per Yaniv-legal decision, either side).
Every record also has the true hands, so "caught" = the opponent's total <= mine (what a call would meet).

Yaniv is legal 0.87 times a round; hm5 calls 0.50 times a round (caught 6.8% by hm5, 7.4% by trained).

