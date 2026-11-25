# endgame: the Yaniv-call decision and play after both hands hold <= 4 cards

Designer topic `endgame`. Code: `training/yaniv/contest_opus_endgame.py` (probes `alt_call`, `alt_swap`,
`alt_trained`; block `add_endgame_units`). Scratch: `/tmp/claude-0/-home-user-yaniv-master/3c7d4d81-ef83-553b-adc8-6a5382f1abf0/scratchpad/endgame/`
(record files `alt_*-{counter,trained}.json`, grouping script `grp.py`).

Hero = hm5 (`yaniv-net-5cards-handmade.json`) in every play-out. Gain = B's margin - A's margin on the same hidden
cards, both continued with hm5's policy; > 0 means the alternative was better. "a round" = sum of gains / rounds,
+- sqrt(sum of squares) / rounds. Rounds from 0-0 (no match-score play).

## Log (every measurement, in order)

