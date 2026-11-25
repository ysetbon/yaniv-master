# twin: keep the high single whose twin the opponent is likely to throw?

Designer `twin`, team opus. Rules: `training/yaniv/contest_opus_twin.py`. Scratch: play-out records and scripts in
the scratchpad `twin/` folder (census.py, run.py, ana.py). Hero = hm5 in every play-out; gain = B's margin − A's
(A = hm5's move), > 0 means the rule's move is better. "occ/rd" = occurrences per round (B ≠ A).

## Situation

Hero holds 5 cards, the opponent ≥ 4, and I hold at least two ranks with exactly one copy worth ≥ HI points
(HI = 8; HI = 7 reported too). In the first version I did NOT exclude positions where the open card pairs one of
my high singles; see variant 0 below - that broke rule (b). Final definition adds: the open card pairs none of my
cards (that is the "pickup" half, measured on its own).

## Variants measured (in order)

### 0. Rule (b) "throw the high single with the fewest unseen copies", first definition (bug) - LOSES
8,000 rounds seed 7101: vs counter −1.74 ± 0.31 each (0.44/rd), vs trained −1.98 ± 0.31 (0.43/rd).
Cause: when the open card is a copy of one of my singles (say open J, I hold J and T), the open J is not
"unseen", so J has the fewest unseen copies and rule (b) threw the J, while hm5 throws the T and takes the J
(528 of 1,271 "10 for 10" cases). That is the pickup position, not the twin question. Fixed by the definition above.
