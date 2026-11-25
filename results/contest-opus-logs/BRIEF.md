# Team opus: common brief for every designer

Contest: write by hand ONE plain net (network.MLP, [125, ..., 1], ReLU hidden, linear out, at most 1,020 hidden
units in total) that plays the real Yaniv game (5 cards, Yaniv at 7) as strongly as possible. It will play
5,000 matches head to head against another team's hand-written net, and be scored against other opponents on
unknown seeds. Matches matter most; round margin second.

## Rules you must keep
- NO training: no gradient descent, no fitting to moves/outcomes, never call train_dataset/train_value/main.py or
  search/self-play/genetic code, never copy weights from a trained net.
- NO automated weight search: no loops/sweeps/hill-climbing over coefficients. Choose each weight by reasoning;
  hand-picking a handful of values after looking at a result is fine. A loop over *positions* in a play-out test
  is fine; a loop over *coefficients* is not.
- Log EVERY variant you measure (reason + result, failures too) in your log file.
- Touch only: your own `training/yaniv/contest_opus_<topic>.py` and `results/contest-opus-logs/<topic>*.md`
  (plus scratch files in /tmp/claude-0/-home-user-yaniv-master/3c7d4d81-ef83-553b-adc8-6a5382f1abf0/scratchpad/<topic>/).
  Do NOT edit contest_opus.py (the lead integrates), engine/features/heuristic/parallel, handmade*.py, model files.
  Do not git commit or push. Do not set YANIV_HAND_SIZE / YANIV_CALL_AT at import time in library code (leave them
  unset: unset = the real game).
- CPU: the machine has 4 cores shared by 4 designers. Use at most 2 processes per run
  (`playout(..., processes=2)`, `contest_opus_eval --workers 2`). Prefer play-outs (cheap) over whole-game tests.

## Starting point
- `training/yaniv/handmade5.py` = the current best hand-written net ("hm5", 558 units). Read
  `results/handmade-5x7.md` first (and `results/handmade-4x7.md` if useful). Unit-writing patterns: handmade.py,
  handmade5.py, handmade_kit.py (Kit; INPUTS lists the 125 inputs and their scales; `kit.trace(x)` shows units).
- `training/yaniv/contest_opus.py`: `build(extras=[f, ...])` = hm5 plus your blocks; each `f(kit)` adds named units
  (kit.unit(layer, name, weights, bias, note)) and output terms (kit.out({...})). Units may read inputs or units
  from earlier layers (hm5 units by name, e.g. G/plan units — see `python -m training.yaniv.contest_opus -v`).
  Save a variant: `from training.yaniv.network import save_net; save_net(net, path, meta={})`.
  Spare budget: about 460 units for the whole team; aim to use <= ~120 for your block.
- `training/yaniv/contest_opus_playout.py`: paired play-outs. `playout(hero, opp, "module:fn", rounds, seed,
  processes=2)` returns (records, rounds, margin); at each hero decision your fn(view, actions, a, ctx) may return
  an alternative move B (or (B, tag)); both A and B are played to the end on the same hidden cards with the hero's
  policy; record gain = B's margin - A's margin (>0: B better). `summary(recs, rounds)` groups by tag. Your rule
  module must be importable from the workers (put it in training/yaniv/contest_opus_<topic>.py, or add your scratch
  dir to PYTHONPATH). ctx["state"]/ctx["opp_hand"] are the truth: use them only for logging, never in a rule.
  Players: "counter", "trained", "hm5", "hm4", or a net path.
- `python -m training.yaniv.contest_opus_eval NET.json [more] --opps counter,trained,hm5 --rounds 4000
  --matches 800 --seed S --workers 2`: whole-game rounds (margin = opponent's points - mine, a round) and matches.
  Noise: 4000 rounds ~ +-0.2 margin; 800 matches ~ +-3.4 points. Mirrored pairs, so two nets on the same seed
  compare better than that.
- Engine: engine.py (State, View, legal_actions, apply_action; stock order is fixed so play-outs are exact pairs),
  features.py (state_features, the 125 inputs), heuristic.py (the "card counter" opponent).
- Check your gains against at least two opponents (counter AND trained); a trick that only beats the counter does
  not count.

## Deliverable (your final message)
Numbers: each rule tested, how often it occurs (per round), points gained per occurrence (+- se), against which
opponents; each net variant measured and its result vs hm5 on the same seeds. If something pays, leave a
function `add_<topic>_units(kit)` in your helper file that the lead can pass to build(extras=[...]), with its
unit count, and the path of a built variant json. Be honest about noise.
