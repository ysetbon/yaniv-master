# Team fable: shared brief for sub-agents

We are one side of a contest in this repo (branch `claude/contest-fable`). We must hand-write ONE plain net
(`network.MLP`, [125, ..., 1], ReLU hidden, linear output, at most 1,020 hidden units in total) that plays the real
Yaniv game (5 cards each, Yaniv at 7 or less) as strongly as possible. Another team does the same; the two nets
will play 5,000 MATCHES against each other and both are scored against other opponents on unknown seeds.
The other team almost certainly starts from `training/yaniv/handmade5.py` too, so winning MATCHES head to head
against handmade5-like nets is what counts most; margin a round against the counter counts less.

## Hard rules (the contest's)
- No training: no gradient descent, no fitting to labelled moves or outcomes (never call train_dataset,
  train_value, main.py, the search/self-play/genetic recipes), no automated weight search (no genetic
  algorithm, hill climbing, random search, scripted sweeps over coefficients). Never copy weights from a trained net.
- Weights are chosen by reasoning about what a unit should detect and how much it matters. Playing games to
  measure a design, and playing two candidate moves out to the end on the same hidden cards, is testing and is
  fine. Hand-picking a HANDFUL of constants after looking at results is fine; a loop that tunes them is not.
- Log EVERY variant / measurement you run with its reason and its result, failures included, in your own log
  file `results/contest-fable-logs/<thread>.md` (commands, seeds, numbers).
- Touch only: `training/yaniv/contest_fable_<thread>*.py` (your helpers/designs) and
  `results/contest-fable-logs/<thread>*`. Do NOT edit `contest_fable.py`, `contest_fable_lab.py` (the lead owns
  them), engine.py, features.py, heuristic.py, parallel.py, handmade*.py, any existing model file, the README.
  Never set YANIV_HAND_SIZE / YANIV_CALL_AT. Do not git commit or push; the lead does that.
- The machine has 4 cores shared by about 5 agents: pass `workers=2` / `--workers 2`, and do not run more than
  one heavy job at a time. Roughly 250 rounds a second per 2 workers against a net, half that against the counter.

## What exists (read before you start)
- `results/handmade-5x7.md` (last experiment's write-up) and `results/handmade-5x7-logs/` (counts.md, measure.md,
  handover.md ...). `results/handmade-4x7.md` for where the 4-card ideas came from.
- `training/yaniv/handmade_kit.py`: `Kit` (named ReLU units -> MLP) and `INPUTS` (the 125 inputs, their scales).
  `training/yaniv/features.py`: how each input is computed. `engine.py`: rules. `heuristic.py`: the card counter.
- `training/yaniv/contest_fable.py`: our net's builder. Right now it rebuilds handmade5 exactly (558 hidden units:
  `add_throw_units(kit, cfg)`, `add_yaniv_units(kit)`, `add_score_units(kit)`, `finish_yaniv`, `fold_output`;
  `build(**over)` takes the table overrides in `config`). Write your design as your own module, e.g.
  `contest_fable_<thread>.py` with a `build(**kw)` that calls those pieces and adds / replaces units, returning
  `(net, kit)`. Spare budget: about 460 units in total for ALL threads, so be frugal and report your unit count.
- `training/yaniv/contest_fable_lab.py`: the bench. Player strings: `counter`, `hm5` (handmade5's net, the bar),
  `trained`, `hm4`, a .json path, or `py:<module>.<function>[@{json kwargs}]` (e.g.
  `py:contest_fable_twin.build@{"X":2}`).
  - `python -m training.yaniv.contest_fable_lab duel HERO OPP... --rounds N --matches M --seed S --workers 2`
    (mirrored pairs; prints the margin with its paired standard error and matches with a 95% range).
  - `paired(hero, opp, "module:probe", n, seed, mode="rounds"|"matches", workers=2, after="")` plays both moves
    out on the same hidden cards: your probe names alternative moves at the hero's decisions; each record has
    `gain` (points margin of the alternative minus the played move), a0/b0/a1/b1 (points added to hero/opponent
    with each move), s/t (match scores before). `summarize(records, rounds, key=...)` groups them.
    `std_info(view, state)`, `describe(action)` describe positions and moves.
  - `results/contest-fable-logs/wtable.npy`: P(win match | my score, opp score) for two equal hm5 players
    (`numpy.load`; `equity(table, mine, theirs)` applies cuts and the end of the match).
- Baselines (20,000 rounds, 4,000 matches, seed 1): hm5 vs counter +1.62 a round, 55.8% rounds, 71.7% matches;
  hm5 vs trained -0.15 a round, 50.8% rounds, 61.5% matches; trained vs counter +1.62, 54.6%, 64.1%.

## How to work
- Define a rule precisely first (in terms the 125 inputs can express: a single ReLU net must be able to compute
  it), measure it with `paired` against at least two of {counter, hm5, trained} (the unknown rival is a
  hand-written net, so hm5 as the opponent matters most), report points gained per occurrence, how often it
  occurs a round, and the standard error. Group by a priori classes, not by best-in-hindsight.
- Then, if it pays, write the units, check on a few hundred positions that the net really makes the intended
  move (`Kit.trace`), and measure the whole net with `duel` against hm5 (head to head), counter and trained:
  20,000+ rounds (about +-0.09 margin) and, where match play is the point, 6,000+ matches (about +-0.6 points).
  A paired gain that does not show up in the duel is not a gain.
- Beware of rules that only exploit the counter. Report honestly; a clear negative result is useful.
- Your final answer to the lead: (1) the rule(s) in one paragraph each, (2) the paired numbers per opponent,
  (3) the duel numbers of your best design against hm5 / counter / trained with seeds, (4) the exact player
  string that builds your best design and its hidden-unit count, (5) what failed, (6) anything surprising.

## Added after the restart (read this)
- Fairness: main now holds the other team's finished entry and the unseen judging opponent. Do NOT open, read,
  import, load or play against any of: `training/yaniv/contest_opus*.py`, `public/model/yaniv-net-contest-opus.json`,
  `results/contest-opus.md`, `results/contest-opus-logs/`, `training/yaniv/heuristic2.py`, `results/counter2.md`,
  `results/counter2-logs/`, `bot_spec("counter2")`. Do not grep or list inside them either.
- Usage is limited: finish narrowly. Write your result to your log as soon as you have it (the log must stand on
  its own if you are cut off), do not start new lines of exploration, and report.
