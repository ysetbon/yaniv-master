# Instructions common to the four designers

Read `BRIEF.md` in this folder first, completely, and follow its rules (no git changes; no training or
automated search of any kind; at most 2 processes, one evaluation at a time).

You write ONE design of a hand-written net for the real game (5 cards, Yaniv at 7), independently of the other
designers (do not read their design files or logs: independence is the point; merging comes later).

- Your design file: `training/yaniv/handmade5_designs/<NAME>.py`. It must expose `build() -> (net, kit)`
  (like `training/yaniv/handmade.py`), build the net from named `Kit` units with each unit's purpose in its
  note, keep every constant named at the top with a comment saying where the number comes from, set no env
  vars, and when run as `python -m training.yaniv.handmade5_designs.<NAME> [OUT.json]` save the net with
  `network.save_net` (default OUT: a file in your scratch dir). You may import and reuse functions from
  `training/yaniv/handmade.py` / `handmade_kit.py` / `handmade_designs/*`, or copy code from them into your
  file to parametrise it. Do not edit any existing file.
- Your log: `results/handmade-5x7-logs/<NAME>.md`: a table of EVERY variant you measured (what changed, the
  reason, working-test result, kept or dropped), then what worked, what did not, and what you would try next.
- What is measured: your net is meant to play **while either player still holds 5 cards**; once both hands
  hold 4 or fewer the 4-card net takes over. So your primary number is the switch player
  `switch:YOURS.json:public/model/yaniv-net-4cards-yaniv7-handmade.json` on the working test (`--work`):
  matches won first, then margin and rounds won. Also report your net alone (whole game), once per kept design.
  The bar: the 4-card net playing the whole game unchanged (its working-test numbers are in BRIEF.md once the
  lead has appended them; if not there yet, measure it yourself once). Your net must still be complete (score
  every legal move sensibly, including Yaniv calls: I can hold 4 cards and be able to call while the opponent
  still holds 5) and should keep the match-score play (aim at exactly 50 / 100).
- Work in small steps with reasons: first reproduce a sane baseline, then one change at a time, each stated as
  "I expect X because Y", measured on the working test (rounds-only `--matches 0` is 4x quicker when a change
  cannot affect match-score play). Differences under about 0.2 margin / 2.5 points of matches are noise: do
  not chase them, and do not hand-tune a coefficient through more than 3-4 values. A diagnostic that shows
  WHERE your net loses points (which kind of position, by card counts) is worth more than another variant.
- Check `results/handmade-5x7-logs/measure.md` (the measuring agent's real-game tables) when you start and
  again after ~15 and ~30 minutes; measure yourself what is missing.
- Budget: aim to finish in about 60-90 minutes of work. Leave no background processes.
- Final reply to the lead, under 350 words: the idea, the final working-test numbers (switch and alone, next
  to the bar), the 3-5 changes that mattered most with their size, what failed, and the path of your saved
  net JSON.
