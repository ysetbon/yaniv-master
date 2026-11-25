# Yaniv

**Play it in your browser: https://ysetbon.github.io/yaniv-master/**

A two-player game of Yaniv. You sit across from a neural network that learned the table by copying a card counter and then playing thousands of games against itself.

A hand is five cards from a 54-card deck with two jokers. Ace is 1, number cards are their pips, 10, jack, queen, and king are 10, and a joker is 0. Throw a single card or a set of one rank (the two jokers pair with each other; there are no runs), then draw the previous open card or a blind card from the deck. With 7 points or fewer you may call Yaniv. If the other hand is the same or lower, the call is Assaf and costs your points plus 30. A score that lands exactly on 50 or 100 is cut by 50. The match ends when a score reaches 100. The lower score wins.

## Run it

```bash
npm install
npm run dev
```

The trained weights already live in `public/model`, so you can play without training again. The same game is published on every push to `main` by `.github/workflows/pages.yml` (a static export under `/yaniv-master`; it needs Settings > Pages > Source: "GitHub Actions" once). Open the app and deal a match. Select cards, then choose whether the next card comes from the deck or from the open pile. "Ask the net" shows the throw the network would make in your seat.

The standard game (http://localhost:3457/) plays team fable's contest net, a net for the real game whose weights were written by hand rather than trained (`training/yaniv/contest_fable.py`, results in `results/contest-fable.md`). It beats the trained net and the earlier hand-written net in matches. "Standard: trained net" (http://localhost:3457/?game=trained) plays the trained net described below, and "Standard: earlier hand-written net" (http://localhost:3457/?game=5x7-handmade) the hand-written net before it.

The menu at the top switches games. "4 cards, Yaniv 7: hand-written net" (http://localhost:3457/?game=4x7-handmade) plays the 4-card game against a net whose weights were written by hand rather than trained (`training/yaniv/handmade.py`, results in `results/handmade-4x7.md`).

## The network

Training lives in `training/yaniv`. It plays a full Yaniv rules engine, scores every legal throw with a small multilayer perceptron, and writes `public/model/yaniv-net.json`.

The net reads the table by rank only, because no rule looks at suits: 96 numbers for the table (ranks in your hand, thrown, the open card, the opponent's known cards and the unseen cards; the scores and counts; the opponent's card count; its estimated hand total, leaning low as real hands do; and the exact chance a Yaniv call now gets caught) and 29 for each move. Moves that differ only by suit, such as which 7 of three is thrown, are one move. The layout is listed in `training/yaniv/features.py`, and `src/lib/yaniv/features.ts` must match it number for number. The net is 125 → 128 → 128 → 1, about 33,000 weights. Nets trained on the older 389-number layout still run through `features_v1.py`, so they can be copied.

```bash
python3 -m training.yaniv.train
npm run test:rules
```

`test:rules` replays fixture games from the Python engine through the TypeScript rules so the table and the trainer cannot drift.

## Watch it train

```bash
pip install -r training/requirements.txt
python main.py
```

`main.py` trains the network and opens a dashboard at http://127.0.0.1:8765:

- **Summary at the top:** a step bar for the whole run, time left, the best generation so far, and its win rate against the card counter with a 95% range.
- **Tables:** six exhibition games run side by side (`--tables` sets how many), separate from the training games, which run too fast to watch. Select one to feature it: the featured table shows both hands face up, a short pause before each move that lifts the cards about to be thrown, the top moves the net weighed, and what Gen 0 would have played instead. Presets seat the newest generation against the previous one, the newest or best against the counter, or the best against Gen 0. When a new generation arrives, every table switches to it. You can pause, step one move at a time (Space and →), and change speed.
- **Is it improving?** Win rate against the counter per generation with 95% bands, a ladder where each generation plays every older one, an Elo rating per generation fitted from all of those rounds (card counter at 0), how often each generation calls Yaniv and gets caught, points per round, what each pass trained on, and the training loss.
- **Final check:** at the end, the top four generations by their own tests replay one shared set of games, and the best of them is the gate's pick. The pick and Gen 0 then both play new games. The better of the two ships.
- **Buttons:** *Stop after this generation* ends early and still tests and saves the best. *Use best in game* writes it to `public/model/` for the game.
- **Past runs:** results from earlier runs saved under `runs/`.

Generation 0 is always the copy of the card counter. After that, `main.py` learns from a search teacher by default. `--recipe genetic` evolves a population instead, and `--recipe selfplay` uses the first self-play recipe.

### How the search recipe works (the default)

The card counter only looks one move ahead. The search player looks further: for each decision it imagines 48 deals that fit what it can see, plays every candidate move to the end of the round in each of them, and keeps the move with the best average result. The net then learns to pick the searched moves.

1. The best net so far plays the card counter for `--search-rounds` rounds (default 150). Two in five start from random match scores, so play near 100 gets practice.
2. At every decision with a real choice, the search compares the net's own move, its four next-best plays, and a Yaniv call when legal. It plays each one out with the net in both seats. A move's value is the point margin of the round, plus or minus 100 when the round ends the match.
3. The opponent's hidden cards are imagined leaning low. The counter throws high cards and keeps low ones, so by mid-round its hidden cards average about 3 points, not the 6 of a random draw.
4. Another move replaces the net's own only when it wins on the same imagined deals by 1.5 standard errors (`--search-confidence`). Without that bar the search switched on luck between close moves and lost to the counter.
5. The net learns the searched moves, mixed with a share of the counter's own moves that shrinks from one each to one in four (`--copied-share`). Starting from a trained net, none are mixed in: they would only pull it back toward the counter.
6. Each pass's net is the next pass's teacher, so the playouts get better as the net does. Each pass is tested on 1,000 rounds and 200 matches against the counter. The gate ranks passes by point margin per round, then matches, because matches are decided by total points, and keeps the best one to ship.

```bash
python main.py --generations 12 --search-rounds 300   # about an hour on 10 game processes
```

To keep improving the net that is in the game instead of starting over from a copy of the counter, start from it. It becomes Gen 0, so only a generation that beats it on new games ships:

```bash
python main.py --start-from public/model/yaniv-net.json --generations 12 --search-rounds 300
```

Any saved net works, such as a run's `runs/<timestamp>/best.json`.

To start a new net from a saved one's moves instead of the counter's, copy it. This also moves a net onto a new input layout:

```bash
python main.py --copy-from runs/20261123-091156/best.json --generations 12 --search-rounds 300
```

`--effort 0.7` uses about 70% of the computer: game processes on 70% of the CPU's logical cores, at below-normal priority so the computer stays responsive, and a 70% cap on GPU memory. The GPU only runs the learning step, a few seconds a pass. The search spends its time on game rules and on building the net's inputs in Python, and asks the net about one position at a time, so it stays on the CPU, where it is faster than sending each small batch to the GPU.

### How the genetic algorithm works (`--recipe genetic`)

1. The first population is the copy plus 23 mutants of it. Every member keeps the same shape as the copy (125 → 128 → 128 → 1), so neuron *i* means roughly the same thing in every member.
2. Each generation, every member plays the same 1,000 rounds against the card counter. Fitness is the share of rounds won, with the points margin as a tie-breaker. Using the same deals means members differ only where their moves differ, not through luck of the deal.
3. The two fittest members carry over unchanged (elites). The rest are children:
   - Parents are picked by tournament: the best of three random members.
   - About half the children are a neuron-by-neuron crossover of two parents.
   - Every child gets Gaussian noise on its weights, scaled to each layer's spread. The noise starts at 0.2, which changes about 3% of moves, and shrinks by 4% each generation.
4. Each generation's champion plays 400 rounds and 200 matches against the counter. The gate, the replay and the final check then work as for self-play, so a champion that only got lucky does not ship.

The dashboard adds a fitness chart (best, median and worst member per generation) and a population grid with one square per network, marking elites, crossovers, mutants and the champion. *Population vs counter* seats a different member at each table.

```bash
python main.py --population 32 --generations 30 --fitness-rounds 1000
```

### How a self-play pass trains (`--recipe selfplay`)

1. About 800 games are played, spread across CPU cores. Two in five start from random match scores, so the net sees positions near the 100 limit, as the copied rounds do.
2. Decisions from won rounds are kept, weighted by the winning margin: 0.5 for a one-point win, 2.0 for 30 points or more.
3. Wins from all passes go into a store of up to 60,000 decisions. Each pass trains on that store plus copied decisions, sliding from two copied per self-play decision down to one per two.
4. The new generation plays 400 rounds and 200 matches against the counter. The gate keeps it if it wins more rounds than the best so far, with matches as the tie-breaker.

### How games are scored

Every test plays deals in mirrored pairs: the same shuffled deck twice, with the seats swapped, so each player holds both hands and card luck cancels. Matches are paired the same way. This halves the noise in one network's score against the counter (about 4.6 points over 200 rounds becomes about 2.4).

The gate ranks by rounds won rather than matches won, because rounds are far more plentiful: 400 mirrored rounds measure a net to about ±2 points, while 200 matches only measure it to about ±3.5. One test per generation still flatters whichever generation got lucky, so when training ends the top `--finalists` generations (default 4) replay the same `--gate-rounds` rounds (default 2,000) and `--gate-matches` matches (default 400). The best of them goes to the final check.

The card counter's decisions are cached in `runs/.cache/` because they are the same every run. `python main.py --legacy` runs the first recipe for comparison.

### A learned decider with expert nets (`--experts`)

`--experts 3` turns the net into a small decider and three expert nets after Gen 0. The decider reads the table and gives each expert a weight that adds up to 1, and a move's score is the experts' scores mixed by those weights. Every expert starts as a copy of Gen 0, and the decider is first taught a simple stage rule (opening: under 4 throws; endgame: a hand of 12 points or less, an opponent with 2 cards or fewer, or a deck of 10 or fewer; middle: the rest), so the mixture starts by playing exactly like Gen 0. The search passes then train all of them together: the experts can specialise, and the decider learns its own boundaries. It trains with NumPy on the CPU. Gen 0 stays a plain net, so the final check compares the mixture with the net it came from.

```bash
python main.py --start-from public/model/yaniv-net.json --experts 3 --effort 0.7 --generations 12 --search-rounds 300
```

### Playing for matches, and overnight runs

The search values a move by how much it changes the chance of winning the match, not by the round's point margin (`--search-value points` brings that back). At the start of a run, 4,000 rounds of Gen 0 against the counter give a table of the chance of winning from every pair of scores, so a ten-point round counts for a lot at 92 and little at 20, and landing on 50 or 100 counts as the cut it is. Every generation is tested on the same 1,000 rounds and 400 matches, and the gate ranks by matches won, then point margin.

`--hours 7.5` stops starting new passes after 7.5 hours; the gate replay and the final check still run, so an overnight run ends with a result:

```bash
python main.py --start-from public/model/yaniv-net.json --effort 0.7 --generations 200 --search-rounds 300 --hours 7.5
```

### A value net, AlphaZero-style (`--value-net`)

The search teacher used to play every imagined round to its end with the net, 48 times per candidate move. That is slow, and noisy: every future draw is luck, so two close moves rarely separate and the teacher mostly agrees with the net it teaches (93–95% in the long runs).

With `--value-net`, each pass:

1. The teacher net plays itself (`--value-rounds`, 1,500 by default). Every position is stored with how its round ended for each player, in the same units the search uses (the change in the chance of winning the match).
2. A second net, the value net (`training/yaniv/value.py`, 193 inputs: what each player sees, and whose turn it is), learns to predict that from the position. It sees the whole imagined deal except the stock order, so it learns the average over the future draws. It keeps learning across passes, from the last 400,000 positions.
3. The search looks `--value-depth` moves ahead (4 by default) on each imagined deal and asks the value net, instead of playing to the end. A round that ends within those moves is scored exactly.

The depth matters. On the real game's net, over the same 600 rounds against the counter (`teacher_test.py`), the search teacher compared with the plain net:

| Teacher | Points a round | Match chance after the round |
| --- | --- | --- |
| Playouts to the end, 96 deals | +0.01 ± 0.47 | +0.58 ± 0.38 |
| Value net after 2 moves, 96 deals | −1.56 ± 0.78 | −0.13 ± 0.59 |
| Value net after 4 moves, 192 deals | +0.28 ± 0.54 | +0.79 ± 0.40 |

After only 2 moves the value net's errors repeat on every imagined deal, so averaging more deals does not remove them, and the teacher played worse than the net it taught. Use `--search-samples 192` with the default depth.

The log and the summary show how much of how rounds ended the value net explains on each pass's new rounds, before it learns them. Rounds hinge on future draws, so even a perfect judge is well under 100%. The value net is only a teacher: the game still plays with the policy net.

### A smaller game that learns faster (`--hand`, `--yaniv-at`)

`--hand 4 --yaniv-at 9` plays 4 cards each with Yaniv at 9 or less. Rounds are about a third shorter (9 throws instead of 12 between two card counters), and each decision has fewer moves to choose from, so every pass is quicker. Its net ships to `public/model/yaniv-net-4cards-yaniv9.json` (with `--export`), never over the standard one, and the game plays it at `/?game=small` (pick the game in the header):

```bash
python main.py --hand 4 --yaniv-at 9 --value-net --effort 0.7 --generations 40 --search-samples 192 --export
```

### Handing the end of a round to a 4-card net (`training/yaniv/hybrid.py`)

A hand never grows: every throw is followed by exactly one draw. So once both players hold 4 cards or fewer, the rest of the round is a 4-card game, and with `--yaniv-at 7` its rules are the real game's. With the current net against the counter, 83% of rounds reach that point and 48% of all decisions come after it.

Train a 4-card net with the real Yaniv limit, then test the switch player (the main net until both hands are small, then the 4-card net) against the main net alone, on the same deals:

```bash
python main.py --hand 4 --yaniv-at 7 --value-net --value-rounds 4000 --effort 0.7 --generations 40 --search-samples 192 --export
python -m training.yaniv.hybrid --endgame public/model/yaniv-net-4cards-yaniv7.json
```

It prints both players' results against the counter (4,000 rounds, 1,000 matches) and the switch player against the main net alone. The game plays the 4-card net itself at `/?game=4x7`.

### Is the teacher stronger than the net? (`training/yaniv/teacher_test.py`)

A net only learns what its teacher does better than it. Once training stalls, test that directly: the net and several search teachers built on it play the card counter on the same deals and match scores, and each teacher is compared with the plain net round by round:

```bash
python -m training.yaniv.teacher_test --net public/model/yaniv-net.json --effort 0.8
```

The teachers are playouts to the end with 96 deals, the value net after 2 moves with 96 deals, and the value net after 4 moves with 192 deals (`--teachers` picks others, `--value runs/<timestamp>/value-NN.json` uses a run's value net). It reports the point margin and the match chance after the round, each with a ± from the paired rounds. A teacher worth a long run beats the net by more than twice its ± on match chance.

### Records of each run

Each run writes `runs/<timestamp>/log.txt` (every trainer line, with times) and a short summary to `runs/<timestamp>/summary.md` and `results/<timestamp>.md`. The `results/` folder is tracked by git, so committing and pushing it shares how a run went without its nets.

### What the numbers say so far

Under the old rules, neither evolution nor self-play produced a net clearly stronger than the card counter: the best won about 52% of rounds and 49% of matches against it.

Under the current rules, the search player itself beat the counter in 124 of 200 matches (62%). A short search run of 2 passes, about 2,200 searched decisions, lifted the net from 196–204 to 214–186 in matches against the counter on the same 400 new games. The gate replay agreed: 54% against Gen 0's 45.5%. Longer runs should go further; the final check still decides what ships.

The net in the game now is a suit-free copy of Gen 2 from a 12-pass search run. Copying a net onto the rank-only inputs made it play better than the net it copied: on the same 2,000 rounds and 600 matches against the counter, Gen 2 won 331–269 and took 0.60 points a round fewer than the counter, and its copy won 362–238 and took 1.26 fewer.

```bash
python main.py --quick                  # small run to check everything works
python main.py --generations 12         # more self-play passes
python main.py --workers 6              # CPU processes for playing games
python main.py --export                 # write the best net to public/model/ when training ends
```

### Training on a GPU

Install a CUDA build of PyTorch, then run with `--device cuda`. The run stops with an explanation if PyTorch cannot see the GPU, rather than quietly falling back to the CPU.

```bash
pip install -r training/requirements-gpu.txt   # Linux. On Windows use the command from pytorch.org
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python main.py --device cuda
```

`--device auto`, the default, uses CUDA, then Apple MPS, then NumPy. The dashboard header shows which one is in use. The GPU runs the gradient steps. Self-play and evaluation stay on the CPU, because they score about twenty moves at a time and the game rules are plain Python.

If the GPU check fails on Windows:

- `OSError: [WinError 1114] ... c10.dll` means Windows could not load PyTorch. Install the latest [Visual C++ Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe), open a new terminal, and try again. If that does not help, uninstall torch, run `python -m pip cache purge`, and reinstall it.
- RTX 50-series cards need a PyTorch build for CUDA 12.8 or newer, for example the `cu128` build. Older builds install, then fail with "no kernel image is available". The build's CUDA version must also be no higher than the one `nvidia-smi` reports for your driver.
