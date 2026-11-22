# Yaniv

A two-player game of Yaniv. You sit across from a neural network that learned the table by copying a card counter and then playing thousands of games against itself.

A hand is five cards. Ace is 1, number cards are their pips, and 10, jack, queen, and king are 10. Throw a single card, a set of one rank, or a same-suit run of three or more, then draw the previous open card or a blind card from the deck. With 5 points or fewer you may call Yaniv. If the other hand is the same or lower, the call is Assaf and costs your points plus 30. The match ends when a score reaches 100. The lower score wins.

## Run it

```bash
npm install
npm run dev
```

The trained weights already live in `public/model`, so you can play without training again. Open the app and deal a match. Select cards, then choose whether the next card comes from the deck or from the open pile. "Ask the net" shows the throw the network would make in your seat.

## The network

Training lives in `training/yaniv`. It plays a full Yaniv rules engine, scores every legal throw with a small multilayer perceptron, and writes `public/model/yaniv-net.json`.

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

Generation 0 is always the copy of the card counter. After that, `main.py` runs a genetic algorithm by default. `--recipe selfplay` uses the earlier self-play recipe instead.

### How the genetic algorithm works

1. The first population is the copy plus 23 mutants of it. Every member keeps the same shape as the copy (377 → 192 → 192 → 1), so neuron *i* means roughly the same thing in every member.
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

### What the numbers say so far

On a test of 1,000 rounds and 400 matches on new games, no recipe has yet produced a generation clearly stronger than the plain copy of the counter. Every candidate won 44–46% of rounds and 38–42% of matches, and those differences are within chance. Learning only from its own wins keeps the net close to the player it copied. The final check exists so a generation that scored well by luck during training does not ship.

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
