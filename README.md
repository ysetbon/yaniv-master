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

`main.py` trains the network and opens a dashboard at http://127.0.0.1:8765. The live table plays the newest generation against the one before it, face up, with the scores the net gave its top moves. You can seat any generation, the card counter, the greedy player, or random play in either chair. Charts update as training runs: agreement with the card counter while it copies, win rate against the counter per generation, points per round, and a ladder where each new generation plays every older one.

A generation here is one checkpoint: generation 0 is the copy of the card counter, and each later one comes from a self-play pass. This is imitation learning plus self-play, not a genetic algorithm. There is no population, mutation, or crossover.

Every generation is saved under `runs/<timestamp>/`. The model the game loads is only replaced when you pass `--export`.

```bash
python main.py --quick                  # small run to check everything works
python main.py --generations 10         # more self-play passes
python main.py --export                 # also write the best net to public/model/
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
