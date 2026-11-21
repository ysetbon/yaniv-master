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
