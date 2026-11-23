/**
 * Suit-free feature layout shared with training/yaniv/features.py; the layout
 * is documented there. Every card is described by its rank (13 plus the
 * jokers), and moves that differ only by suit are one move for the net
 * (`netActions`). Vectors are plain numbers; the network casts them to
 * float32 on the way in.
 */

import {
  DECK_SIZE,
  MAX_PLIES,
  RANKS,
  SCORE_CUTS,
  YANIV_LIMIT,
  type Action,
  type View,
  bestKeepPoints,
  cardPoints,
  cardsFromMask,
  isValidCombo,
  maskPoints,
  rankOf,
} from "./engine";

export const STATE_DIM = 5 * RANKS + 19 + 5 + 2;
export const ACTION_DIM = RANKS + 1 + 14;
export const INPUT_DIM = STATE_DIM + ACTION_DIM;
const YANIV_SLOT = RANKS + 1 + 2;

/** Low-card lean of the opponent's hidden cards, by throws so far (ply / 2). Same as HIDDEN_BETA in Python. */
const HIDDEN_BETA = [0.04, 0.09, 0.14, 0.18, 0.21, 0.24, 0.26];

function discardMask(view: View): bigint {
  let mask = 0n;
  for (const card of view.discard) mask |= 1n << BigInt(card);
  return mask;
}

export function poolMask(view: View): bigint {
  const all = (1n << BigInt(DECK_SIZE)) - 1n;
  return all & ~view.myHand & ~discardMask(view) & ~view.oppKnown;
}

export function oppMinPoints(view: View): number {
  const known = cardsFromMask(view.oppKnown);
  const knownPts = known.reduce((sum, card) => sum + cardPoints(card), 0);
  const need = view.oppSize - known.length;
  if (need <= 0) return knownPts;
  const poolPts = cardsFromMask(poolMask(view))
    .map(cardPoints)
    .sort((a, b) => a - b);
  if (need > poolPts.length) return knownPts;
  return knownPts + poolPts.slice(0, need).reduce((sum, points) => sum + points, 0);
}

export function hiddenLean(view: View): number {
  const beta = HIDDEN_BETA[Math.min(Math.floor(view.ply / 2), HIDDEN_BETA.length - 1)];
  const pool = cardsFromMask(poolMask(view));
  if (!pool.length) return 0;
  let weighted = 0;
  let total = 0;
  for (const card of pool) {
    const weight = Math.exp(-beta * cardPoints(card));
    weighted += weight * cardPoints(card);
    total += weight;
  }
  return weighted / total;
}

function choose(n: number, k: number): number {
  let out = 1;
  for (let i = 1; i <= k; i += 1) out = (out * (n - k + i)) / i;
  return Math.round(out);
}

/** Share of the opponent's possible hidden hands that would catch a Yaniv call now (exact). */
export function assafChance(view: View): number {
  const mine = maskPoints(view.myHand);
  const known = cardsFromMask(view.oppKnown);
  const room = mine - known.reduce((sum, card) => sum + cardPoints(card), 0);
  const need = view.oppSize - known.length;
  const pool = cardsFromMask(poolMask(view)).map(cardPoints);
  if (need <= 0) return room >= 0 ? 1 : 0;
  if (need > pool.length) return 1;
  if (room < 0) return 0;
  const ways = Array.from({ length: need + 1 }, () => Array.from({ length: room + 1 }, () => 0));
  ways[0][0] = 1;
  for (const points of pool) {
    for (let k = need; k > 0; k -= 1) {
      for (let s = room; s >= points; s -= 1) ways[k][s] += ways[k - 1][s - points];
    }
  }
  return ways[need].reduce((sum, n) => sum + n, 0) / choose(pool.length, need);
}

function addRanks(cards: number[], vec: number[], at: number): void {
  for (const card of cards) vec[at + rankOf(card)] += 0.25;
}

export function stateFeatures(view: View): number[] {
  const vec = Array.from({ length: STATE_DIM }, () => 0);
  const pool = cardsFromMask(poolMask(view));
  const poolN = pool.length;
  const poolPts = pool.map(cardPoints);
  const poolAvg = poolN ? poolPts.reduce((sum, p) => sum + p, 0) / poolN : 0;
  const poolLow = poolN ? poolPts.filter((p) => p <= 3).length / poolN : 0;
  const hand = cardsFromMask(view.myHand);
  const myPts = maskPoints(view.myHand);
  const keep = bestKeepPoints(view.myHand);
  const known = cardsFromMask(view.oppKnown);
  const knownPts = known.reduce((sum, card) => sum + cardPoints(card), 0);
  const need = view.oppSize - known.length;
  const sorted = poolPts.slice().sort((a, b) => a - b);
  const minimum = need > 0 && need <= poolN ? knownPts + sorted.slice(0, need).reduce((sum, p) => sum + p, 0) : knownPts;
  const openCard = view.discard[view.discard.length - 1];
  const yanivLegal = hand.length > 0 && myPts <= YANIV_LIMIT ? 1 : 0;

  addRanks(hand, vec, 0);
  addRanks(view.discard.slice(0, -1), vec, RANKS);
  vec[2 * RANKS + rankOf(openCard)] = 1;
  addRanks(known, vec, 3 * RANKS);
  addRanks(pool, vec, 4 * RANKS);
  const at = 5 * RANKS;
  const scalars = [
    myPts / 40,
    hand.length / 5,
    view.oppSize / 5,
    view.stockSize / 40,
    view.myScore / 120,
    view.oppScore / 120,
    poolN / 40,
    poolAvg / 10,
    poolLow,
    yanivLegal,
    view.ply / MAX_PLIES,
    keep / 40,
    (myPts - keep) / 40,
    knownPts / 40,
    known.length / 5,
    minimum / 40,
    yanivLegal && myPts < minimum ? 1 : 0,
    cardPoints(openCard) / 10,
    view.myScore + myPts in SCORE_CUTS ? 1 : 0,
  ];
  for (let i = 0; i < scalars.length; i += 1) vec[at + i] = scalars[i];
  vec[at + 19 + Math.min(5, Math.max(1, view.oppSize)) - 1] = 1;
  vec[at + 24] = (knownPts + Math.max(0, need) * hiddenLean(view)) / 40;
  vec[at + 25] = yanivLegal ? assafChance(view) : 0;
  if (vec.length !== STATE_DIM) throw new Error(`state dim ${vec.length}`);
  return vec;
}

/** What the net can tell apart about a move: the rank and number thrown, and the draw. */
export function actionKey(action: Action): string {
  if (action.kind === "yaniv") return "yaniv";
  const cards = cardsFromMask(action.discardMask);
  return `${rankOf(cards[0])}|${cards.length}|${action.draw}`;
}

/** One move per actionKey, in first-seen order. */
export function netActions(actions: Action[]): Action[] {
  const seen = new Set<string>();
  const out: Action[] = [];
  for (const action of actions) {
    const key = actionKey(action);
    if (!seen.has(key)) {
      seen.add(key);
      out.push(action);
    }
  }
  return out;
}

export function actionFeatures(view: View, action: Action): number[] {
  const vec = Array.from({ length: ACTION_DIM }, () => 0);
  if (action.kind === "yaniv") {
    vec[YANIV_SLOT] = 1;
    return vec;
  }
  const chosen = cardsFromMask(action.discardMask);
  if (!isValidCombo(chosen)) throw new Error("illegal throw");
  const remaining = view.myHand & ~action.discardMask;
  const left = cardsFromMask(remaining);
  const remainPts = maskPoints(remaining);
  const ranks = left.map(rankOf);
  const openCard = view.discard[view.discard.length - 1];
  const discardPts = chosen.reduce((sum, card) => sum + cardPoints(card), 0);
  const drawStock = action.draw === "stock" ? 1 : 0;
  const drawOpen = action.draw === "open" ? 1 : 0;
  let pairs = 0;
  let incoming = 0;
  let resultCallable = 0;
  if (drawOpen) {
    incoming = cardPoints(openCard);
    if (ranks.includes(rankOf(openCard))) pairs = 1;
    if (remainPts + cardPoints(openCard) <= YANIV_LIMIT) resultCallable = 1;
  } else if (drawStock) {
    const pool = cardsFromMask(poolMask(view));
    if (pool.length) {
      incoming = pool.reduce((sum, card) => sum + cardPoints(card), 0) / pool.length;
      if (remainPts + Math.max(...pool.map(cardPoints)) <= YANIV_LIMIT) resultCallable = 1;
    }
  }
  const single = chosen.length === 1 ? 1 : 0;
  const topPts = cardPoints(action.top);
  vec[rankOf(chosen[0])] = 1;
  vec[RANKS] = chosen.length / 4;
  const scalars = [
    drawStock,
    drawOpen,
    0,
    discardPts / 40,
    remainPts / 40,
    left.length / 5,
    topPts / 10,
    incoming / 10,
    pairs,
    1 - single,
    single,
    new Set(ranks).size < ranks.length ? 1 : 0,
    resultCallable,
    topPts <= 4 ? 1 : 0,
  ];
  for (let i = 0; i < scalars.length; i += 1) vec[RANKS + 1 + i] = scalars[i];
  return vec;
}
