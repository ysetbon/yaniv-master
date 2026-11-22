/**
 * Feature layout shared with training/yaniv/features.py.
 * Vectors are plain numbers; the network casts them to float32 on the way in.
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
  popcount,
  rankOf,
} from "./engine";

// state: 4 card masks, 2 rank counts, suit counts, 19 scalars
export const STATE_DIM = 4 * DECK_SIZE + 2 * RANKS + 4 + 19;
// action: thrown mask, face-up card, 14 scalars
export const ACTION_DIM = 2 * DECK_SIZE + 14;
const YANIV_SLOT = 2 * DECK_SIZE + 2;
export const INPUT_DIM = STATE_DIM + ACTION_DIM;

function discardMask(view: View): bigint {
  let mask = 0n;
  for (const card of view.discard) mask |= 1n << BigInt(card);
  return mask;
}

export function poolMask(view: View): bigint {
  const buried = discardMask(view);
  const all = (1n << BigInt(DECK_SIZE)) - 1n;
  return all & ~view.myHand & ~buried & ~view.oppKnown;
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

function rankCounts(mask: bigint): number[] {
  const counts = Array.from({ length: RANKS }, () => 0);
  for (const card of cardsFromMask(mask)) counts[rankOf(card)] += 1;
  return counts.map((count) => count / 4);
}

function suitCounts(mask: bigint): number[] {
  const counts = Array.from({ length: 4 }, () => 0);
  for (const card of cardsFromMask(mask)) if (card < 52) counts[Math.floor(card / 13)] += 1;
  return counts.map((count) => count / 5);
}

function bits(mask: bigint): number[] {
  const out = Array.from({ length: DECK_SIZE }, () => 0);
  for (const card of cardsFromMask(mask)) out[card] = 1;
  return out;
}

export function stateFeatures(view: View): number[] {
  const buried = discardMask(view);
  const pool = poolMask(view);
  const poolCards = cardsFromMask(pool);
  const poolN = poolCards.length;
  let poolAvg = 0;
  let poolLow = 0;
  if (poolN) {
    const poolPts = poolCards.map(cardPoints);
    poolAvg = poolPts.reduce((sum, points) => sum + points, 0) / poolN;
    poolLow = poolPts.filter((points) => points <= 3).length / poolN;
  }
  const myPts = maskPoints(view.myHand);
  const mySize = popcount(view.myHand);
  const keep = bestKeepPoints(view.myHand);
  const shed = myPts - keep;
  const knownCards = cardsFromMask(view.oppKnown);
  const knownPts = knownCards.reduce((sum, card) => sum + cardPoints(card), 0);
  const minimum = oppMinPoints(view);
  const openCard = view.discard[view.discard.length - 1];
  const yanivLegal = mySize > 0 && myPts <= YANIV_LIMIT ? 1 : 0;
  const guaranteed = yanivLegal && myPts < minimum ? 1 : 0;
  // Losing the round with this hand would land exactly on 50 or 100 and cut the score.
  const cutIfLost = view.myScore + myPts in SCORE_CUTS ? 1 : 0;
  const vec = bits(view.myHand)
    .concat(bits(buried))
    .concat(bits(1n << BigInt(openCard)))
    .concat(bits(view.oppKnown))
    .concat(rankCounts(view.myHand))
    .concat(rankCounts(pool))
    .concat(suitCounts(view.myHand))
    .concat([
      myPts / 40,
      mySize / 5,
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
      shed / 40,
      knownPts / 40,
      knownCards.length / 5,
      minimum / 40,
      guaranteed,
      cardPoints(openCard) / 10,
      cutIfLost,
    ]);
  if (vec.length !== STATE_DIM) throw new Error(`state dim ${vec.length}`);
  return vec;
}

function keptPair(mask: bigint): boolean {
  const seen = Array.from({ length: RANKS }, () => 0);
  for (const card of cardsFromMask(mask)) {
    seen[rankOf(card)] += 1;
    if (seen[rankOf(card)] >= 2) return true;
  }
  return false;
}

export function actionFeatures(view: View, action: Action): number[] {
  if (action.kind === "yaniv") {
    const vec = bits(0n).concat(bits(0n)).concat(Array.from({ length: ACTION_DIM - 2 * DECK_SIZE }, () => 0));
    vec[YANIV_SLOT] = 1;
    if (vec.length !== ACTION_DIM) throw new Error(`action dim ${vec.length}`);
    return vec;
  }
  const chosen = cardsFromMask(action.discardMask);
  if (!isValidCombo(chosen)) throw new Error("illegal throw");
  const remaining = view.myHand & ~action.discardMask;
  const remainPts = maskPoints(remaining);
  const remainN = popcount(remaining);
  const discardPts = chosen.reduce((sum, card) => sum + cardPoints(card), 0);
  const isSingle = chosen.length === 1 ? 1 : 0;
  const isSet = 1 - isSingle;
  const drawStock = action.draw === "stock" ? 1 : 0;
  const drawOpen = action.draw === "open" ? 1 : 0;
  const openCard = view.discard[view.discard.length - 1];
  let pairs = 0;
  let incoming = 0;
  let resultCallable = 0;
  if (drawOpen) {
    incoming = cardPoints(openCard);
    if (cardsFromMask(remaining).some((card) => rankOf(card) === rankOf(openCard))) pairs = 1;
    if (remainPts + cardPoints(openCard) <= YANIV_LIMIT) resultCallable = 1;
  } else if (drawStock) {
    const pool = cardsFromMask(poolMask(view));
    if (pool.length) {
      incoming = pool.reduce((sum, card) => sum + cardPoints(card), 0) / pool.length;
      const worst = Math.max(...pool.map(cardPoints));
      if (remainPts + worst <= YANIV_LIMIT) resultCallable = 1;
    }
  }
  const tasty = cardPoints(action.top) <= 4 ? 1 : 0;
  const kept = keptPair(remaining) ? 1 : 0;
  const vec = bits(action.discardMask)
    .concat(bits(1n << BigInt(action.top)))
    .concat([
      drawStock,
      drawOpen,
      0, // YANIV_SLOT: 1 only for a Yaniv call
      discardPts / 40,
      remainPts / 40,
      remainN / 5,
      cardPoints(action.top) / 10,
      incoming / 10,
      pairs,
      isSet,
      isSingle,
      kept,
      resultCallable,
      tasty,
    ]);
  if (vec.length !== ACTION_DIM) throw new Error(`action dim ${vec.length}`);
  return vec;
}
