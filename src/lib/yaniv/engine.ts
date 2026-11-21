/**
 * Two-player Yaniv. Masks are bigint because JavaScript bitwise operators
 * are 32-bit and a deck needs 52 bits. This file matches training/yaniv/engine.py.
 */

export const YANIV_LIMIT = 5;
export const ASSAF_PENALTY = 30;
export const MATCH_LIMIT = 100;
export const MAX_PLIES = 48;

export const RANK_CHARS = "A23456789TJQK";
export const SUIT_CHARS = "♣♦♥♠";
export const SUIT_NAMES = ["clubs", "diamonds", "hearts", "spades"] as const;

export const POINTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10];

export type Draw = "stock" | "open" | "none";
export type RoundReason = "yaniv" | "go_out" | "showdown";

export interface Action {
  kind: "play" | "yaniv";
  discardMask: bigint;
  top: number;
  draw: Draw;
}

export interface RoundResult {
  added: [number, number];
  points: [number, number];
  hands: [bigint, bigint];
  assaf: boolean;
  caller: number | null;
  reason: RoundReason;
  winner: number | null;
}

export interface State {
  hands: [bigint, bigint];
  known: [bigint, bigint];
  stock: number[];
  discard: number[];
  scores: [number, number];
  turn: number;
  ply: number;
  terminal: boolean;
  result: RoundResult | null;
}

export interface View {
  myHand: bigint;
  myKnown: bigint;
  oppKnown: bigint;
  oppSize: number;
  stockSize: number;
  discard: number[];
  myScore: number;
  oppScore: number;
  ply: number;
  seat: number;
}

export interface Rng {
  random(): number;
}

export function cardPoints(card: number): number {
  return POINTS[card % 13];
}

export function cardName(card: number): string {
  const rank = card % 13;
  const label = rank === 9 ? "10" : RANK_CHARS[rank];
  return label + SUIT_CHARS[Math.floor(card / 13)];
}

export function cardLabel(card: number): string {
  const rank = ["ace", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "jack", "queen", "king"][card % 13];
  return `${rank} of ${SUIT_NAMES[Math.floor(card / 13)]}`;
}

export function popcount(mask: bigint): number {
  let n = 0;
  let bits = mask;
  while (bits) {
    bits &= bits - 1n;
    n += 1;
  }
  return n;
}

export function cardsFromMask(mask: bigint): number[] {
  const out: number[] = [];
  let bits = mask;
  while (bits) {
    const lsb = bits & -bits;
    out.push(bitIndex(lsb));
    bits ^= lsb;
  }
  return out;
}

function bitIndex(lsb: bigint): number {
  let index = 0;
  let value = lsb;
  while (value > 1n) {
    value >>= 1n;
    index += 1;
  }
  return index;
}

export function maskOf(cards: number[]): bigint {
  let mask = 0n;
  for (const card of cards) mask |= 1n << BigInt(card);
  return mask;
}

export function maskPoints(mask: bigint): number {
  let total = 0;
  for (const card of cardsFromMask(mask)) total += cardPoints(card);
  return total;
}

export function isValidCombo(cards: number[]): boolean {
  const n = cards.length;
  if (n === 0) return false;
  if (n === 1) return true;
  const rank0 = cards[0] % 13;
  if (cards.every((card) => card % 13 === rank0)) return true;
  if (n < 3) return false;
  const suit0 = Math.floor(cards[0] / 13);
  if (cards.some((card) => Math.floor(card / 13) !== suit0)) return false;
  const ranks = cards.map((card) => card % 13).sort((a, b) => a - b);
  for (let i = 1; i < n; i += 1) {
    if (ranks[i] !== ranks[i - 1] + 1) return false;
  }
  return true;
}

export function bestKeepPoints(mask: bigint): number {
  if (mask === 0n) return 0;
  const cards = cardsFromMask(mask);
  if (cards.length === 1 || isValidCombo(cards)) return 0;
  let total = 0;
  let maxSingle = 0;
  const byRankPoints = new Map<number, number>();
  const byRankCount = new Map<number, number>();
  const bySuit = new Map<number, number[]>();
  for (const card of cards) {
    const points = cardPoints(card);
    total += points;
    if (points > maxSingle) maxSingle = points;
    const rank = card % 13;
    byRankPoints.set(rank, (byRankPoints.get(rank) ?? 0) + points);
    byRankCount.set(rank, (byRankCount.get(rank) ?? 0) + 1);
    const suit = Math.floor(card / 13);
    const group = bySuit.get(suit);
    if (group) group.push(card);
    else bySuit.set(suit, [card]);
  }
  let best = total - maxSingle;
  for (const [rank, count] of byRankCount) {
    if (count >= 2) {
      const rem = total - (byRankPoints.get(rank) ?? 0);
      if (rem < best) best = rem;
    }
  }
  for (const group of bySuit.values()) {
    if (group.length < 3) continue;
    group.sort((a, b) => (a % 13) - (b % 13));
    let start = 0;
    while (start < group.length) {
      let end = start;
      while (end + 1 < group.length && group[end + 1] % 13 === (group[end] % 13) + 1) end += 1;
      if (end - start + 1 >= 3) {
        let shed = 0;
        for (let i = start; i <= end; i += 1) shed += cardPoints(group[i]);
        const rem = total - shed;
        if (rem < best) best = rem;
      }
      start = end + 1;
    }
  }
  return best;
}

export function makeView(state: State, seat: number): View {
  const other = 1 - seat;
  return {
    myHand: state.hands[seat],
    myKnown: state.known[seat],
    oppKnown: state.known[other],
    oppSize: popcount(state.hands[other]),
    stockSize: state.stock.length,
    discard: state.discard.slice(),
    myScore: state.scores[seat],
    oppScore: state.scores[other],
    ply: state.ply,
    seat,
  };
}

export function shuffle<T>(items: T[], rng: Rng): void {
  for (let i = items.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng.random() * (i + 1));
    const swap = items[i];
    items[i] = items[j];
    items[j] = swap;
  }
}

export function newRound(scores: [number, number], leader: number, rng: Rng): State {
  const deck = Array.from({ length: 52 }, (_, card) => card);
  shuffle(deck, rng);
  const hands: [bigint, bigint] = [0n, 0n];
  for (let i = 0; i < 10; i += 1) hands[i % 2] |= 1n << BigInt(deck[i]);
  return {
    hands,
    known: [0n, 0n],
    stock: deck.slice(11),
    discard: [deck[10]],
    scores: [scores[0], scores[1]],
    turn: leader,
    ply: 0,
    terminal: false,
    result: null,
  };
}

function finish(
  state: State,
  added: [number, number],
  assaf: boolean,
  caller: number | null,
  reason: RoundReason,
  winner: number | null,
): void {
  state.terminal = true;
  state.result = {
    added: [added[0], added[1]],
    points: [maskPoints(state.hands[0]), maskPoints(state.hands[1])],
    hands: [state.hands[0], state.hands[1]],
    assaf,
    caller,
    reason,
    winner,
  };
  state.scores = [state.scores[0] + added[0], state.scores[1] + added[1]];
}

export function resolveShowdown(state: State): void {
  const p0 = maskPoints(state.hands[0]);
  const p1 = maskPoints(state.hands[1]);
  if (p0 < p1) finish(state, [0, p1], false, null, "showdown", 0);
  else if (p1 < p0) finish(state, [p0, 0], false, null, "showdown", 1);
  else finish(state, [p0, p1], false, null, "showdown", null);
}

export function legalActions(view: View): Action[] {
  if (view.ply >= MAX_PLIES || view.myHand === 0n) return [];
  const cards = cardsFromMask(view.myHand);
  const n = cards.length;
  const actions: Action[] = [];
  const stockOk = view.stockSize > 0;
  for (let subset = 1; subset < 1 << n; subset += 1) {
    const chosen: number[] = [];
    for (let i = 0; i < n; i += 1) if (subset & (1 << i)) chosen.push(cards[i]);
    if (!isValidCombo(chosen)) continue;
    const discardMask = maskOf(chosen);
    if (chosen.length === n) {
      const top = chosen.reduce((best, card) => {
        const bestPoints = cardPoints(best);
        const points = cardPoints(card);
        if (points > bestPoints || (points === bestPoints && card > best)) return card;
        return best;
      });
      actions.push({ kind: "play", discardMask, top, draw: "none" });
      continue;
    }
    for (const top of chosen) {
      if (stockOk) actions.push({ kind: "play", discardMask, top, draw: "stock" });
      actions.push({ kind: "play", discardMask, top, draw: "open" });
    }
  }
  if (maskPoints(view.myHand) <= YANIV_LIMIT) {
    actions.push({ kind: "yaniv", discardMask: 0n, top: -1, draw: "none" });
  }
  actions.sort((a, b) => {
    const kind = (a.kind === "play" ? 0 : 1) - (b.kind === "play" ? 0 : 1);
    if (kind) return kind;
    if (a.discardMask !== b.discardMask) return a.discardMask < b.discardMask ? -1 : 1;
    if (a.top !== b.top) return a.top - b.top;
    const draw = (value: Draw) => (value === "stock" ? 0 : value === "open" ? 1 : 2);
    return draw(a.draw) - draw(b.draw);
  });
  return actions;
}

export function applyAction(state: State, action: Action): void {
  if (state.terminal) throw new Error("round is over");
  if (state.ply >= MAX_PLIES) {
    resolveShowdown(state);
    return;
  }
  const me = state.turn;
  if (action.kind === "yaniv") {
    if (maskPoints(state.hands[me]) > YANIV_LIMIT) throw new Error("Yaniv is only legal at 5 or fewer");
    const opp = 1 - me;
    const mine = maskPoints(state.hands[me]);
    const theirs = maskPoints(state.hands[opp]);
    if (mine < theirs) {
      const added: [number, number] = [0, 0];
      added[opp] = theirs;
      finish(state, added, false, me, "yaniv", me);
    } else {
      const added: [number, number] = [0, 0];
      added[me] = mine + ASSAF_PENALTY;
      finish(state, added, true, me, "yaniv", opp);
    }
    return;
  }
  const hand = state.hands[me];
  const discardMask = action.discardMask;
  if (discardMask === 0n || (discardMask & hand) !== discardMask) throw new Error("discard is not in hand");
  const chosen = cardsFromMask(discardMask);
  if (!isValidCombo(chosen) || !chosen.includes(action.top)) throw new Error("illegal combination");
  const remaining = hand & ~discardMask;
  const ordered = chosen.filter((card) => card !== action.top).concat(action.top);
  state.known[me] &= ~discardMask;
  state.discard.push(...ordered);
  state.ply += 1;
  if (remaining === 0n) {
    state.hands[me] = 0n;
    state.known[me] = 0n;
    const opp = 1 - me;
    const added: [number, number] = [0, 0];
    added[opp] = maskPoints(state.hands[opp]);
    finish(state, added, false, null, "go_out", me);
    return;
  }
  let card: number;
  if (action.draw === "stock") {
    const drawn = state.stock.pop();
    if (drawn === undefined) throw new Error("stock is empty");
    card = drawn;
  } else if (action.draw === "open") {
    const idx = state.discard.length - 1 - ordered.length;
    card = state.discard[idx];
    state.discard.splice(idx, 1);
    state.known[me] |= 1n << BigInt(card);
  } else {
    throw new Error("a throw that leaves cards must draw");
  }
  state.hands[me] = remaining | (1n << BigInt(card));
  state.turn = 1 - me;
}

export function cloneState(state: State): State {
  return {
    hands: [state.hands[0], state.hands[1]],
    known: [state.known[0], state.known[1]],
    stock: state.stock.slice(),
    discard: state.discard.slice(),
    scores: [state.scores[0], state.scores[1]],
    turn: state.turn,
    ply: state.ply,
    terminal: state.terminal,
    result: state.result
      ? {
          added: [state.result.added[0], state.result.added[1]],
          points: [state.result.points[0], state.result.points[1]],
          hands: [state.result.hands[0], state.result.hands[1]],
          assaf: state.result.assaf,
          caller: state.result.caller,
          reason: state.result.reason,
          winner: state.result.winner,
        }
      : null,
  };
}

export function defaultTop(cards: number[]): number {
  return cards.reduce((best, card) => {
    const bestPoints = cardPoints(best);
    const points = cardPoints(card);
    if (points > bestPoints || (points === bestPoints && card > best)) return card;
    return best;
  });
}

export function sortHand(cards: number[]): number[] {
  return cards.slice().sort((a, b) => (a % 13) - (b % 13) || a - b);
}
