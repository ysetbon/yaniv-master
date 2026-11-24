/**
 * Two-player Yaniv. Masks are bigint because JavaScript bitwise operators
 * are 32-bit and a deck needs 54 bits. This file matches training/yaniv/engine.py.
 *
 * Cards 0..51 are suit * 13 + rank (rank 0 = Ace ... 12 = King); 52 and 53
 * are the red and black jokers, worth 0. Throws are a single card or a set
 * of one rank; there are no runs, and a joker pairs only with the other
 * joker. A score landing exactly on 50 or 100 after a round is cut by 50.
 */

export const DECK_SIZE = 54;
export const JOKER_RANK = 13;
export const RANKS = 14;
export const SCORE_CUTS: Readonly<Record<number, number>> = { 50: 0, 100: 50 };
// `let` so a smaller game (setRules) can change them; imports see the change.
export let YANIV_LIMIT = 7;
export let HAND_SIZE = 5;
export const ASSAF_PENALTY = 30;
export const MATCH_LIMIT = 100;
export const MAX_PLIES = 48;

export const RANK_CHARS = "A23456789TJQK";
export const SUIT_CHARS = "♣♦♥♠";
export const SUIT_NAMES = ["clubs", "diamonds", "hearts", "spades"] as const;

export const POINTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10];

/**
 * The games the page offers, picked with /?game=<id>. The smaller ones are
 * trained with main.py --hand H --yaniv-at Y, and each net lives in its own file.
 */
export const GAMES = [
  { id: "standard", hand: 5, yanivAt: 7, label: "Standard" },
  { id: "4x7", hand: 4, yanivAt: 7, label: "4 cards, Yaniv 7" },
  { id: "small", hand: 4, yanivAt: 9, label: "4 cards, Yaniv 9" },
] as const;
export type GameRules = (typeof GAMES)[number];

/** The game a /?game= value names; the standard game for anything else. */
export function gameById(id: string | null): GameRules {
  return GAMES.find((game) => game.id === id) ?? GAMES[0];
}

/** Cards each and the Yaniv limit, as main.py's --hand and --yaniv-at. Call before dealing. */
export function setRules(hand: number, yanivAt: number): void {
  HAND_SIZE = hand;
  YANIV_LIMIT = yanivAt;
}

/** The model file suffix the trainer uses for these rules: "" for the standard game. */
export function rulesSuffix(hand = HAND_SIZE, yanivAt = YANIV_LIMIT): string {
  return hand === 5 && yanivAt === 7 ? "" : `-${hand}cards-yaniv${yanivAt}`;
}

export type Draw = "stock" | "open" | "none";
export type RoundReason = "yaniv" | "showdown";

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

export function isJoker(card: number): boolean {
  return card >= 52;
}

export function rankOf(card: number): number {
  return card >= 52 ? JOKER_RANK : card % 13;
}

export function cardPoints(card: number): number {
  return card >= 52 ? 0 : POINTS[card % 13];
}

export function cardName(card: number): string {
  if (card >= 52) return "★";
  const rank = card % 13;
  const label = rank === 9 ? "10" : RANK_CHARS[rank];
  return label + SUIT_CHARS[Math.floor(card / 13)];
}

export function cardLabel(card: number): string {
  if (card >= 52) return card === 52 ? "red joker" : "black joker";
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
  if (cards.length === 0) return false;
  const rank0 = rankOf(cards[0]);
  return cards.every((card) => rankOf(card) === rank0);
}

export function bestKeepPoints(mask: bigint): number {
  if (mask === 0n) return 0;
  const cards = cardsFromMask(mask);
  if (isValidCombo(cards)) return 0;
  let total = 0;
  const byRank = new Map<number, number>();
  for (const card of cards) {
    const points = cardPoints(card);
    total += points;
    byRank.set(rankOf(card), (byRank.get(rankOf(card)) ?? 0) + points);
  }
  return total - Math.max(...byRank.values());
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
  const deck = Array.from({ length: DECK_SIZE }, (_, card) => card);
  shuffle(deck, rng);
  const hands: [bigint, bigint] = [0n, 0n];
  const dealt = 2 * HAND_SIZE;
  for (let i = 0; i < dealt; i += 1) hands[i % 2] |= 1n << BigInt(deck[i]);
  return {
    hands,
    known: [0n, 0n],
    stock: deck.slice(dealt + 1),
    discard: [deck[dealt]],
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
  const cut = (total: number) => SCORE_CUTS[total] ?? total;
  state.scores = [cut(state.scores[0] + added[0]), cut(state.scores[1] + added[1])];
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
    if (maskPoints(state.hands[me]) > YANIV_LIMIT) throw new Error(`Yaniv is only legal at ${YANIV_LIMIT} or fewer`);
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
  if (action.draw !== "stock" && action.draw !== "open") throw new Error("every throw must draw from the deck or the open card");
  if (action.draw === "stock" && state.stock.length === 0) throw new Error("stock is empty");
  const remaining = hand & ~discardMask;
  const ordered = chosen.filter((card) => card !== action.top).concat(action.top);
  state.known[me] &= ~discardMask;
  state.discard.push(...ordered);
  state.ply += 1;
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
    throw new Error("every throw must draw from the deck or the open card");
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
  return cards.slice().sort((a, b) => rankOf(a) - rankOf(b) || a - b);
}
