import type { Net } from "./net";
import { scoreActions } from "./net";
import {
  type Action,
  type View,
  cardName,
  cardPoints,
  cardsFromMask,
  legalActions,
  maskPoints,
} from "../yaniv/engine";
import { actionFeatures, stateFeatures } from "../yaniv/features";

export interface ScoredAction {
  action: Action;
  q: number;
}

export function chooseAction(net: Net, view: View): ScoredAction[] {
  const actions = legalActions(view);
  if (!actions.length) return [];
  const state = stateFeatures(view);
  const rows = actions.map((action) => actionFeatures(view, action));
  const scores = scoreActions(net, state, rows);
  const ranked = actions.map((action, index) => ({ action, q: scores[index] }));
  ranked.sort((a, b) => b.q - a.q);
  return ranked;
}

export function describeAction(view: View, action: Action, ranked: ScoredAction[]): string {
  const mine = maskPoints(view.myHand);
  const call = ranked.find((item) => item.action.kind === "yaniv");
  const chosen = ranked.find(
    (item) =>
      item.action.kind === action.kind &&
      item.action.discardMask === action.discardMask &&
      item.action.top === action.top &&
      item.action.draw === action.draw,
  );
  if (action.kind === "yaniv") {
    const margin = chosen && ranked[1] ? "It likes calling more than any throw." : "It calls.";
    return `Calls Yaniv on ${mine}. ${margin}`;
  }
  const thrown = cardsFromMask(action.discardMask);
  const names = thrown.map(cardName).join(" ");
  const shed = thrown.reduce((sum, card) => sum + cardPoints(card), 0);
  if (action.draw === "none") return `Throws ${names} and goes out for zero.`;
  const open = view.discard[view.discard.length - 1];
  const draw = action.draw === "open" ? `takes the ${cardName(open)}` : "draws from the deck";
  let line = `Throws ${names} (${shed} points) and ${draw}.`;
  if (call && chosen && chosen.q >= call.q) line += " Calling looked worse than playing on.";
  return line;
}
