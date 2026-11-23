import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { bestKeepPoints, applyAction, legalActions, resolveShowdown, type Action, type State, type View } from "./engine";
import { actionFeatures, netActions, stateFeatures } from "./features";

interface RawState {
  hands: [string, string];
  known: [string, string];
  stock: number[];
  discard: number[];
  scores: [number, number];
  turn: number;
  ply: number;
  terminal: boolean;
  result: {
    added: [number, number];
    points: [number, number];
    hands: [string, string];
    assaf: boolean;
    caller: number | null;
    reason: "yaniv" | "showdown";
    winner: number | null;
  } | null;
}

interface RawAction {
  kind: "play" | "yaniv";
  discard_mask: string;
  top: number;
  draw: "stock" | "open" | "none";
}

const fixtures = JSON.parse(readFileSync(new URL("./fixtures.json", import.meta.url), "utf8")) as {
  keeps: Array<{ mask: string; keep: number }>;
  steps: Array<{ before: RawState; action: RawAction | null; after: RawState }>;
  features: Array<{
    view: {
      my_hand: string;
      my_known: string;
      opp_known: string;
      opp_size: number;
      stock_size: number;
      discard: number[];
      my_score: number;
      opp_score: number;
      ply: number;
      seat: number;
    };
    state: number[];
    actions: Array<{ action: RawAction; features: number[] }>;
  }>;
};

function asState(raw: RawState): State {
  return {
    hands: [BigInt(raw.hands[0]), BigInt(raw.hands[1])],
    known: [BigInt(raw.known[0]), BigInt(raw.known[1])],
    stock: raw.stock.slice(),
    discard: raw.discard.slice(),
    scores: [raw.scores[0], raw.scores[1]],
    turn: raw.turn,
    ply: raw.ply,
    terminal: raw.terminal,
    result: raw.result
      ? {
          added: [raw.result.added[0], raw.result.added[1]],
          points: [raw.result.points[0], raw.result.points[1]],
          hands: [BigInt(raw.result.hands[0]), BigInt(raw.result.hands[1])],
          assaf: raw.result.assaf,
          caller: raw.result.caller,
          reason: raw.result.reason,
          winner: raw.result.winner,
        }
      : null,
  };
}

function asAction(raw: RawAction): Action {
  return { kind: raw.kind, discardMask: BigInt(raw.discard_mask), top: raw.top, draw: raw.draw };
}

function asView(raw: (typeof fixtures.features)[number]["view"]): View {
  return {
    myHand: BigInt(raw.my_hand),
    myKnown: BigInt(raw.my_known),
    oppKnown: BigInt(raw.opp_known),
    oppSize: raw.opp_size,
    stockSize: raw.stock_size,
    discard: raw.discard.slice(),
    myScore: raw.my_score,
    oppScore: raw.opp_score,
    ply: raw.ply,
    seat: raw.seat,
  };
}

function sameState(actual: State, expected: RawState): void {
  assert.equal(actual.hands[0], BigInt(expected.hands[0]));
  assert.equal(actual.hands[1], BigInt(expected.hands[1]));
  assert.equal(actual.known[0], BigInt(expected.known[0]));
  assert.equal(actual.known[1], BigInt(expected.known[1]));
  assert.deepEqual(actual.stock, expected.stock);
  assert.deepEqual(actual.discard, expected.discard);
  assert.deepEqual(actual.scores, expected.scores);
  assert.equal(actual.turn, expected.turn);
  assert.equal(actual.ply, expected.ply);
  assert.equal(actual.terminal, expected.terminal);
  if (!expected.result) {
    assert.equal(actual.result, null);
    return;
  }
  assert.ok(actual.result);
  assert.deepEqual(actual.result.added, expected.result.added);
  assert.deepEqual(actual.result.points, expected.result.points);
  assert.equal(actual.result.hands[0], BigInt(expected.result.hands[0]));
  assert.equal(actual.result.hands[1], BigInt(expected.result.hands[1]));
  assert.equal(actual.result.assaf, expected.result.assaf);
  assert.equal(actual.result.caller, expected.result.caller);
  assert.equal(actual.result.reason, expected.result.reason);
  assert.equal(actual.result.winner, expected.result.winner);
}

function close(actual: number[], expected: number[], label: string): void {
  assert.equal(actual.length, expected.length, label);
  for (let i = 0; i < actual.length; i += 1) {
    assert.ok(Math.abs(actual[i] - expected[i]) < 1e-5, `${label}[${i}] ${actual[i]} vs ${expected[i]}`);
  }
}

let keepMiss = 0;
for (const row of fixtures.keeps) {
  if (bestKeepPoints(BigInt(row.mask)) !== row.keep) keepMiss += 1;
}
assert.equal(keepMiss, 0, "best-keep mismatch");

for (const step of fixtures.steps) {
  const state = asState(step.before);
  if (step.action) applyAction(state, asAction(step.action));
  else resolveShowdown(state);
  sameState(state, step.after);
}

for (const feature of fixtures.features) {
  const view = asView(feature.view);
  close(stateFeatures(view), feature.state, "state");
  const legal = netActions(legalActions(view));
  assert.ok(legal.length >= feature.actions.length);
  feature.actions.forEach((row, index) => {
    const action = asAction(row.action);
    assert.equal(legal[index].kind, action.kind);
    assert.equal(legal[index].discardMask, action.discardMask);
    assert.equal(legal[index].top, action.top);
    assert.equal(legal[index].draw, action.draw);
    close(actionFeatures(view, action), row.features, `action ${index}`);
  });
}

console.log(`parity ok: ${fixtures.keeps.length} keeps, ${fixtures.steps.length} steps, ${fixtures.features.length} views`);
