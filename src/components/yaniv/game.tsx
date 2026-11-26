"use client";

import { useEffect, useMemo, useRef, useState, type CSSProperties, type ReactNode } from "react";
import { BookOpen, Eye, EyeOff, RotateCcw, ScrollText } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Separator } from "@/components/ui/separator";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { PlayingCard } from "@/components/yaniv/playing-card";
import { chooseAction, describeAction } from "@/lib/ai/agent";
import { loadNet, type Net } from "@/lib/ai/net";
import {
  ASSAF_PENALTY,
  HAND_SIZE,
  gameById,
  rulesSuffix,
  setRules,
  MATCH_LIMIT,
  YANIV_LIMIT,
  type Action,
  type RoundResult,
  type State,
  applyAction,
  cardName,
  cardPoints,
  cardsFromMask,
  cloneState,
  defaultTop,
  isValidCombo,
  makeView,
  maskOf,
  maskPoints,
  newRound,
  popcount,
  resolveShowdown,
  sortHand,
} from "@/lib/yaniv/engine";

interface Report {
  expert_rounds: number;
  expert_decisions: number;
  self_play_games: number;
  self_play_winning_decisions: number;
  layers: number[];
  seconds: number | null;
  match_win_rate_vs_benchmark: number;
  match_record_vs_benchmark: { wins: number; losses: number; ties: number };
  round_win_rate_vs_benchmark: number;
  round_record_vs_benchmark: {
    wins: number;
    losses: number;
    ties: number;
    hero_avg: number;
    villain_avg: number;
    assaf_rate: number;
  };
  round_win_rate_vs_greedy: number;
  round_record_vs_greedy: { hero_avg: number; villain_avg: number };
  round_win_rate_vs_random: number;
  benchmark_vs_greedy: { wins: number; losses: number; win_rate: number; hero_avg: number; villain_avg: number; assaf_rate: number };
  recipe?: string;
  best_generation?: number;
  self_play_iterations?: number;
  config?: { population?: number; fitness_rounds?: number; search_samples?: number; copy_from?: string };
  experts?: number;
  decider?: number[] | null;
}

interface LogLine {
  id: number;
  text: string;
  who: "you" | "net";
}

interface Match {
  state: State;
  leader: number;
  round: number;
  /** Scores when this round was dealt, so the result can show a 50/100 cut. */
  startScores: [number, number];
}

const rng = { random: () => Math.random() };

/** localStorage key for the Tips switch: "on" or "off". */
const TIPS_KEY = "yaniv-tips";

function mulberry32(seed: number) {
  let state = seed >>> 0;
  return {
    random() {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      return state / 4294967296;
    },
  };
}

/** Load the page again for another game: the rules are set once, when the page boots. */
/** Where the model files are served: /model, under the site's sub-path on GitHub Pages (next.config.ts). */
const MODEL_DIR = `${process.env.NEXT_PUBLIC_BASE_PATH ?? ""}/model`;

function reloadAs(id: string): void {
  const url = new URL(window.location.href);
  if (id === "standard") url.searchParams.delete("game");
  else url.searchParams.set("game", id);
  window.location.assign(url.href);
}

function freshMatch(): Match {
  return { state: newRound([0, 0], 0, rng), leader: 0, round: 1, startScores: [0, 0] };
}

export function Game() {
  const [net, setNet] = useState<Net | null>(null);
  const [report, setReport] = useState<Report | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [booting, setBooting] = useState(true);
  const [attempt, setAttempt] = useState(0);
  const [match, setMatch] = useState<Match>(() => ({
    state: newRound([0, 0], 0, mulberry32(1)),
    leader: 0,
    round: 1,
    startScores: [0, 0],
  }));
  const [selected, setSelected] = useState<number[]>([]);
  const [topOverride, setTopOverride] = useState<number | null>(null);
  const [log, setLog] = useState<LogLine[]>([]);
  const logSeq = useRef(1);
  const dealt = useRef(false);
  const [note, setNote] = useState("Your lead. Throw one card or a set of the same rank, then draw.");
  const [hintCards, setHintCards] = useState<number[]>([]);
  const [matchOver, setMatchOver] = useState(false);
  const [sudden, setSudden] = useState(0);
  const [game, setGame] = useState<string>("standard");
  // Tips are the helper texts around the table (the status line, your points, the throw summary, the net's card
  // count). They start hidden; the eye button in the menu shows them, and the choice is remembered on this device.
  const [showTips, setShowTips] = useState(false);
  // An answer from "Ask the net" shows even with tips hidden: it is what the player asked for.
  const [noteIsHint, setNoteIsHint] = useState(false);

  function toggleTips() {
    const next = !showTips;
    setShowTips(next);
    try {
      window.localStorage.setItem(TIPS_KEY, next ? "on" : "off");
    } catch {
      // Not remembered, but still switched for this visit.
    }
  }
  const small = game !== "standard";

  useEffect(() => {
    let cancel = false;
    async function boot() {
      try {
        if (window.localStorage.getItem(TIPS_KEY) === "on") setShowTips(true);
      } catch {
        // Storage can be blocked (private mode): tips simply start hidden.
      }
      const wanted = gameById(new URLSearchParams(window.location.search).get("game"));
      setRules(wanted.hand, wanted.yanivAt);
      setGame(wanted.id);
      const suffix = rulesSuffix() + wanted.net;
      try {
        const [model, dossier] = await Promise.all([
          loadNet(`${MODEL_DIR}/yaniv-net${suffix}.json`).catch((cause) => {
            if (!rulesSuffix() || wanted.net) throw cause;
            throw new Error(
              `No net for ${wanted.hand} cards with Yaniv at ${wanted.yanivAt} yet. Train one with: python main.py ` +
                `--hand ${wanted.hand} --yaniv-at ${wanted.yanivAt} --value-net --export`,
            );
          }),
          // A hand-written net has no training report: do not ask for one.
          wanted.net
            ? null
            : fetch(`${MODEL_DIR}/training-report${suffix}.json`).then((response) => (response.ok ? response.json() : null)),
        ]);
        if (cancel) return;
        setNet(model);
        setReport(dossier as Report | null);
        if (!dealt.current) {
          dealt.current = true;
          setMatch(freshMatch());
        }
        setError(null);
      } catch (cause) {
        if (!cancel) setError(cause instanceof Error ? cause.message : "The model did not load.");
      } finally {
        if (!cancel) setBooting(false);
      }
    }
    void boot();
    return () => {
      cancel = true;
    };
  }, [attempt]);

  const state = match.state;
  const humanView = useMemo(() => makeView(state, 0), [state]);
  const hand = sortHand(cardsFromMask(state.hands[0]));
  const points = maskPoints(state.hands[0]);
  const selectionValid = selected.length > 0 && isValidCombo(selected);
  const top = selected.length && selected.includes(topOverride ?? -1) ? (topOverride as number) : selected.length ? defaultTop(selected) : -1;
  const canCall = !state.terminal && state.turn === 0 && points <= YANIV_LIMIT && hand.length > 0;
  const openCard = state.discard[state.discard.length - 1];

  const aiTurn = Boolean(net) && !matchOver && !state.terminal && state.turn === 1 && state.ply < 48;

  useEffect(() => {
    if (!aiTurn || !net) return;
    const timer = window.setTimeout(() => {
      const view = makeView(state, 1);
      const ranked = chooseAction(net, view);
      const next = cloneState(state);
      if (!ranked.length) {
        resolveShowdown(next);
        setNote("Forty-eight throws and nobody called. Hands are shown.");
      } else {
        const choice = ranked[0].action;
        const sentence = describeAction(view, choice, ranked);
        applyAction(next, choice);
        if (!next.terminal && next.ply >= 48) {
          resolveShowdown(next);
          setNote("Forty-eight throws and nobody called. Hands are shown.");
        } else {
          setNote(sentence);
        }
        const id = logSeq.current;
        logSeq.current += 1;
        const line = `The net ${sentence.charAt(0).toLowerCase()}${sentence.slice(1)}`;
        setLog((lines) => [{ id, text: line, who: "net" as const }, ...lines].slice(0, 8));
      }
      setNoteIsHint(false);
      setMatch((current) => ({ ...current, state: next }));
    }, 720);
    return () => window.clearTimeout(timer);
  }, [aiTurn, net, state]);

  function push(text: string) {
    const id = logSeq.current;
    logSeq.current += 1;
    setLog((lines) => [{ id, text, who: "you" as const }, ...lines].slice(0, 8));
  }

  function toggle(card: number) {
    if (state.turn !== 0 || state.terminal || aiTurn) return;
    setHintCards([]);
    setSelected((current) => (current.includes(card) ? current.filter((item) => item !== card) : current.concat(card)));
  }

  function commit(action: Action, line: string) {
    const next = cloneState(state);
    applyAction(next, action);
    if (!next.terminal && next.ply >= 48) {
      resolveShowdown(next);
      setNote("Forty-eight throws and nobody called. Hands are shown.");
    } else if (next.terminal && next.result) {
      setNote(resultHeadline(next.result, 0));
    }
    push(line);
    setSelected([]);
    setTopOverride(null);
    setHintCards([]);
    setNoteIsHint(false);
    setMatch((current) => ({ ...current, state: next }));
  }

  function throwCards(draw: "stock" | "open") {
    if (!selectionValid || state.turn !== 0 || state.terminal) return;
    if (draw === "stock" && state.stock.length === 0) return;
    const action: Action = { kind: "play", discardMask: maskOf(selected), top, draw };
    const thrown = selected.map(cardName).join(" ");
    const line = `You throw ${thrown} and ${draw === "open" ? `take the ${cardName(openCard)}` : "draw from the deck"}.`;
    commit(action, line);
  }

  function callYaniv() {
    if (!canCall) return;
    commit({ kind: "yaniv", discardMask: 0n, top: -1, draw: "none" }, `You call Yaniv on ${points}.`);
  }

  function askNet() {
    if (!net || state.turn !== 0 || state.terminal) return;
    const ranked = chooseAction(net, humanView);
    if (!ranked.length) return;
    const best = ranked[0].action;
    if (best.kind === "yaniv") {
      setHintCards([]);
      setNote(`The net would call Yaniv on ${points}.`);
      setNoteIsHint(true);
      return;
    }
    setHintCards(cardsFromMask(best.discardMask));
    setTopOverride(best.top);
    setSelected(cardsFromMask(best.discardMask));
    setNote(`Hint: ${describeAction(humanView, best, ranked)}`);
    setNoteIsHint(true);
  }

  function nextRound() {
    const scores = state.scores;
    const limitReached = Math.max(scores[0], scores[1]) >= MATCH_LIMIT;
    if (limitReached && scores[0] !== scores[1]) {
      setMatchOver(true);
      return;
    }
    if (limitReached && sudden >= 2) {
      setMatchOver(true);
      return;
    }
    const leader = 1 - match.leader;
    setSudden(limitReached ? sudden + 1 : 0);
    setSelected([]);
    setMatch({
      state: newRound(scores, leader, rng),
      leader,
      round: match.round + 1,
      startScores: [scores[0], scores[1]],
    });
    setNote(leader === 0 ? "Your lead." : "The net leads this round.");
  }

  function restart() {
    setMatch(freshMatch());
    setSelected([]);
    setLog([]);
    setMatchOver(false);
    setSudden(0);
    setHintCards([]);
    setNote(`New match. Scores add up, and when someone passes ${MATCH_LIMIT} the lower score wins.`);
  }

  if (booting) {
    return (
      <Shell>
        <Status title="Shuffling the deck" body="Loading the trained network." />
      </Shell>
    );
  }
  if (!net) {
    return (
      <Shell>
        <Status
          title="The network is not at the table yet"
          body={error ?? "The weight file has not been written."}
          action={
            <div className="flex flex-wrap gap-2">
              <Button
                className="h-10 bg-[#f2c75c] px-4 text-[#2a2118] hover:bg-[#f6d685]"
                onClick={() => {
                  setBooting(true);
                  setError(null);
                  setAttempt((value) => value + 1);
                }}
              >
                Try again
              </Button>
              {small ? (
                <button
                  type="button"
                  onClick={() => reloadAs("standard")}
                  className="inline-flex h-10 items-center rounded-lg border border-[#f2c75c]/40 px-4 text-sm text-[#f6edd9] hover:bg-white/10"
                >
                  Standard game
                </button>
              ) : null}
            </div>
          }
        />
      </Shell>
    );
  }

  const result = state.result;
  const showResult = Boolean(result) && !matchOver;
  const shed = selectionValid ? selected.reduce((sum, card) => sum + cardPoints(card), 0) : 0;
  const yourTurn = state.turn === 0 && !state.terminal && !aiTurn;
  const canThrow = selectionValid && yourTurn;
  const netCards = popcount(state.hands[1]);

  return (
    <Shell handSize={HAND_SIZE} tipsOff={!showTips}>
      {/* Title, scores and menu. On a sideways phone they share one row. */}
      <div className="flex flex-col gap-2 short:flex-row short:items-center short:gap-3 squat:flex-row squat:items-center desk:gap-0">
        <header className="flex items-center justify-between gap-x-3 gap-y-2 short:contents squat:contents desk:flex-wrap desk:gap-x-4 desk:gap-y-3">
          <div className="min-w-0 short:order-1 squat:sr-only">
            <h1 className="font-display text-2xl leading-none whitespace-nowrap text-[#f6edd9] tab:text-3xl desk:text-4xl">
              Yaniv <span className="text-[#f2c75c]">יניב</span>
            </h1>
            <p className="mt-1 hidden text-sm text-[#e7d7b8]/75 desk:block">
              {gameById(game).net
                ? "You against a neural net whose weights were written by hand, not trained."
                : "You against a neural net that learned the game."}{" "}
              {rulesSuffix() ? `Small game: ${HAND_SIZE} cards, Yaniv at ${YANIV_LIMIT}.` : null}
            </p>
          </div>
          <div className="flex shrink-0 items-center gap-1.5 tiny:gap-1 short:order-3 squat:order-3 desk:gap-2">
            <RulesButton />
            <button
              type="button"
              onClick={toggleTips}
              aria-pressed={showTips}
              aria-label={showTips ? "Hide tips" : "Show tips"}
              title={showTips ? "Hide tips" : "Show tips"}
              className={
                "inline-flex h-9 items-center gap-1.5 rounded-lg border px-2.5 text-sm transition tiny:w-8 tiny:justify-center tiny:px-0 desk:px-3 " +
                (showTips
                  ? "border-[#f2c75c] bg-[#f2c75c]/15 text-[#f6edd9]"
                  : "border-[#f2c75c]/40 text-[#f6edd9]/80 hover:bg-white/10")
              }
            >
              {showTips ? <Eye aria-hidden="true" className="size-4" /> : <EyeOff aria-hidden="true" className="size-4" />}
              <span className="narrow:hidden short:slim:hidden squat:hidden">Tips</span>
            </button>
            <InfoButton log={log} report={report} handmade={Boolean(gameById(game).net)} />
            <Button
              variant="outline"
              aria-label="New match"
              className="h-9 gap-1.5 border-[#f2c75c]/40 bg-transparent px-2.5 text-[#f6edd9] hover:bg-white/10 hover:text-[#f6edd9] tiny:w-8 tiny:justify-center tiny:px-0 desk:px-3"
              onClick={restart}
            >
              <RotateCcw aria-hidden="true" className="desk:hidden" />
              <span className="narrow:hidden short:slim:hidden squat:hidden">New match</span>
            </Button>
          </div>
        </header>

        <ScoreBoard you={state.scores[0]} net={state.scores[1]} round={match.round} />
      </div>

      <div className="mt-2 flex min-h-0 flex-1 flex-col desk:mt-4 desk:grid desk:grid-cols-[minmax(0,1fr)_320px] desk:gap-4">
        <section
          aria-label="Table"
          className="felt relative flex min-h-0 flex-1 flex-col justify-between gap-2 overflow-hidden rounded-[1.25rem] border border-[#5a3a1f] px-2 py-3 shadow-[inset_0_0_90px_rgba(0,0,0,0.4),0_20px_50px_rgba(0,0,0,0.35)] short:grid short:grid-cols-[minmax(0,0.8fr)_minmax(0,1.2fr)] short:gap-x-3 short:px-3 short:py-2 tab:px-4 desk:justify-start desk:gap-7 desk:rounded-[1.75rem] desk:px-8 desk:py-7"
        >
          {/* Left on a sideways phone: the net and the piles. */}
          <div className="flex flex-1 flex-col justify-around gap-2 short:justify-evenly desk:flex-none desk:justify-start desk:gap-7">
            {/* The net */}
            <div className="flex flex-col items-center gap-1.5 squat:flex-row squat:justify-center squat:gap-3 desk:gap-2">
              <div className="flex items-center gap-2 text-sm text-[#f6edd9] squat:flex-col squat:items-start squat:gap-0.5">
                <span className="text-[11px] font-semibold tracking-[0.18em] text-[#f2c75c] uppercase desk:text-xs">The net</span>
                {showTips && (
                  <span className="text-xs text-[#f6edd9]/70 desk:text-sm">
                    {netCards} {netCards === 1 ? "card" : "cards"}
                  </span>
                )}
                {aiTurn && <ThinkingPill />}
              </div>
              <div className="flex justify-center pl-[calc(var(--net-w)*0.4)] [--cw:var(--net-w)]">
                {Array.from({ length: netCards }, (_, index) => (
                  <PlayingCard key={index} faceDown size="fluid" className="-ml-[calc(var(--net-w)*0.4)]" />
                ))}
              </div>
            </div>

            {/* Piles */}
            <div className="flex items-start justify-center gap-8 [--cw:var(--pile-w)] tab:gap-14">
              <PileButton
                label="Deck"
                detail={`${state.stock.length} left`}
                active={canThrow && state.stock.length > 0}
                disabled={!canThrow || state.stock.length === 0}
                onClick={() => throwCards("stock")}
                ariaLabel="Throw the selected cards and draw from the deck"
                showLabel={showTips}
              >
                <div className="relative">
                  {state.stock.length > 2 && (
                    <div className="absolute top-[6px] left-[6px]" aria-hidden="true">
                      <PlayingCard faceDown size="fluid" />
                    </div>
                  )}
                  {state.stock.length > 1 && (
                    <div className="absolute top-[3px] left-[3px]" aria-hidden="true">
                      <PlayingCard faceDown size="fluid" />
                    </div>
                  )}
                  {state.stock.length > 0 ? (
                    <PlayingCard faceDown size="fluid" />
                  ) : (
                    <div className="h-[calc(var(--cw)*1.4)] w-(--cw) rounded-lg border-2 border-dashed border-white/25" />
                  )}
                </div>
              </PileButton>
              <PileButton
                label="Open card"
                detail={`${cardPoints(openCard)} ${cardPoints(openCard) === 1 ? "point" : "points"}`}
                active={canThrow}
                disabled={!canThrow}
                onClick={() => throwCards("open")}
                ariaLabel={`Throw the selected cards and take the ${cardName(openCard)}`}
                showLabel={showTips}
              >
                <PlayingCard card={openCard} size="fluid" className="ring-2 ring-[#f2c75c]/70" />
              </PileButton>
            </div>
          </div>

          {/* Right on a sideways phone: what happened, your hand and your moves. */}
          <div className={(showTips ? "flex-[1.4]" : "flex-1") + " flex flex-col justify-around gap-2 short:justify-evenly desk:flex-none desk:justify-start desk:gap-7"}>
            {/* Status line. Hidden with the tips, but kept for screen readers. */}
            <div className={showTips || noteIsHint ? "flex justify-center" : "sr-only"}>
              <p
                className="flex min-h-[2.5rem] max-w-xl items-center rounded-2xl bg-black/25 px-3 py-1.5 text-center text-[13px] leading-snug text-[#f6edd9] backdrop-blur-[2px] short:min-h-0 short:py-1 short:text-xs desk:min-h-[2.75rem] desk:px-4 desk:py-2.5 desk:text-sm"
                aria-live="polite"
              >
                {aiTurn ? "The net is choosing a throw." : note}
              </p>
            </div>

            {/* Your hand */}
            <div className="flex flex-col items-center gap-1.5 desk:gap-3">
              <div className="flex w-full max-w-xl items-center justify-between gap-2">
                <span className="text-[11px] font-semibold tracking-[0.18em] whitespace-nowrap text-[#f2c75c] uppercase desk:text-xs">
                  Your hand
                </span>
                <span
                  className={
                    (showTips ? "" : "hidden ") +
                    "truncate rounded-full px-2.5 py-0.5 text-xs font-medium desk:px-3 desk:py-1 desk:text-sm " +
                    (points <= YANIV_LIMIT ? "bg-[#f2c75c] text-[#2a2118]" : "bg-black/30 text-[#f6edd9]")
                  }
                >
                  {points} <span className="desk:hidden">{points === 1 ? "pt" : "pts"}</span>
                  <span className="hidden desk:inline">{points === 1 ? "point" : "points"}</span> ·{" "}
                  {points <= YANIV_LIMIT ? "Yaniv is open" : `Yaniv at ${YANIV_LIMIT} or less`}
                </span>
              </div>
              <div className="flex justify-center gap-1.5 pt-[calc(var(--hand-w)*0.24)] [--cw:var(--hand-w)] tab:gap-2.5 desk:gap-3">
                {hand.map((card) => (
                  <PlayingCard
                    key={card}
                    card={card}
                    size="fluid"
                    selected={selected.includes(card)}
                    hinted={hintCards.includes(card)}
                    onClick={() => toggle(card)}
                  />
                ))}
              </div>
              {selectionValid && selected.length > 1 ? (
                <div className="flex min-h-5 flex-wrap items-center justify-center gap-1.5">
                  <span className="text-xs text-[#f6edd9]/80">Leave face up:</span>
                  {selected.map((card) => (
                    <button
                      key={card}
                      type="button"
                      onClick={() => setTopOverride(card)}
                      aria-pressed={card === top}
                      className={
                        "min-h-7 rounded-full px-3 py-0.5 text-xs font-medium transition " +
                        (card === top ? "bg-[#f2c75c] text-[#2a2118]" : "bg-black/30 text-[#f6edd9] hover:bg-black/45")
                      }
                    >
                      {cardName(card)}
                    </button>
                  ))}
                </div>
              ) : !showTips ? null : (
                <p className="min-h-5 text-center text-xs leading-tight text-[#f6edd9]/80 desk:text-sm">
                  {!yourTurn
                    ? state.terminal
                      ? "Round over."
                      : "Wait for the net's throw."
                    : !selected.length
                      ? "Tap one card, or two or more of the same rank."
                      : selectionValid
                        ? `Throwing ${shed} ${shed === 1 ? "point" : "points"}. ${points - shed} stay before the draw.`
                        : "Not one combination. Pick one card, or cards of the same rank."}
                </p>
              )}
            </div>

            {/* Actions. With tips off the two throw buttons are left out: tapping the deck or the open card throws. */}
            <div
              className={
                "mx-auto grid w-full grid-cols-2 gap-2 short:gap-1.5 squat:gap-1.5 " +
                (showTips
                  ? "max-w-2xl short:grid-cols-4 squat:grid-cols-4 tab:max-w-3xl tab:grid-cols-4"
                  : "max-w-md")
              }
            >
              {showTips && (
                <Button
                  className="h-11 px-2 leading-tight whitespace-normal bg-[#f6edd9] text-[#1c1916] hover:bg-white short:h-10 short:text-[13px] squat:text-[13px] desk:h-12"
                  disabled={!canThrow || state.stock.length === 0}
                  onClick={() => throwCards("stock")}
                >
                  <span>
                    Throw · <span className="short:hidden squat:hidden">draw from </span>deck
                  </span>
                </Button>
              )}
              {showTips && (
                <Button
                  className="h-11 border border-[#f6edd9]/25 bg-[#0f3a26] px-2 leading-tight whitespace-normal text-[#f6edd9] hover:bg-[#15503a] short:h-10 short:text-[13px] squat:text-[13px] desk:h-12"
                  disabled={!canThrow}
                  onClick={() => throwCards("open")}
                >
                  Throw · take {cardName(openCard)}
                </Button>
              )}
              <Button
                className={
                  "h-11 px-2 leading-tight font-semibold whitespace-normal short:h-10 short:text-[13px] squat:text-[13px] desk:h-12 " +
                  (canCall && !aiTurn
                    ? "bg-[#f2c75c] text-[#2a2118] shadow-[0_0_0_3px_rgba(242,199,92,0.3)] hover:bg-[#f6d685]"
                    : "border border-[#f2c75c]/40 bg-transparent text-[#f2c75c]")
                }
                disabled={!canCall || aiTurn}
                onClick={callYaniv}
              >
                Call Yaniv{canCall ? ` · ${points}` : ""}
              </Button>
              <Button
                variant="ghost"
                className="h-11 px-2 leading-tight whitespace-normal text-[#f6edd9] hover:bg-white/10 hover:text-[#f6edd9] short:h-10 short:text-[13px] squat:text-[13px] desk:h-12"
                onClick={askNet}
                disabled={aiTurn || state.turn !== 0}
              >
                Ask the net
              </Button>
            </div>
          </div>
        </section>

        {/* On a phone these live behind the Log button instead, so the table fits the screen. */}
        <aside className="hidden flex-col gap-3 desk:flex">
          <RoundLog log={log} />
          <Dossier report={report} handmade={Boolean(gameById(game).net)} />
        </aside>
      </div>

      {showResult && result && (
        <ResultSheet result={result} scores={state.scores} startScores={match.startScores} onContinue={nextRound} />
      )}
      {matchOver && <MatchSheet scores={state.scores} onRestart={restart} />}
    </Shell>
  );
}

/** On a phone the page is exactly one screen tall and the table stretches to fill it: no scrolling mid-game. */
function Shell({ children, handSize = 5, tipsOff = false }: { children: ReactNode; handSize?: number; tipsOff?: boolean }) {
  return (
    <div className="app-frame">
      <main
        className="yaniv-sizes safe-pad mx-auto flex h-full max-w-6xl flex-col desk:h-auto desk:min-h-dvh"
        style={{ "--hand-n": handSize } as CSSProperties}
        data-tips={tipsOff ? "off" : "on"}
      >
        {children}
      </main>
    </div>
  );
}

function Status({ title, body, action }: { title: string; body: string; action?: ReactNode }) {
  return (
    <div className="mx-auto mt-24 max-w-md rounded-2xl border border-[#f2c75c]/30 bg-[#1a140f] p-6 text-[#f6edd9]">
      <h1 className="font-display text-3xl">{title}</h1>
      <p className="mt-2 text-sm text-[#f6edd9]/80">{body}</p>
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

function ThinkingPill() {
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-black/30 px-2 py-0.5 text-xs text-[#f6edd9]">
      thinking
      <span className="flex gap-0.5" aria-hidden="true">
        <span className="h-1 w-1 animate-pulse rounded-full bg-[#f2c75c]" />
        <span className="h-1 w-1 animate-pulse rounded-full bg-[#f2c75c] [animation-delay:150ms]" />
        <span className="h-1 w-1 animate-pulse rounded-full bg-[#f2c75c] [animation-delay:300ms]" />
      </span>
    </span>
  );
}

function PileButton({
  label,
  detail,
  active,
  disabled,
  onClick,
  ariaLabel,
  showLabel,
  children,
}: {
  label: string;
  /** With tips off the label is left out: the gold glow shows when the pile can be tapped. */
  showLabel: boolean;
  detail: string;
  active: boolean;
  disabled: boolean;
  onClick: () => void;
  ariaLabel: string;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      aria-label={ariaLabel}
      className={
        "group flex flex-col items-center gap-1 rounded-xl p-1 text-center desk:gap-2 desk:p-1.5 transition focus-visible:ring-2 focus-visible:ring-white focus-visible:outline-none " +
        (active ? "cursor-pointer hover:-translate-y-1" : "cursor-default")
      }
    >
      <span className={"rounded-lg transition " + (active ? "shadow-[0_0_0_3px_rgba(242,199,92,0.55)]" : "")}>{children}</span>
      <span className={showLabel ? "flex flex-col leading-tight" : "hidden"}>
        <span className="text-[11px] font-semibold tracking-[0.14em] text-[#f6edd9] uppercase desk:text-xs">{label}</span>
        <span className="text-[11px] text-[#f6edd9]/70 squat:hidden desk:text-xs">{active ? "tap to draw" : detail}</span>
      </span>
    </button>
  );
}

function ScoreBoard({ you, net, round }: { you: number; net: number; round: number }) {
  return (
    <div className="grid grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] items-center gap-3 rounded-2xl border border-[#f2c75c]/15 bg-[#1a140f]/80 px-3 py-2 short:order-2 short:flex-1 short:py-1.5 squat:order-2 squat:min-w-0 squat:flex-1 squat:gap-2 squat:px-2.5 squat:py-1.5 tiny:squat:grid-cols-2 tiny:squat:px-2 desk:mt-4 desk:px-4 desk:py-3">
      <ScoreSide name="You" value={you} />
      <div className="text-center tiny:squat:hidden">
        <p className="text-[10px] tracking-[0.18em] text-[#f6edd9]/60 uppercase">
          <span className="squat:hidden">Round</span>
          <span className="hidden squat:inline">Rd</span>
        </p>
        <p className="font-display text-xl leading-none text-[#f6edd9] desk:text-2xl">{round}</p>
      </div>
      <ScoreSide
        name={
          <>
            <span className="squat:hidden">The </span>net
          </>
        }
        value={net}
        alignRight
      />
    </div>
  );
}

function ScoreSide({ name, value, alignRight = false }: { name: ReactNode; value: number; alignRight?: boolean }) {
  const hot = value >= 80;
  const width = Math.min(100, (value / MATCH_LIMIT) * 100);
  return (
    <div className={"min-w-0 " + (alignRight ? "text-right" : "")}>
      <div className={"flex items-baseline gap-2 tiny:gap-1 " + (alignRight ? "justify-end" : "")}>
        <span className="truncate text-[11px] tiny:shrink-0 tracking-[0.14em] text-[#f2c75c] uppercase desk:text-xs">{name}</span>
        <span className={"font-display text-xl leading-none desk:text-2xl " + (hot ? "text-[#ff8a7a]" : "text-[#f6edd9]")}>
          {value}
          <span className="ml-0.5 font-sans text-xs text-[#f6edd9]/45 tiny:hidden short:slim:hidden squat:hidden">/{MATCH_LIMIT}</span>
        </span>
      </div>
      <div className={"mt-1.5 flex h-1.5 overflow-hidden rounded-full bg-white/10 " + (alignRight ? "justify-end" : "")}>
        <div className={"h-full rounded-full " + (hot ? "bg-[#e0614f]" : "bg-[#f2c75c]")} style={{ width: `${width}%` }} />
      </div>
    </div>
  );
}

function RoundLog({ log }: { log: LogLine[] }) {
  return (
    <div className="rounded-2xl border border-[#f2c75c]/15 bg-[#1a140f]/80 p-4 text-sm text-[#f6edd9]">
      <p className="text-xs font-semibold tracking-[0.18em] text-[#f2c75c] uppercase">This round</p>
      <ol className="mt-3 space-y-2.5">
        {log.length === 0 && <li className="text-[#f6edd9]/60">No throws yet.</li>}
        {log.map((line) => (
          <li key={line.id} className="flex gap-2.5">
            <span
              className={"mt-1.5 h-2 w-2 shrink-0 rounded-full " + (line.who === "you" ? "bg-[#f2c75c]" : "bg-[#8fd3ff]")}
              aria-hidden="true"
            />
            <span className="leading-snug">{line.text}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}

function resultHeadline(result: RoundResult, seat: number): string {
  const mine = result.points[seat];
  const theirs = result.points[1 - seat];
  if (result.reason === "showdown") return `Showdown. You have ${mine}, the net has ${theirs}.`;
  if (result.assaf && result.caller === seat) return `Assaf. The net had ${theirs}, which catches your ${mine}.`;
  if (result.assaf && result.caller !== seat) return `Assaf. You had ${mine}, enough to catch the call.`;
  if (result.caller === seat) return `Yaniv. Your ${mine} is lower than their ${theirs}.`;
  return `The net calls Yaniv on ${theirs}. You had ${mine}.`;
}

function ResultSheet({
  result,
  scores,
  startScores,
  onContinue,
}: {
  result: RoundResult;
  scores: [number, number];
  startScores: [number, number];
  onContinue: () => void;
}) {
  const youWon = result.added[0] === 0 && result.added[1] > 0;
  return (
    <div className="safe-pad fixed inset-0 z-40 grid place-items-center bg-black/60 backdrop-blur-[2px]">
      <div className="max-h-full w-full max-w-lg overflow-y-auto rounded-2xl bg-[#fbf6ee] p-4 text-[#241c14] shadow-2xl tab:p-6">
        <p className={"text-xs font-semibold tracking-[0.18em] uppercase " + (youWon ? "text-[#1d6a42]" : "text-[#8a6232]")}>
          {result.assaf ? "Assaf" : result.reason === "showdown" ? "Showdown" : "Yaniv"}
        </p>
        <h2 className="font-display mt-1 text-2xl leading-tight short:text-xl tab:text-3xl">{resultHeadline(result, 0)}</h2>
        <p className="mt-2 text-sm text-[#5c5144] short:hidden">
          {result.assaf
            ? `A caught call adds your hand plus ${ASSAF_PENALTY}. The player with the lower or equal hand scores 0.`
            : "The lower hand scores 0. The other hand adds its points."}
        </p>
        <div className="mt-4 grid grid-cols-2 gap-3 short:mt-2">
          <Reveal
            title="You"
            cards={cardsFromMask(result.hands[0])}
            points={result.points[0]}
            added={result.added[0]}
            before={startScores[0]}
            after={scores[0]}
          />
          <Reveal
            title="The net"
            cards={cardsFromMask(result.hands[1])}
            points={result.points[1]}
            added={result.added[1]}
            before={startScores[1]}
            after={scores[1]}
          />
        </div>
        <Separator className="my-4 short:my-2" />
        <p className="text-sm">
          Match score: you <b>{scores[0]}</b>, the net <b>{scores[1]}</b>. The match ends once someone passes {MATCH_LIMIT}.
        </p>
        {/* Pinned to the bottom of the box, so it stays in reach when a sideways phone has to scroll the box. */}
        <div className="sticky -bottom-4 -mx-4 mt-4 bg-[#fbf6ee] px-4 pb-4 short:mt-2 short:pt-1 tab:static tab:mx-0 tab:px-0 tab:pb-0 short:tab:sticky short:tab:-bottom-6 short:tab:-mx-6 short:tab:px-6 short:tab:pb-6">
          <Button className="h-11 w-full bg-[#1d6a42] text-[#f6edd9] hover:bg-[#24804f]" onClick={onContinue}>
            Continue
          </Button>
        </div>
      </div>
    </div>
  );
}

function Reveal({
  title,
  cards,
  points,
  added,
  before,
  after,
}: {
  title: string;
  cards: number[];
  points: number;
  added: number;
  before: number;
  after: number;
}) {
  const landed = before + added;
  const cut = landed !== after;
  return (
    <div className="min-w-0">
      <p className="text-sm font-semibold">{title}</p>
      <p className="text-xs text-[#5c5144]">
        {points} in hand · {added === 0 ? "scores 0" : `+${added}`}
      </p>
      {cut && (
        <p className="mt-1 inline-block rounded-full bg-[#1d6a42]/10 px-2 py-0.5 text-xs font-medium text-[#1d6a42]">
          Landed on {landed}, cut to {after}
        </p>
      )}
      <div className="mt-2 flex min-h-16 flex-wrap gap-1">
        {cards.length === 0 && <span className="text-xs">Empty hand</span>}
        {sortHand(cards).map((card) => (
          <PlayingCard key={card} card={card} size="sm" />
        ))}
      </div>
    </div>
  );
}

function MatchSheet({ scores, onRestart }: { scores: [number, number]; onRestart: () => void }) {
  const youWin = scores[0] < scores[1];
  const tie = scores[0] === scores[1];
  return (
    <div className="safe-pad fixed inset-0 z-40 grid place-items-center bg-black/60 backdrop-blur-[2px]">
      <div className="max-h-full w-full max-w-md overflow-y-auto rounded-2xl bg-[#fbf6ee] p-4 text-[#241c14] shadow-2xl tab:p-6">
        <h2 className="font-display text-3xl tab:text-4xl">{tie ? "Split match" : youWin ? "You take the match" : "The net takes the match"}</h2>
        <p className="mt-2 text-sm text-[#5c5144]">
          {tie
            ? `Both finish on ${scores[0]}.`
            : `You finish on ${scores[0]}. The net finishes on ${scores[1]}. The lower score wins once someone passes ${MATCH_LIMIT}.`}
        </p>
        <Button className="mt-4 h-11 w-full bg-[#1d6a42] text-[#f6edd9] hover:bg-[#24804f]" onClick={onRestart}>
          Deal another match
        </Button>
      </div>
    </div>
  );
}

function DossierBody({ report, handmade = false }: { report: Report | null; handmade?: boolean }) {
  return report ? (
    <div className="space-y-3 text-sm leading-relaxed text-[#f6edd9]/90">
      {report.config?.copy_from && !report.self_play_iterations ? (
        <p>
          It learned by copying an earlier net that a search teacher had trained, over{" "}
          {report.expert_decisions.toLocaleString()} decisions across {report.expert_rounds.toLocaleString()} rounds. It
          reads the cards by rank only, since no rule here looks at suits, and that makes it play better than the net it
          copied.
        </p>
      ) : report.recipe === "search" && report.self_play_iterations ? (
        <p>
          It first copied a card counter for {report.expert_decisions.toLocaleString()} decisions across{" "}
          {report.expert_rounds.toLocaleString()} rounds. Then a search teacher took over: over{" "}
          {report.self_play_iterations} passes it played {report.self_play_games.toLocaleString()} rounds against the counter, and
          for each of {report.self_play_winning_decisions.toLocaleString()} decisions it imagined{" "}
          {report.config?.search_samples ?? "dozens of"} possible deals, played every candidate move to the end of the round, and
          kept the best. The net learned those moves, and each improved net became the next teacher.{" "}
          {report.best_generation === 0
            ? "No pass beat the copy on new games, so the net here is the copy itself."
            : `The net here is from pass ${report.best_generation}.`}
        </p>
      ) : (report.recipe === "genetic" || report.recipe === "search") && !report.self_play_iterations ? (
        <p>
          It copied a card counter for {report.expert_decisions.toLocaleString()} decisions across{" "}
          {report.expert_rounds.toLocaleString()} rounds of the current rules. It has not evolved past that copy yet, so it
          plays about as well as the counter it learned from.
        </p>
      ) : report.recipe === "genetic" ? (
        <p>
          It first copied a card counter for {report.expert_decisions.toLocaleString()} decisions
          across {report.expert_rounds.toLocaleString()} rounds. Then a population of {report.config?.population ?? "many"} networks
          evolved for {report.self_play_iterations ?? 0} generations. Each generation every network played{" "}
          {report.config?.fitness_rounds ?? "a few hundred"} rounds against the counter on the same deals, and the fittest were kept
          or bred into the next generation by crossover and mutation.{" "}
          {report.best_generation === 0
            ? "No evolved champion beat the copy on new games, so the net here is the copy itself."
            : `The net here is the champion of generation ${report.best_generation}.`}
        </p>
      ) : (
        <p>
          It first copied a card counter for {report.expert_decisions.toLocaleString()} decisions
          across {report.expert_rounds.toLocaleString()} rounds, then played {report.self_play_games.toLocaleString()} games
          against itself and that counter. Only the throws from rounds it won were kept
          ({report.self_play_winning_decisions.toLocaleString()} of them).
        </p>
      )}
      <p>
        In direct matches it won {report.match_record_vs_benchmark.wins}, lost {report.match_record_vs_benchmark.losses},
        and tied {report.match_record_vs_benchmark.ties}. Over the measured rounds it averages {report.round_record_vs_benchmark.hero_avg.toFixed(1)} points
        to the counter&apos;s {report.round_record_vs_benchmark.villain_avg.toFixed(1)}. {Math.round(report.round_record_vs_benchmark.assaf_rate * 100)}% of its Yaniv calls are caught.
      </p>
      <p>
        Against a player who only dumps the fattest set it wins {Math.round(report.round_win_rate_vs_greedy * 100)}% of rounds,
        averaging {report.round_record_vs_greedy.hero_avg.toFixed(1)} points to {report.round_record_vs_greedy.villain_avg.toFixed(1)}.
        Against random legal throws it wins {Math.round(report.round_win_rate_vs_random * 100)}%.
        The counter it copied wins {Math.round(report.benchmark_vs_greedy.win_rate * 100)}% of those same dump-the-fat-set rounds.
        {report.experts && report.experts > 1 && report.decider
          ? `It is a learned decider (${report.decider.join(" → ")}) that weighs ${report.experts} nets of ${report.layers.join(" → ")}, so different nets can lead in different parts of a round.`
          : `The network is ${report.layers.join(" → ")}.`} {report.seconds ? `The training run took ${Math.round(report.seconds / 60)} minutes.` : ""}
      </p>
    </div>
  ) : handmade ? (
    <p className="text-sm text-[#f6edd9]/80">
      This net was not trained: every weight was written by hand, unit by unit, from what each unit should detect.
    </p>
  ) : (
    <p className="text-sm text-[#f6edd9]/80">The training notes are not in this build.</p>
  );
}

function Dossier({ report, handmade = false }: { report: Report | null; handmade?: boolean }) {
  return (
    <div className="rounded-2xl border border-[#f2c75c]/15 bg-[#1a140f]/80 p-4">
      <p className="text-xs font-semibold tracking-[0.18em] text-[#f2c75c] uppercase">How it learned</p>
      <div className="mt-3">
        <DossierBody report={report} handmade={handmade} />
      </div>
    </div>
  );
}

/** Phones only: the round log and the training notes, in a sheet that slides up over the table. */
function InfoButton({ log, report, handmade }: { log: LogLine[]; report: Report | null; handmade: boolean }) {
  return (
    <Sheet>
      <SheetTrigger
        aria-label="Round log and how the net learned"
        className="inline-flex h-9 items-center gap-1.5 rounded-lg border border-[#f2c75c]/40 px-2.5 text-sm text-[#f6edd9] hover:bg-white/10 tiny:w-8 tiny:justify-center tiny:px-0 desk:hidden"
      >
        <ScrollText aria-hidden="true" className="size-4" />
        <span className="narrow:hidden short:slim:hidden squat:hidden">Log</span>
      </SheetTrigger>
      <SheetContent side="bottom" className="max-h-[calc(var(--app-h,100dvh)*0.85)] overflow-y-auto border-[#f2c75c]/20 bg-[#1a140f] pb-[max(env(safe-area-inset-bottom),2rem)] text-[#f6edd9]">
        <SheetHeader>
          <SheetTitle className="text-[#f6edd9]">Log</SheetTitle>
          <SheetDescription className="text-[#f6edd9]/70">This round&apos;s throws, latest first, and how the net learned.</SheetDescription>
        </SheetHeader>
        <div className="space-y-4 px-4">
          <RoundLog log={log} />
          <Dossier report={report} handmade={handmade} />
        </div>
      </SheetContent>
    </Sheet>
  );
}

function RulesButton() {
  return (
    <Dialog>
      <DialogTrigger
        aria-label="Rules"
        className="inline-flex h-9 items-center gap-1.5 rounded-lg border border-[#f2c75c]/40 px-2.5 text-sm text-[#f6edd9] hover:bg-white/10 tiny:w-8 tiny:justify-center tiny:px-0 desk:px-3"
      >
        <BookOpen aria-hidden="true" className="size-4 desk:hidden" />
        <span className="narrow:hidden short:slim:hidden squat:hidden">Rules</span>
      </DialogTrigger>
      <DialogContent className="max-h-[calc(var(--app-h,100dvh)*0.85)] overflow-y-auto bg-[#fbf6ee] text-[#241c14] sm:max-w-lg">
        <DialogHeader>
          <DialogTitle className="font-display text-2xl">How this table plays</DialogTitle>
          <DialogDescription>Two players, one deck with two jokers. Lowest score wins.</DialogDescription>
        </DialogHeader>
        <ul className="list-disc space-y-2 pl-5 text-sm">
          <li>{HAND_SIZE === 5 ? "Five" : HAND_SIZE} cards each. Ace is 1, numbers are their pips, 10, jack, queen and king are 10, and the two jokers are 0.</li>
          <li>On your turn throw one combination: a single card, or two or more cards of the same rank. Two jokers together count as a pair. A joker is not wild.</li>
          <li>You choose which thrown card stays face up. Then draw either the previous open card or the top of the deck. You cannot draw a card you just threw.</li>
          <li>If your whole hand is one combination you may throw all of it, but you still draw, so you always hold at least one card.</li>
          <li>
            With {YANIV_LIMIT} points or fewer you may call Yaniv instead of throwing. If your hand is strictly lower, you score 0 and the net scores its hand. If the net has the same or less, you are Assaf: you take your points plus {ASSAF_PENALTY}, and the net scores 0.
          </li>
          <li>The deck is not reshuffled. When it is empty you can only take the open card. After 48 throws the hands are shown and the lower one scores 0.</li>
          <li>Scores add up. Landing exactly on 50 cuts your score to 0, and landing exactly on 100 cuts it to 50.</li>
          <li>When someone passes {MATCH_LIMIT}, the lower score wins the match.</li>
        </ul>
      </DialogContent>
    </Dialog>
  );
}
