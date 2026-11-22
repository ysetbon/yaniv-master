"use client";

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Badge } from "@/components/ui/badge";
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
  seconds: number;
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
  config?: { population?: number; fitness_rounds?: number };
}

interface LogLine {
  id: number;
  text: string;
}

const rng = { random: () => Math.random() };

function mulberry32(seed: number) {
  let state = seed >>> 0;
  return {
    random() {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      return state / 4294967296;
    },
  };
}

function freshMatch(): { state: State; leader: number; round: number } {
  return { state: newRound([0, 0], 0, rng), leader: 0, round: 1 };
}

export function Game() {
  const [net, setNet] = useState<Net | null>(null);
  const [report, setReport] = useState<Report | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [booting, setBooting] = useState(true);
  const [attempt, setAttempt] = useState(0);
  const [match, setMatch] = useState(() => ({ state: newRound([0, 0], 0, mulberry32(1)), leader: 0, round: 1 }));
  const [selected, setSelected] = useState<number[]>([]);
  const [topOverride, setTopOverride] = useState<number | null>(null);
  const [log, setLog] = useState<LogLine[]>([]);
  const logSeq = useRef(1);
  const dealt = useRef(false);
  const [note, setNote] = useState("Your deal. Throw a card, a set, or a same-suit run, then draw.");
  const [hintCards, setHintCards] = useState<number[]>([]);
  const [matchOver, setMatchOver] = useState(false);
  const [sudden, setSudden] = useState(0);

  useEffect(() => {
    let cancel = false;
    async function boot() {
      try {
        const [model, dossier] = await Promise.all([
          loadNet(),
          fetch("/model/training-report.json").then((response) => (response.ok ? response.json() : null)),
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
  const canCall = !state.terminal && state.turn === 0 && points <= YANIV_LIMIT && points > 0;
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
        setLog((lines) => [{ id, text: line }, ...lines].slice(0, 8));
      }
      setMatch((current) => ({ ...current, state: next }));
    }, 720);
    return () => window.clearTimeout(timer);
  }, [aiTurn, net, state]);

  function push(text: string) {
    const id = logSeq.current;
    logSeq.current += 1;
    setLog((lines) => [{ id, text }, ...lines].slice(0, 8));
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
      return;
    }
    setHintCards(cardsFromMask(best.discardMask));
    setTopOverride(best.top);
    setSelected(cardsFromMask(best.discardMask));
    setNote(`Hint: ${describeAction(humanView, best, ranked)}`);
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
    setMatch({ state: newRound(scores, leader, rng), leader, round: match.round + 1 });
    setNote(leader === 0 ? "Your lead." : "The net leads this round.");
  }

  function restart() {
    setMatch(freshMatch());
    setSelected([]);
    setLog([]);
    setMatchOver(false);
    setSudden(0);
    setHintCards([]);
    setNote("New match. First to 100 loses if the other score is lower.");
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
            <Button
              className="bg-[#e6c27a] text-[#2a2118] hover:bg-[#efd7a4]"
              onClick={() => {
                setBooting(true);
                setError(null);
                setAttempt((value) => value + 1);
              }}
            >
              Try again
            </Button>
          }
        />
      </Shell>
    );
  }

  const result = state.result;
  const showResult = Boolean(result) && !matchOver;
  const shed = selectionValid ? selected.reduce((sum, card) => sum + cardPoints(card), 0) : 0;

  return (
    <Shell>
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="font-display text-3xl leading-none text-[#f6edd9] sm:text-4xl">
            Yaniv <span className="text-[#e6c27a]">יניב</span>
          </p>
          <p className="mt-1 text-sm text-[#e7d7b8]/80">You against a neural net that learned by playing itself.</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Score name="You" value={state.scores[0]} hot={state.scores[0] >= 80} />
          <Score name="The net" value={state.scores[1]} hot={state.scores[1] >= 80} />
          <Badge variant="outline" className="border-[#e6c27a]/40 text-[#f6edd9]">
            Round {match.round}
          </Badge>
          <RulesButton />
          <Button variant="outline" className="border-[#e6c27a]/40 bg-transparent text-[#f6edd9] hover:bg-white/10" onClick={restart}>
            New match
          </Button>
        </div>
      </header>

      <div className="mt-4 grid gap-4 lg:grid-cols-[minmax(0,1fr)_300px]">
        <section className="felt relative overflow-hidden rounded-[2rem] border border-[#6a4324] px-3 py-4 shadow-[inset_0_0_80px_rgba(0,0,0,0.35)] sm:px-6 sm:py-6">
          <div className="flex items-center justify-between gap-3 text-[#f4ead4]">
            <div>
              <p className="text-xs tracking-[0.16em] text-[#e6c27a] uppercase">The net</p>
              <p className="text-sm">{popcount(state.hands[1])} cards {aiTurn ? "· counting" : ""}</p>
            </div>
            <div className="flex justify-end pl-6">
              {Array.from({ length: popcount(state.hands[1]) }, (_, index) => (
                <PlayingCard key={index} faceDown className={index ? "-ml-5 sm:-ml-7" : ""} />
              ))}
            </div>
          </div>

          <div className="mx-auto mt-6 flex max-w-md items-end justify-center gap-6 sm:mt-8">
            <button
              type="button"
              disabled={!selectionValid || state.turn !== 0 || state.stock.length === 0}
              onClick={() => throwCards("stock")}
              className="group text-center disabled:opacity-70"
            >
              <div className="relative mx-auto h-[6.6rem] w-16 sm:h-32 sm:w-[4.4rem]">
                <PlayingCard faceDown className="absolute top-1 left-1" />
                <PlayingCard faceDown />
                <span className="absolute -right-2 -bottom-2 rounded-full bg-[#1a120c] px-1.5 text-[10px] text-[#f6edd9]">
                  {state.stock.length}
                </span>
              </div>
              <span className="mt-2 block text-xs text-[#f6edd9]">Deck</span>
            </button>
            <button
              type="button"
              disabled={!selectionValid || state.turn !== 0}
              onClick={() => throwCards("open")}
              className="text-center disabled:opacity-80"
            >
              <PlayingCard card={openCard} />
              <span className="mt-2 block text-xs text-[#f6edd9]">Open · {cardPoints(openCard)} pts</span>
            </button>
          </div>

          <p className="mx-auto mt-6 max-w-xl text-center text-sm text-[#f6edd9]" aria-live="polite">
            {aiTurn ? "The net is choosing a throw." : note}
          </p>

          <div className="mt-5">
            <div className="mb-2 flex items-baseline justify-between text-[#f6edd9]">
              <p className="text-xs tracking-[0.16em] text-[#e6c27a] uppercase">Your hand</p>
              <p className="text-sm">
                {points} points {points <= YANIV_LIMIT ? "· Yaniv is legal" : `· call at ${YANIV_LIMIT} or less`}
              </p>
            </div>
            <div className="flex justify-center pl-4">
              {hand.map((card, index) => (
                <PlayingCard
                  key={card}
                  card={card}
                  selected={selected.includes(card)}
                  hinted={hintCards.includes(card)}
                  onClick={() => toggle(card)}
                  className={index ? "-ml-4 sm:-ml-5" : ""}
                />
              ))}
            </div>
          </div>

          <div className="mt-4 flex flex-col gap-2 sm:flex-row sm:items-center">
            <Button
              className="h-11 bg-[#f6edd9] text-[#1c1916] hover:bg-white"
              disabled={!selectionValid || state.turn !== 0 || state.stock.length === 0 || aiTurn}
              onClick={() => throwCards("stock")}
            >
              Throw, then draw blind
            </Button>
            <Button
              className="h-11 bg-[#245c3e] text-[#f6edd9] hover:bg-[#2d7350]"
              disabled={!selectionValid || state.turn !== 0 || aiTurn}
              onClick={() => throwCards("open")}
            >
              Throw, then take {cardName(openCard)}
            </Button>
            <Button
              className="h-11 border border-[#e6c27a] bg-transparent text-[#e6c27a] hover:bg-[#e6c27a]/15"
              disabled={!canCall || aiTurn}
              onClick={callYaniv}
            >
              Call Yaniv{canCall ? ` · ${points}` : ""}
            </Button>
            <Button variant="ghost" className="text-[#f6edd9] hover:bg-white/10" onClick={askNet} disabled={aiTurn || state.turn !== 0}>
              Ask the net
            </Button>
          </div>
          <p className="mt-2 min-h-5 text-xs text-[#f6edd9]/80">
            {!selected.length
              ? "Select cards. A pair, three or four of a kind, or a same-suit run of three or more."
              : selectionValid
                ? `Throwing ${shed} points${selected.length > 1 ? `, ${cardName(top)} stays face up` : ""}. ${points - shed} remain before the draw.`
                : "That selection is not one combination."}
          </p>
          {selectionValid && selected.length > 1 && (
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <span className="text-xs text-[#f6edd9]/80">Leave face up</span>
              {selected.map((card) => (
                <button
                  key={card}
                  type="button"
                  onClick={() => setTopOverride(card)}
                  className={
                    "rounded-full px-2 py-0.5 text-xs " +
                    (card === top ? "bg-[#e6c27a] text-[#2a2118]" : "bg-black/20 text-[#f6edd9]")
                  }
                >
                  {cardName(card)}
                </button>
              ))}
            </div>
          )}
        </section>

        <aside className="flex flex-col gap-3">
          <Dossier report={report} />
          <div className="rounded-2xl border border-[#e6c27a]/20 bg-[#1a140f]/80 p-4 text-sm text-[#f6edd9]">
            <p className="text-xs tracking-[0.16em] text-[#e6c27a] uppercase">This round</p>
            <ul className="mt-2 space-y-2">
              {log.length === 0 && <li className="text-[#f6edd9]/70">No throws yet.</li>}
              {log.map((line) => (
                <li key={line.id}>{line.text}</li>
              ))}
            </ul>
          </div>
        </aside>
      </div>

      {showResult && result && (
        <ResultSheet result={result} scores={state.scores} onContinue={nextRound} />
      )}
      {matchOver && <MatchSheet scores={state.scores} onRestart={restart} />}
    </Shell>
  );
}

function Shell({ children }: { children: ReactNode }) {
  return <main className="mx-auto min-h-dvh max-w-6xl px-3 py-4 sm:px-6 sm:py-6">{children}</main>;
}

function Status({ title, body, action }: { title: string; body: string; action?: ReactNode }) {
  return (
    <div className="mx-auto mt-24 max-w-md rounded-2xl border border-[#e6c27a]/30 bg-[#1a140f] p-6 text-[#f6edd9]">
      <h1 className="font-display text-3xl">{title}</h1>
      <p className="mt-2 text-sm text-[#f6edd9]/80">{body}</p>
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

function Score({ name, value, hot }: { name: string; value: number; hot: boolean }) {
  return (
    <div className={"rounded-xl px-3 py-1.5 text-right " + (hot ? "bg-[#6d2a24]" : "bg-[#1a140f]/80")}>
      <p className="text-[10px] tracking-[0.14em] text-[#e6c27a] uppercase">{name}</p>
      <p className="font-display text-xl leading-none text-[#f6edd9]">{value}</p>
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
  onContinue,
}: {
  result: RoundResult;
  scores: [number, number];
  onContinue: () => void;
}) {
  return (
    <div className="fixed inset-0 z-40 grid place-items-center bg-black/55 p-4">
      <div className="w-full max-w-lg rounded-2xl bg-[#fbf6ee] p-5 text-[#241c14] shadow-2xl">
        <p className="text-xs tracking-[0.16em] text-[#8a6232] uppercase">{result.assaf ? "Assaf" : result.reason === "showdown" ? "Showdown" : "Yaniv"}</p>
        <h2 className="font-display text-3xl leading-tight">{resultHeadline(result, 0)}</h2>
        <p className="mt-2 text-sm text-[#5c5144]">
          {result.assaf
            ? `A caught call adds your hand plus ${ASSAF_PENALTY}. The player with the lower or equal hand scores 0.`
            : "The lower hand scores 0. The other hand adds its points."}
        </p>
        <div className="mt-4 grid grid-cols-2 gap-3">
          <Reveal title="You" cards={cardsFromMask(result.hands[0])} points={result.points[0]} added={result.added[0]} />
          <Reveal title="The net" cards={cardsFromMask(result.hands[1])} points={result.points[1]} added={result.added[1]} />
        </div>
        <Separator className="my-4" />
        <p className="text-sm">Match score: you {scores[0]}, the net {scores[1]}. The match ends at {MATCH_LIMIT}.</p>
        <Button className="mt-4 h-11 w-full bg-[#245c3e] text-[#f6edd9] hover:bg-[#2d7350]" onClick={onContinue}>
          Continue
        </Button>
      </div>
    </div>
  );
}

function Reveal({ title, cards, points, added }: { title: string; cards: number[]; points: number; added: number }) {
  return (
    <div>
      <p className="text-sm font-medium">{title}</p>
      <p className="text-xs text-[#5c5144]">
        {points} in hand · {added === 0 ? "scores 0" : `+${added}`}
      </p>
      <div className="mt-2 flex min-h-16 flex-wrap gap-1">
        {cards.length === 0 && <span className="text-xs">Empty hand</span>}
        {sortHand(cards).map((card) => (
          <PlayingCard key={card} card={card} className="h-16 w-11 sm:h-20 sm:w-14" />
        ))}
      </div>
    </div>
  );
}

function MatchSheet({ scores, onRestart }: { scores: [number, number]; onRestart: () => void }) {
  const youWin = scores[0] < scores[1];
  const tie = scores[0] === scores[1];
  return (
    <div className="fixed inset-0 z-40 grid place-items-center bg-black/55 p-4">
      <div className="w-full max-w-md rounded-2xl bg-[#fbf6ee] p-5 text-[#241c14]">
        <h2 className="font-display text-4xl">{tie ? "Split match" : youWin ? "You take the match" : "The net takes the match"}</h2>
        <p className="mt-2 text-sm text-[#5c5144]">
          {tie
            ? `Both finish on ${scores[0]}.`
            : `You finish on ${scores[0]}. The net finishes on ${scores[1]}. Lower score wins once someone reaches ${MATCH_LIMIT}.`}
        </p>
        <Button className="mt-4 h-11 w-full bg-[#245c3e] text-[#f6edd9] hover:bg-[#2d7350]" onClick={onRestart}>
          Deal another match
        </Button>
      </div>
    </div>
  );
}

function Dossier({ report }: { report: Report | null }) {
  const body = report ? (
    <div className="space-y-3 text-sm text-[#f6edd9]">
      {report.recipe === "genetic" ? (
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
        The counter still has the edge when they sit down together.
      </p>
      <p>
        Against a player who only dumps the fattest set it wins {Math.round(report.round_win_rate_vs_greedy * 100)}% of rounds,
        averaging {report.round_record_vs_greedy.hero_avg.toFixed(1)} points to {report.round_record_vs_greedy.villain_avg.toFixed(1)}.
        Against random legal throws it wins {Math.round(report.round_win_rate_vs_random * 100)}%.
        The counter it copied wins {Math.round(report.benchmark_vs_greedy.win_rate * 100)}% of those same dump-the-fat-set rounds.
        The network is {report.layers.join(" → ")}. The training run took {Math.round(report.seconds / 60)} minutes.
      </p>
    </div>
  ) : (
    <p className="text-sm text-[#f6edd9]/80">The training notes are not in this build.</p>
  );
  return (
    <>
      <div className="hidden rounded-2xl border border-[#e6c27a]/20 bg-[#1a140f]/80 p-4 lg:block">
        <p className="text-xs tracking-[0.16em] text-[#e6c27a] uppercase">How it learned</p>
        <div className="mt-2">{body}</div>
      </div>
      <Sheet>
        <SheetTrigger className="inline-flex h-8 items-center rounded-lg border border-[#e6c27a]/40 px-2.5 text-sm text-[#f6edd9] lg:hidden">
          How it learned
        </SheetTrigger>
        <SheetContent side="bottom" className="max-h-[80dvh] overflow-y-auto bg-[#1a140f] pb-8 text-[#f6edd9]">
          <SheetHeader>
            <SheetTitle className="text-[#f6edd9]">How it learned</SheetTitle>
            <SheetDescription className="text-[#f6edd9]/70">
              {report?.recipe === "genetic" ? "Copied a card counter, then evolved a population of nets." : "Copied a card counter, then learned from its own wins."}
            </SheetDescription>
          </SheetHeader>
          {body}
        </SheetContent>
      </Sheet>
    </>
  );
}

function RulesButton() {
  return (
    <Dialog>
      <DialogTrigger className="inline-flex h-8 items-center rounded-lg border border-[#e6c27a]/40 px-2.5 text-sm text-[#f6edd9] hover:bg-white/10">
        Rules
      </DialogTrigger>
      <DialogContent className="max-h-[85dvh] overflow-y-auto bg-[#fbf6ee] text-[#241c14] sm:max-w-lg">
        <DialogHeader>
          <DialogTitle className="font-display text-2xl">How this table plays</DialogTitle>
          <DialogDescription>Two players, one deck, lowest score at 100 wins.</DialogDescription>
        </DialogHeader>
        <ul className="space-y-2 text-sm">
          <li>Five cards each. Ace is 1, numbers are their pips, and 10, jack, queen, and king are 10.</li>
          <li>On your turn throw one combination: a single card, two or more of a rank, or three or more consecutive cards of one suit. Ace is low only, so A-2-3 is a run and queen-king-ace is not.</li>
          <li>You choose which thrown card stays face up. Then draw either the previous open card or the top of the deck. You cannot draw a card you just threw.</li>
          <li>If your whole hand is one combination you may throw all of it, but you still draw, so you always hold at least one card.</li>
          <li>With 5 points or fewer you may call Yaniv instead of throwing. If your hand is strictly lower, you score 0 and the net scores its hand. If the net has the same or less, you are Assaf: you take your points plus 30, and the net scores 0.</li>
          <li>The deck is not reshuffled. When it is empty you can only take the open card. After 48 throws the hands are shown and the lower one scores 0.</li>
          <li>Scores add up. When someone reaches 100, the lower score wins the match.</li>
        </ul>
      </DialogContent>
    </Dialog>
  );
}
