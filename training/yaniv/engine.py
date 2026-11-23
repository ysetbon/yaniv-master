"""Two-player Yaniv.

Rules encoded here are the table the neural net trains on, and they must
match the TypeScript port card for card.

- 54-card deck: 52 cards and two jokers. Ace is 1, numbers are face value,
  10/J/Q/K are 10, a joker is 0.
- Five cards each (HAND_SIZE). One card starts the discard pile. The rest is the stock.
- On your turn, throw one legal combination, then draw one card.
  A combination is a single card or two or more of the same rank. There are
  no runs. The two jokers are a rank of their own, so they pair only with
  each other; a joker is not wild.
- You choose which thrown card stays face up. The card you may draw from
  the discard is the one that was face up before you threw, not a card you
  just played.
- Every throw is followed by a draw: the previous open card or the top of
  the stock. That holds when you throw your whole hand too, so a hand
  never empties and a round only ends by Yaniv or showdown.
- At the start of your turn, with 7 points or fewer (YANIV_LIMIT), you may call Yaniv
  instead of throwing. If your hand is strictly lower than the opponent's,
  you score 0 and they score their hand. If they have the same or less,
  you are Assaf: you score your hand plus 30, and they score 0.
- The stock is never reshuffled. Once it is empty you can only take the
  open card. After 48 throws the round ends in a showdown: the lower hand
  scores 0, a tie scores both hands, and there is no Assaf.
- After a round's points are added, a score that lands exactly on 50 or
  100 is cut by 50 (50 becomes 0, 100 becomes 50).
- A match ends when someone reaches 100. The lower score wins.

A smaller game, quicker to learn, comes from two environment variables read
when this module is first imported (main.py sets them from --hand and
--yaniv-at, and worker processes inherit them): YANIV_HAND_SIZE cards each
(3 to 5, default 5) and Yaniv at YANIV_CALL_AT points or fewer (default 7).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


def _rule(name: str, default: int, low: int, high: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    value = int(raw)
    if not low <= value <= high:
        raise ValueError(f"{name}={value}: give {low} to {high}")
    return value


HAND_SIZE = _rule("YANIV_HAND_SIZE", 5, 3, 5)
YANIV_LIMIT = _rule("YANIV_CALL_AT", 7, 1, 20)
STANDARD_RULES = HAND_SIZE == 5 and YANIV_LIMIT == 7
# Appended to the shipped model's file name, so a smaller game's net never
# replaces the standard one: "" or e.g. "-4cards-yaniv9".
RULES_SUFFIX = "" if STANDARD_RULES else f"-{HAND_SIZE}cards-yaniv{YANIV_LIMIT}"
RULES_NAME = (f"{HAND_SIZE} cards each, Yaniv at {YANIV_LIMIT} or less"
              + ("" if STANDARD_RULES else " (a smaller game than the standard 5 cards, Yaniv at 7)"))
ASSAF_PENALTY = 30
MATCH_LIMIT = 100
MAX_PLIES = 48

RANK_CHARS = "A23456789TJQK"
SUIT_CHARS = "♣♦♥♠"

DECK_SIZE = 54
JOKERS = (52, 53)  # 52 is the red joker, 53 the black one
JOKER_RANK = 13
RANKS = 14  # Ace ... King, then the jokers
SCORE_CUTS = {50: 0, 100: 50}  # a score landing exactly here drops to the value

# card id = suit * 13 + rank for 0..51, rank 0 = Ace ... 12 = King; 52, 53 = jokers
POINTS = [1 if r == 0 else 10 if r >= 9 else r + 1 for r in range(13)]
POINTS_OF = [POINTS[c % 13] for c in range(52)] + [0, 0]


def is_joker(card: int) -> bool:
    return card >= 52


def rank_of(card: int) -> int:
    return JOKER_RANK if card >= 52 else card % 13


def card_points(card: int) -> int:
    return POINTS_OF[card]


def card_name(card: int) -> str:
    if card >= 52:
        return "★"
    return RANK_CHARS[card % 13] + SUIT_CHARS[card // 13]


def popcount(mask: int) -> int:
    return mask.bit_count()


def cards_from_mask(mask: int) -> list[int]:
    out: list[int] = []
    while mask:
        bit = mask & -mask
        out.append(bit.bit_length() - 1)
        mask ^= bit
    return out


@lru_cache(maxsize=250_000)
def mask_points(mask: int) -> int:
    total = 0
    bits = mask
    while bits:
        bit = bits & -bits
        total += POINTS_OF[bit.bit_length() - 1]
        bits ^= bit
    return total


def is_valid_combo(cards: list[int]) -> bool:
    """A single card, or two or more of one rank. There are no runs."""
    if not cards:
        return False
    rank0 = rank_of(cards[0])
    return all(rank_of(c) == rank0 for c in cards)


@lru_cache(maxsize=250_000)
def best_keep_points(mask: int) -> int:
    """Fewest points you can hold after one legal throw, before drawing.

    Going out is 0. Points are never negative, so the best throw is the
    legal combination worth the most points: the highest single card or an
    entire same-rank set.
    """
    if mask == 0:
        return 0
    cards = cards_from_mask(mask)
    if is_valid_combo(cards):
        return 0
    total = 0
    by_rank: dict[int, int] = {}
    for c in cards:
        p = POINTS_OF[c]
        total += p
        rank = rank_of(c)
        by_rank[rank] = by_rank.get(rank, 0) + p
    return total - max(by_rank.values())


def max_shed_points(mask: int) -> int:
    if mask == 0:
        return 0
    return mask_points(mask) - best_keep_points(mask)


@dataclass(frozen=True)
class Action:
    kind: str  # "play" or "yaniv"
    discard_mask: int = 0
    top: int = -1
    draw: str = "none"  # "stock", "open", "none"

    def to_json(self) -> dict:
        return {
            "kind": self.kind,
            "discard_mask": self.discard_mask,
            "top": self.top,
            "draw": self.draw,
        }

    @staticmethod
    def from_json(data: dict) -> "Action":
        return Action(
            kind=data["kind"],
            discard_mask=int(data["discard_mask"]),
            top=int(data["top"]),
            draw=data["draw"],
        )


@dataclass
class RoundResult:
    added: list[int]
    points: list[int]
    hands: list[int]
    assaf: bool
    caller: int | None
    reason: str  # "yaniv" or "showdown"
    winner: int | None

    def to_json(self) -> dict:
        return {
            "added": self.added,
            "points": self.points,
            "hands": self.hands,
            "assaf": self.assaf,
            "caller": self.caller,
            "reason": self.reason,
            "winner": self.winner,
        }


@dataclass
class State:
    hands: list[int]
    known: list[int]
    stock: list[int]  # last element is the top of the stock
    discard: list[int]  # last element is face up
    scores: list[int]
    turn: int
    ply: int
    terminal: bool = False
    result: RoundResult | None = None

    def to_json(self) -> dict:
        return {
            "hands": list(self.hands),
            "known": list(self.known),
            "stock": list(self.stock),
            "discard": list(self.discard),
            "scores": list(self.scores),
            "turn": self.turn,
            "ply": self.ply,
            "terminal": self.terminal,
            "result": None if self.result is None else self.result.to_json(),
        }


@dataclass(frozen=True)
class View:
    """What the player about to act is allowed to know."""

    my_hand: int
    my_known: int
    opp_known: int
    opp_size: int
    stock_size: int
    discard: tuple[int, ...]
    my_score: int
    opp_score: int
    ply: int
    seat: int


def make_view(state: State, seat: int) -> View:
    return View(
        my_hand=state.hands[seat],
        my_known=state.known[seat],
        opp_known=state.known[1 - seat],
        opp_size=popcount(state.hands[1 - seat]),
        stock_size=len(state.stock),
        discard=tuple(state.discard),
        my_score=state.scores[seat],
        opp_score=state.scores[1 - seat],
        ply=state.ply,
        seat=seat,
    )


def new_round(scores: list[int], leader: int, rng) -> State:
    deck = list(range(DECK_SIZE))
    rng.shuffle(deck)
    hands = [0, 0]
    dealt = 2 * HAND_SIZE
    for i in range(dealt):
        hands[i % 2] |= 1 << deck[i]
    return State(
        hands=hands,
        known=[0, 0],
        stock=deck[dealt + 1:],
        discard=[deck[dealt]],
        scores=[scores[0], scores[1]],
        turn=leader,
        ply=0,
    )


def _finish(
    state: State,
    added: list[int],
    assaf: bool,
    caller: int | None,
    reason: str,
    winner: int | None,
) -> None:
    points = [mask_points(state.hands[0]), mask_points(state.hands[1])]
    state.terminal = True
    state.result = RoundResult(
        added=[added[0], added[1]],
        points=points,
        hands=[state.hands[0], state.hands[1]],
        assaf=assaf,
        caller=caller,
        reason=reason,
        winner=winner,
    )
    totals = [state.scores[0] + added[0], state.scores[1] + added[1]]
    state.scores = [SCORE_CUTS.get(t, t) for t in totals]


def resolve_showdown(state: State) -> None:
    p0 = mask_points(state.hands[0])
    p1 = mask_points(state.hands[1])
    if p0 < p1:
        added = [0, p1]
        winner: int | None = 0
    elif p1 < p0:
        added = [p0, 0]
        winner = 1
    else:
        added = [p0, p1]
        winner = None
    _finish(state, added, False, None, "showdown", winner)


def legal_actions(view: View) -> list[Action]:
    if view.ply >= MAX_PLIES or view.my_hand == 0:
        return []
    cards = cards_from_mask(view.my_hand)
    n = len(cards)
    actions: list[Action] = []
    stock_ok = view.stock_size > 0
    for subset in range(1, 1 << n):
        chosen = [cards[i] for i in range(n) if subset & (1 << i)]
        if not is_valid_combo(chosen):
            continue
        mask = 0
        for c in chosen:
            mask |= 1 << c
        for top in chosen:
            if stock_ok:
                actions.append(Action("play", mask, top, "stock"))
            actions.append(Action("play", mask, top, "open"))
    if mask_points(view.my_hand) <= YANIV_LIMIT:
        actions.append(Action("yaniv", 0, -1, "none"))
    actions.sort(
        key=lambda a: (
            0 if a.kind == "play" else 1,
            a.discard_mask,
            a.top,
            0 if a.draw == "stock" else 1 if a.draw == "open" else 2,
        )
    )
    return actions


def apply_action(state: State, action: Action, rng=None) -> None:
    """Apply a legal action. `rng` is unused; draws follow the stock order."""
    del rng
    if state.terminal:
        raise RuntimeError("round is over")
    if state.ply >= MAX_PLIES:
        resolve_showdown(state)
        return
    me = state.turn
    if action.kind == "yaniv":
        if mask_points(state.hands[me]) > YANIV_LIMIT:
            raise ValueError(f"Yaniv is only legal at {YANIV_LIMIT} points or fewer")
        opp = 1 - me
        mine = mask_points(state.hands[me])
        theirs = mask_points(state.hands[opp])
        added = [0, 0]
        if mine < theirs:
            added[opp] = theirs
            _finish(state, added, False, me, "yaniv", me)
        else:
            added[me] = mine + ASSAF_PENALTY
            _finish(state, added, True, me, "yaniv", opp)
        return

    hand = state.hands[me]
    discard_mask = action.discard_mask
    if discard_mask == 0 or (discard_mask & hand) != discard_mask:
        raise ValueError("discard is not a subset of the hand")
    chosen = cards_from_mask(discard_mask)
    if not is_valid_combo(chosen) or action.top not in chosen:
        raise ValueError("illegal combination")
    if action.draw not in ("stock", "open"):
        raise ValueError("every throw must draw from the stock or the open card")
    if action.draw == "stock" and not state.stock:
        raise ValueError("stock is empty")
    remaining = hand & ~discard_mask
    ordered = [c for c in chosen if c != action.top] + [action.top]
    state.known[me] &= ~discard_mask
    state.discard.extend(ordered)
    state.ply += 1
    if action.draw == "stock":
        if not state.stock:
            raise ValueError("stock is empty")
        card = state.stock.pop()
    elif action.draw == "open":
        idx = len(state.discard) - 1 - len(ordered)
        card = state.discard[idx]
        del state.discard[idx]
        state.known[me] |= 1 << card
    else:
        raise ValueError("every throw must draw from the stock or the open card")
    state.hands[me] = remaining | (1 << card)
    state.turn = 1 - me


def play_round(chooser0, chooser1, rng, scores: list[int], leader: int) -> State:
    state = new_round(scores, leader, rng)
    guard = 0
    while not state.terminal and guard < MAX_PLIES + 2:
        guard += 1
        if state.ply >= MAX_PLIES:
            resolve_showdown(state)
            break
        view = make_view(state, state.turn)
        actions = legal_actions(view)
        if not actions:
            resolve_showdown(state)
            break
        chooser = chooser0 if state.turn == 0 else chooser1
        action = chooser(view, actions, rng)
        apply_action(state, action)
    if not state.terminal:
        resolve_showdown(state)
    return state


def play_match(chooser0, chooser1, rng, max_rounds: int = 40) -> list[int]:
    scores = [0, 0]
    leader = 0
    rounds = 0
    while max(scores) < MATCH_LIMIT and rounds < max_rounds:
        state = play_round(chooser0, chooser1, rng, scores, leader)
        scores = state.scores
        leader = 1 - leader
        rounds += 1
    extra = 0
    while scores[0] == scores[1] and extra < 3:
        state = play_round(chooser0, chooser1, rng, scores, leader)
        scores = state.scores
        leader = 1 - leader
        extra += 1
    return scores
