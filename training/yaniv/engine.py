"""Two-player Yaniv.

Rules encoded here are the table the neural net trains on, and they must
match the TypeScript port card for card.

- 52-card deck. Ace is 1, numbers are face value, 10/J/Q/K are 10.
- Five cards each. One card starts the discard pile. The rest is the stock.
- On your turn, throw one legal combination, then draw one card.
  A combination is a single card, two or more of the same rank, or three or
  more consecutive ranks of one suit. Ace is low only (A-2-3 is a run,
  K-A-2 is not).
- You choose which thrown card stays face up. The card you may draw from
  the discard is the one that was face up before you threw, not a card you
  just played.
- If the whole hand is one combination, you may throw it and go out for 0.
- Otherwise you draw the previous open card or the top of the stock.
- At the start of your turn, with 5 points or fewer, you may call Yaniv
  instead of throwing. If your hand is strictly lower than the opponent's,
  you score 0 and they score their hand. If they have the same or less,
  you are Assaf: you score your hand plus 30, and they score 0.
- The stock is never reshuffled. Once it is empty you can only take the
  open card. After 48 throws the round ends in a showdown: the lower hand
  scores 0, a tie scores both hands, and there is no Assaf.
- A match ends when someone reaches 100. The lower score wins.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

YANIV_LIMIT = 5
ASSAF_PENALTY = 30
MATCH_LIMIT = 100
MAX_PLIES = 48

RANK_CHARS = "A23456789TJQK"
SUIT_CHARS = "♣♦♥♠"

# card id = suit * 13 + rank, rank 0 = Ace ... 12 = King
POINTS = [1 if r == 0 else 10 if r >= 9 else r + 1 for r in range(13)]
POINTS_OF = [POINTS[c % 13] for c in range(52)]


def card_points(card: int) -> int:
    return POINTS_OF[card]


def card_name(card: int) -> str:
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
    n = len(cards)
    if n == 0:
        return False
    if n == 1:
        return True
    rank0 = cards[0] % 13
    if all(c % 13 == rank0 for c in cards):
        return True
    if n < 3:
        return False
    suit0 = cards[0] // 13
    if any(c // 13 != suit0 for c in cards):
        return False
    ranks = sorted(c % 13 for c in cards)
    for i in range(1, n):
        if ranks[i] != ranks[i - 1] + 1:
            return False
    return True


@lru_cache(maxsize=250_000)
def best_keep_points(mask: int) -> int:
    """Fewest points you can hold after one legal throw, before drawing.

    Going out is 0. Points are positive, so the best throw is the legal
    combination worth the most points: the highest single card, an entire
    same-rank set, or an entire same-suit contiguous run.
    """
    if mask == 0:
        return 0
    cards = cards_from_mask(mask)
    if len(cards) == 1 or is_valid_combo(cards):
        return 0
    total = 0
    by_rank_points: dict[int, int] = {}
    by_rank_count: dict[int, int] = {}
    by_suit: dict[int, list[int]] = {}
    max_single = 0
    for c in cards:
        p = POINTS_OF[c]
        total += p
        if p > max_single:
            max_single = p
        rank = c % 13
        by_rank_points[rank] = by_rank_points.get(rank, 0) + p
        by_rank_count[rank] = by_rank_count.get(rank, 0) + 1
        by_suit.setdefault(c // 13, []).append(c)
    best = total - max_single
    for rank, count in by_rank_count.items():
        if count >= 2:
            rem = total - by_rank_points[rank]
            if rem < best:
                best = rem
    for group in by_suit.values():
        if len(group) < 3:
            continue
        group.sort(key=lambda c: c % 13)
        start = 0
        while start < len(group):
            end = start
            while (
                end + 1 < len(group)
                and group[end + 1] % 13 == group[end] % 13 + 1
            ):
                end += 1
            if end - start + 1 >= 3:
                shed = sum(POINTS_OF[group[i]] for i in range(start, end + 1))
                rem = total - shed
                if rem < best:
                    best = rem
            start = end + 1
    return best


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
    reason: str  # "yaniv", "go_out", "showdown"
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
    deck = list(range(52))
    rng.shuffle(deck)
    hands = [0, 0]
    for i in range(10):
        hands[i % 2] |= 1 << deck[i]
    return State(
        hands=hands,
        known=[0, 0],
        stock=deck[11:],
        discard=[deck[10]],
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
    state.scores = [state.scores[0] + added[0], state.scores[1] + added[1]]


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
        if len(chosen) == n:
            top = max(chosen, key=lambda c: (POINTS_OF[c], c))
            actions.append(Action("play", mask, top, "none"))
            continue
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
            raise ValueError("Yaniv is only legal at 5 points or fewer")
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
    remaining = hand & ~discard_mask
    ordered = [c for c in chosen if c != action.top] + [action.top]
    state.known[me] &= ~discard_mask
    state.discard.extend(ordered)
    state.ply += 1
    if remaining == 0:
        state.hands[me] = 0
        state.known[me] = 0
        opp = 1 - me
        added = [0, 0]
        added[opp] = mask_points(state.hands[opp])
        _finish(state, added, False, None, "go_out", me)
        return
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
        raise ValueError("a throw that leaves cards must draw")
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
