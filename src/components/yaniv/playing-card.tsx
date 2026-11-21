"use client";

import { RANK_CHARS, cardLabel } from "@/lib/yaniv/engine";

const PIPS: Record<number, Array<[number, number]>> = {
  1: [[50, 50]],
  2: [[50, 24], [50, 76]],
  3: [[50, 22], [50, 50], [50, 78]],
  4: [[30, 24], [70, 24], [30, 76], [70, 76]],
  5: [[30, 24], [70, 24], [50, 50], [30, 76], [70, 76]],
  6: [[30, 22], [70, 22], [30, 50], [70, 50], [30, 78], [70, 78]],
  7: [[30, 20], [70, 20], [50, 36], [30, 52], [70, 52], [30, 80], [70, 80]],
  8: [[30, 18], [70, 18], [50, 34], [30, 50], [70, 50], [50, 66], [30, 82], [70, 82]],
  9: [[30, 16], [70, 16], [30, 36], [70, 36], [50, 50], [30, 64], [70, 64], [30, 84], [70, 84]],
  10: [[30, 15], [70, 15], [50, 30], [30, 40], [70, 40], [30, 60], [70, 60], [50, 70], [30, 85], [70, 85]],
};

export function PlayingCard({
  card,
  selected = false,
  hinted = false,
  faceDown = false,
  dimmed = false,
  onClick,
  className = "",
}: {
  card?: number;
  selected?: boolean;
  hinted?: boolean;
  faceDown?: boolean;
  dimmed?: boolean;
  onClick?: () => void;
  className?: string;
}) {
  const shared =
    "relative h-[5.6rem] w-10 shrink-0 rounded-md shadow-md transition duration-150 sm:h-28 sm:w-[4.4rem] " +
    className;
  if (faceDown || card === undefined) {
    return (
      <div
        className={`${shared} border border-[#d6b36a]/70 bg-[#143d2c]`}
        aria-hidden={faceDown}
      >
        <div className="absolute inset-1 rounded-[4px] border border-[#e6c98a]/50 bg-[repeating-linear-gradient(45deg,#1c5c40_0_6px,#164832_6px_12px)]" />
      </div>
    );
  }
  const rank = card % 13;
  const suit = Math.floor(card / 13);
  const red = suit === 1 || suit === 2;
  const symbol = "♣♦♥♠"[suit];
  const face = rank >= 10;
  const layout = PIPS[rank === 0 ? 1 : rank + 1] ?? [];
  const Tag = onClick ? "button" : "div";
  return (
    <Tag
      type={onClick ? "button" : undefined}
      onClick={onClick}
      aria-label={cardLabel(card)}
      aria-pressed={onClick ? selected : undefined}
      className={
        shared +
        " border bg-[#fbf6ee] text-left " +
        (red ? "text-[#c3362c] " : "text-[#1c1916] ") +
        (selected ? "-translate-y-3 border-[#e6c27a] ring-2 ring-[#e6c27a] " : "border-[#e7dcc6] ") +
        (hinted && !selected ? "ring-2 ring-[#9ad7ff] " : "") +
        (dimmed ? "opacity-45 " : "") +
        (onClick ? "cursor-pointer hover:-translate-y-1 " : "")
      }
    >
      <span className="absolute top-0.5 left-1 text-[11px] leading-none font-semibold sm:text-sm">
        {rank === 9 ? "10" : RANK_CHARS[rank]}
        <span className="block text-[10px] sm:text-xs">{symbol}</span>
      </span>
      {face ? (
        <span className="absolute inset-0 grid place-items-center text-2xl font-semibold tracking-tight sm:text-3xl">
          <span>
            {"JQK"[rank - 10]}
            <span className="ml-0.5 text-sm sm:text-base">{symbol}</span>
          </span>
        </span>
      ) : (
        layout.map(([x, y], index) => (
          <span
            key={index}
            className="absolute text-[11px] leading-none sm:text-sm"
            style={{ left: `${x}%`, top: `${y}%`, transform: `translate(-50%, -50%)${y > 55 ? " rotate(180deg)" : ""}` }}
          >
            {symbol}
          </span>
        ))
      )}
      <span className="sr-only">{cardLabel(card)}</span>
    </Tag>
  );
}
