"use client";

import { RANK_CHARS, cardLabel } from "@/lib/yaniv/engine";

/** "fluid" takes its width from the --cw custom property that the surrounding table sets. */
export type CardSize = "sm" | "md" | "lg" | "fluid";

const SIZES: Record<CardSize, string> = {
  sm: "h-16 w-11 rounded-md",
  md: "h-[5.4rem] w-[3.8rem] rounded-lg sm:h-28 sm:w-20",
  lg: "h-[5.9rem] w-[4.1rem] rounded-lg sm:h-[8.4rem] sm:w-[6rem]",
  fluid: "h-[calc(var(--cw)*1.4)] w-(--cw) rounded-[calc(var(--cw)*0.1)]",
};

const CORNER: Record<CardSize, string> = {
  sm: "text-[10px]",
  md: "text-xs sm:text-sm",
  lg: "text-sm sm:text-base",
  fluid: "text-[length:max(10px,calc(var(--cw)*0.18))]",
};

const CENTER: Record<CardSize, string> = {
  sm: "text-xl",
  md: "text-3xl sm:text-4xl",
  lg: "text-3xl sm:text-5xl",
  fluid: "text-[length:calc(var(--cw)*0.5)]",
};

const SUITS = "♣♦♥♠";
const FIRST_JOKER = 52;

/** Card ids 0..51 are suit * 13 + rank. 52 is the red joker, 53 the black one. */
function isJokerCard(card: number): boolean {
  return card >= FIRST_JOKER;
}

function labelFor(card: number): string {
  if (isJokerCard(card)) return card === FIRST_JOKER ? "red joker" : "black joker";
  return cardLabel(card);
}

export function PlayingCard({
  card,
  selected = false,
  hinted = false,
  faceDown = false,
  dimmed = false,
  size = "md",
  onClick,
  className = "",
}: {
  card?: number;
  selected?: boolean;
  hinted?: boolean;
  faceDown?: boolean;
  dimmed?: boolean;
  size?: CardSize;
  onClick?: () => void;
  className?: string;
}) {
  const shared = `relative shrink-0 select-none transition duration-150 ease-out ${SIZES[size]} ${className}`;
  if (faceDown || card === undefined) {
    return (
      <div
        className={`${shared} border border-[#c9a45c]/60 bg-[#1b2440] shadow-[0_2px_6px_rgba(0,0,0,0.45)]`}
        aria-hidden="true"
      >
        <div className="absolute inset-[5px] rounded-[5px] border border-[#c9a45c]/45 bg-[radial-gradient(circle_at_center,#2b3a66_0,#1b2440_70%)]">
          <div className="absolute inset-0 rounded-[4px] bg-[repeating-linear-gradient(45deg,transparent_0_7px,rgba(201,164,92,0.16)_7px_8px),repeating-linear-gradient(-45deg,transparent_0_7px,rgba(201,164,92,0.16)_7px_8px)]" />
          <div className="absolute top-1/2 left-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rotate-45 border border-[#c9a45c]/70 bg-[#1b2440]" />
        </div>
      </div>
    );
  }

  const Tag = onClick ? "button" : "div";
  const label = labelFor(card);
  const interactive = onClick ? "cursor-pointer hover:-translate-y-1.5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white " : "";
  const state =
    (selected ? (size === "fluid" ? "-translate-y-[calc(var(--cw)*0.22)] " : "-translate-y-4 ") + "ring-[3px] ring-[#f2c75c] shadow-[0_10px_22px_rgba(0,0,0,0.45)] " : "shadow-[0_2px_6px_rgba(0,0,0,0.35)] ") +
    (hinted && !selected ? "ring-[3px] ring-[#8fd3ff] " : "") +
    (dimmed ? "opacity-45 " : "");

  if (isJokerCard(card)) {
    const red = card === FIRST_JOKER;
    return (
      <Tag
        type={onClick ? "button" : undefined}
        onClick={onClick}
        aria-label={label}
        aria-pressed={onClick ? selected : undefined}
        className={`${shared} border border-[#e2d6bf] bg-[#fbf7ef] ${red ? "text-[#c3362c]" : "text-[#1c1916]"} ${state}${interactive}`}
      >
        <span className={`absolute top-1 left-1.5 leading-none font-bold ${CORNER[size]}`}>★</span>
        <span className={`absolute right-1.5 bottom-1 rotate-180 leading-none font-bold ${CORNER[size]}`}>★</span>
        <span className="absolute inset-0 flex flex-col items-center justify-center gap-0.5">
          <span className={`leading-none ${CENTER[size]}`}>★</span>
          {size === "fluid" ? (
            <span className="text-[length:max(7px,calc(var(--cw)*0.11))] font-semibold tracking-[0.15em]">JOKER</span>
          ) : (
            size !== "sm" && <span className="text-[9px] font-semibold tracking-[0.2em] sm:text-[10px]">JOKER</span>
          )}
        </span>
        <span className="sr-only">{label}</span>
      </Tag>
    );
  }

  const rank = card % 13;
  const suit = Math.floor(card / 13);
  const red = suit === 1 || suit === 2;
  const symbol = SUITS[suit];
  const rankText = rank === 9 ? "10" : RANK_CHARS[rank];
  const face = rank >= 10;
  return (
    <Tag
      type={onClick ? "button" : undefined}
      onClick={onClick}
      aria-label={label}
      aria-pressed={onClick ? selected : undefined}
      className={`${shared} border border-[#e2d6bf] bg-[#fbf7ef] ${red ? "text-[#c3362c]" : "text-[#1c1916]"} ${state}${interactive}`}
    >
      <span className={`absolute top-1 left-1.5 flex flex-col items-center leading-none font-semibold ${CORNER[size]}`}>
        <span>{rankText}</span>
        <span className="mt-0.5">{symbol}</span>
      </span>
      {size !== "sm" && (
        <span className={`absolute right-1.5 bottom-1 flex rotate-180 flex-col items-center leading-none font-semibold ${CORNER[size]}`}>
          <span>{rankText}</span>
          <span className="mt-0.5">{symbol}</span>
        </span>
      )}
      <span className="absolute inset-0 grid place-items-center">
        {face ? (
          <span className={`font-display leading-none font-semibold ${CENTER[size]}`}>
            {"JQK"[rank - 10]}
          </span>
        ) : (
          <span className={`leading-none ${CENTER[size]} ${rank === 0 ? "scale-125" : ""}`}>{symbol}</span>
        )}
      </span>
      <span className="sr-only">{label}</span>
    </Tag>
  );
}
