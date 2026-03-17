import type { Card } from "../api";
import styles from "./CardView.module.css";

const SUIT_SYMBOLS: Record<string, string> = {
  hearts: "♥",
  diamonds: "♦",
  clubs: "♣",
  spades: "♠",
};

function isRed(suit: string): boolean {
  return suit === "hearts" || suit === "diamonds";
}

interface CardViewProps {
  card?: Card;
  faceDown?: boolean;
  clickable?: boolean;
  onClick?: () => void;
  small?: boolean;
}

export default function CardView({
  card,
  faceDown = false,
  clickable = false,
  onClick,
  small = false,
}: CardViewProps) {
  if (faceDown || !card) {
    return (
      <div
        className={`${styles.card} ${styles.faceDown} ${small ? styles.small : ""}`}
      />
    );
  }

  const symbol = SUIT_SYMBOLS[card.suit] ?? card.suit;
  const red = isRed(card.suit);

  return (
    <div
      className={`${styles.card} ${red ? styles.red : styles.black} ${clickable ? styles.clickable : ""} ${small ? styles.small : ""}`}
      onClick={clickable && onClick ? onClick : undefined}
      role={clickable ? "button" : undefined}
      tabIndex={clickable ? 0 : undefined}
      onKeyDown={
        clickable && onClick
          ? (e) => {
              if (e.key === "Enter" || e.key === " ") onClick();
            }
          : undefined
      }
      aria-label={clickable ? `Play ${card.label}` : card.label}
    >
      <span className={styles.rankTop}>{card.rank}</span>
      <span className={styles.suitCenter}>{symbol}</span>
      <span className={styles.rankBottom}>{card.rank}</span>
    </div>
  );
}
