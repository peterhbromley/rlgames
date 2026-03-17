import type { Card, LegalAction } from "../api";
import CardView from "./CardView";
import styles from "./Hand.module.css";

interface HandProps {
  cards: Card[];
  legalActions: LegalAction[];
  isMyTurn: boolean;
  disabled: boolean;
  onPlay: (actionId: number) => void;
}

export default function Hand({
  cards,
  legalActions,
  isMyTurn,
  disabled,
  onPlay,
}: HandProps) {
  const playableActionIds = new Set(
    legalActions.filter((a) => a.type === "play").map((a) => a.id)
  );

  // Map card id → action id for playable cards
  // The action label is like "H7" or "CA"; card.label matches exactly
  const cardLabelToActionId = new Map<string, number>(
    legalActions
      .filter((a) => a.type === "play")
      .map((a) => [a.label, a.id])
  );

  return (
    <div className={styles.container}>
      <span className={styles.label}>Your Hand</span>
      <div className={styles.cards}>
        {cards.length === 0 ? (
          <span className={styles.empty}>No cards</span>
        ) : (
          cards.map((card) => {
            const actionId = cardLabelToActionId.get(card.label);
            const clickable =
              isMyTurn && !disabled && actionId !== undefined;
            return (
              <CardView
                key={card.id}
                card={card}
                clickable={clickable}
                onClick={
                  clickable ? () => onPlay(actionId!) : undefined
                }
              />
            );
          })
        )}
      </div>
      {isMyTurn && playableActionIds.size > 0 && !disabled && (
        <p className={styles.hint}>Click a card to play it.</p>
      )}
    </div>
  );
}
