import type { GameState } from "../api";
import CardView from "./CardView";
import styles from "./TrickArea.module.css";

interface TrickAreaProps {
  state: GameState;
}

export default function TrickArea({ state }: TrickAreaProps) {
  const { trump_card, current_trick, num_players } = state;

  return (
    <div className={styles.container}>
      <div className={styles.trumpSection}>
        <span className={styles.label}>Trump</span>
        {trump_card ? (
          <CardView card={trump_card} small />
        ) : (
          <span className={styles.noTrump}>None</span>
        )}
      </div>

      <div className={styles.trickSection}>
        <span className={styles.label}>Current Trick</span>
        <div className={styles.trickCards}>
          {current_trick.length === 0 ? (
            <span className={styles.emptyTrick}>—</span>
          ) : (
            current_trick.map((entry) => (
              <div key={entry.player} className={styles.trickEntry}>
                <span className={styles.playerLabel}>
                  {entry.player === 0 ? "You" : `A${entry.player}`}
                </span>
                <CardView card={entry.card} small />
              </div>
            ))
          )}
          {/* Placeholder slots for missing players so layout stays stable */}
          {Array.from({
            length: Math.max(0, num_players - current_trick.length),
          }).map((_, i) => (
            <div key={`empty-${i}`} className={styles.trickEntry}>
              <span className={styles.playerLabel}>&nbsp;</span>
              <div className={styles.emptySlot} />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
