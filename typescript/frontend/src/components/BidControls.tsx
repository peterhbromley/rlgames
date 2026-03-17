import type { LegalAction } from "../api";
import styles from "./BidControls.module.css";

interface BidControlsProps {
  legalActions: LegalAction[];
  disabled: boolean;
  onBid: (actionId: number) => void;
}

export default function BidControls({
  legalActions,
  disabled,
  onBid,
}: BidControlsProps) {
  const bidActions = legalActions.filter((a) => a.type === "bid");

  if (bidActions.length === 0) return null;

  return (
    <div className={styles.container}>
      <span className={styles.label}>Place your bid:</span>
      <div className={styles.buttons}>
        {bidActions.map((action) => {
          // Extract the numeric bid value from the label (e.g. "Bid 3" or "3")
          const match = action.label.match(/\d+/);
          const bidValue = match ? match[0] : action.label;
          return (
            <button
              key={action.id}
              className={styles.bidButton}
              disabled={disabled}
              onClick={() => onBid(action.id)}
            >
              Bid {bidValue}
            </button>
          );
        })}
      </div>
    </div>
  );
}
