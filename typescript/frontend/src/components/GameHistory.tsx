import type { AgentStats } from "../stats";
import styles from "./GameHistory.module.css";

interface Props {
  agentLabel: string;
  stats: AgentStats;
}

export default function GameHistory({ agentLabel, stats }: Props) {
  if (stats.history.length === 0) return null;

  return (
    <div className={styles.container}>
      <div className={styles.heading}>Recent Hands vs {agentLabel}</div>
      <table className={styles.table}>
        <thead>
          <tr>
            <th>#</th>
            <th>Bid</th>
            <th>Won</th>
            <th>Your Score</th>
            <th>Opp Avg</th>
          </tr>
        </thead>
        <tbody>
          {stats.history.map((r) => (
            <tr key={r.hand} className={r.userBidCorrect ? styles.hit : styles.miss}>
              <td>{r.hand}</td>
              <td>{r.userBid}</td>
              <td>{r.userTricksWon}</td>
              <td>{r.userScore}</td>
              <td>{r.agentAvgScore.toFixed(1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
