import type { StatsMap } from "../stats";
import styles from "./StatsPanel.module.css";

interface Props {
  stats: StatsMap;
  agentLabels: Record<string, string>;
  onClear: () => void;
}

export default function StatsPanel({ stats, agentLabels, onClear }: Props) {
  const entries = Object.entries(stats).filter(([, s]) => s.gamesPlayed > 0);

  if (entries.length === 0) return null;

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <span className={styles.heading}>Your Stats</span>
        <button className={styles.clearBtn} onClick={onClear}>
          Reset
        </button>
      </div>
      <table className={styles.table}>
        <thead>
          <tr>
            <th>Opponent</th>
            <th>Games</th>
            <th>Your Avg</th>
            <th>Opp Avg</th>
            <th>Your Bid %</th>
            <th>Opp Bid %</th>
          </tr>
        </thead>
        <tbody>
          {entries.map(([agent, s]) => {
            const userAvg = s.userScoreSum / s.gamesPlayed;
            const agentAvg = s.agentScoreSum / s.gamesPlayed;
            const userBidPct = Math.round((s.userCorrectBids / s.gamesPlayed) * 100);
            const agentBidPct = Math.round((s.agentCorrectBidSum / s.gamesPlayed) * 100);
            const label = agentLabels[agent] ?? agent.toUpperCase();
            const winning = userAvg > agentAvg;
            return (
              <tr key={agent}>
                <td>{label}</td>
                <td>{s.gamesPlayed}</td>
                <td className={winning ? styles.better : styles.worse}>
                  {userAvg.toFixed(2)}
                </td>
                <td>{agentAvg.toFixed(2)}</td>
                <td>{userBidPct}%</td>
                <td>{agentBidPct}%</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
