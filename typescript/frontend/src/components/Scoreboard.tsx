import type { GameState } from "../api";
import styles from "./Scoreboard.module.css";

interface ScoreboardProps {
  state: GameState;
}

export default function Scoreboard({ state }: ScoreboardProps) {
  const { players, current_player, num_tricks, scores, phase } = state;

  return (
    <div className={styles.container}>
      <table className={styles.table}>
        <thead>
          <tr>
            <th>Player</th>
            <th>Bid</th>
            <th>Tricks Won</th>
            <th>Round Tricks</th>
            {scores && <th>Score</th>}
          </tr>
        </thead>
        <tbody>
          {players.map((p, i) => {
            const isCurrent = i === current_player && phase !== "terminal";
            return (
              <tr
                key={i}
                className={isCurrent ? styles.currentPlayer : undefined}
              >
                <td>
                  {i === 0 ? "You" : `Agent ${i}`}
                  {isCurrent && (
                    <span className={styles.turnIndicator}> ←</span>
                  )}
                </td>
                <td>{p.bid ?? "—"}</td>
                <td>{p.tricks_won}</td>
                <td>{num_tricks}</td>
                {scores && <td>{scores[i]}</td>}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
