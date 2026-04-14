import { useState, useCallback, useEffect, useRef } from "react";
import { createSession, deleteSession, getAgents, streamAction } from "./api";
import type { GameState, SessionResponse } from "./api";
import Scoreboard from "./components/Scoreboard";
import TrickArea from "./components/TrickArea";
import BidControls from "./components/BidControls";
import Hand from "./components/Hand";
import StatsPanel from "./components/StatsPanel";
import GameHistory from "./components/GameHistory";
import { loadStats, recordGame, clearStats } from "./stats";
import type { StatsMap } from "./stats";
import styles from "./App.module.css";

type AppPhase = "start" | "playing" | "error";
type AgentType = "nfsp" | "dqn" | "ppo";

const AGENT_OPTIONS: { value: AgentType; label: string }[] = [
  { value: "ppo", label: "PPO" },
  { value: "nfsp", label: "NFSP" },
  { value: "dqn", label: "DQN" },
];

const AGENT_MOVE_DELAY_MS = 700;
const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

export default function App() {
  const [appPhase, setAppPhase] = useState<AppPhase>("start");
  const [agentType, setAgentType] = useState<AgentType>("ppo");
  // null = still fetching; string[] = loaded (may be empty)
  const [availableAgents, setAvailableAgents] = useState<AgentType[] | null>(null);
  const [stats, setStats] = useState<StatsMap>(() => loadStats());
  const statsRecordedRef = useRef(false);
  const [session, setSession] = useState<SessionResponse | null>(null);
  // displayedState is what the board renders — lags behind session.state during animation.
  const [displayedState, setDisplayedState] = useState<GameState | null>(null);
  const [loading, setLoading] = useState(false);
  const [animating, setAnimating] = useState(false);
  const [animatingPlayer, setAnimatingPlayer] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleError = (e: unknown) => {
    setError(e instanceof Error ? e.message : String(e));
    setAppPhase("error");
    setLoading(false);
    setAnimating(false);
    setAnimatingPlayer(null);
  };

  useEffect(() => {
    getAgents()
      .then((agents) => {
        const types = agents
          .map((a) => a.agent as AgentType)
          .filter((a) => AGENT_OPTIONS.some((o) => o.value === a));
        setAvailableAgents(types);
        // If the default selection isn't available, fall back to the first loaded agent.
        setAgentType((current) =>
          types.includes(current) ? current : (types[0] ?? current)
        );
      })
      .catch(() => {
        // Server unreachable — leave availableAgents null so the user can still
        // attempt to start; the error will surface at createSession time.
      });
  }, []);

  // Record stats when the current game first reaches a terminal state.
  useEffect(() => {
    if (
      !statsRecordedRef.current &&
      !animating &&
      displayedState?.phase === "terminal" &&
      displayedState.scores &&
      session
    ) {
      statsRecordedRef.current = true;
      setStats(recordGame(session.agent, displayedState.scores, displayedState.players));
    }
  }, [displayedState, animating, session]);

  const applySession = useCallback(async (response: SessionResponse) => {
    setSession(response);
    setLoading(false);

    if (response.transitions.length > 0) {
      setAnimating(true);
      for (const t of response.transitions) {
        setAnimatingPlayer(t.player);
        setDisplayedState(t.state);
        await sleep(AGENT_MOVE_DELAY_MS);
      }
      setAnimatingPlayer(null);
      setAnimating(false);
    }

    setDisplayedState(response.state);
  }, []);

  const startGame = useCallback(async () => {
    setLoading(true);
    setError(null);
    statsRecordedRef.current = false;
    try {
      await applySession(await createSession(agentType));
      setAppPhase("playing");
    } catch (e) {
      handleError(e);
    }
  }, [applySession, agentType]);

  const playAgain = useCallback(async () => {
    setLoading(true);
    setError(null);
    statsRecordedRef.current = false;
    try {
      if (session) await deleteSession(session.session_id);
      await applySession(await createSession(agentType));
    } catch (e) {
      handleError(e);
    }
  }, [session, applySession, agentType]);

  const handleAction = useCallback(
    async (actionId: number) => {
      if (!session || loading || animating) return;
      setLoading(true);
      try {
        let firstEvent = true;
        for await (const event of streamAction(session.session_id, actionId)) {
          if (firstEvent) {
            setLoading(false);
            setAnimating(true);
            firstEvent = false;
          }
          setDisplayedState(event.state);
          if (event.type === "transition") {
            setAnimatingPlayer(event.player);
          } else {
            // final: settle and re-enable interaction
            setAnimatingPlayer(null);
            setAnimating(false);
          }
        }
      } catch (e) {
        handleError(e);
      }
    },
    [session, loading, animating]
  );

  // ── Start screen ──────────────────────────────────────────────────────────
  if (appPhase === "start") {
    const agentOptions =
      availableAgents !== null
        ? AGENT_OPTIONS.filter((o) => availableAgents.includes(o.value))
        : AGENT_OPTIONS;
    const noAgents = availableAgents !== null && availableAgents.length === 0;

    return (
      <div className={styles.startScreen}>
        <h1 className={styles.title}>Oh Hell</h1>
        <p className={styles.subtitle}>A trick-taking card game</p>
        <div className={styles.agentSelector}>
          <label htmlFor="agent-select">Opponent: </label>
          <select
            id="agent-select"
            value={agentType}
            onChange={(e) => setAgentType(e.target.value as AgentType)}
            disabled={noAgents || loading}
          >
            {agentOptions.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>
        {noAgents && (
          <p className={styles.warningMsg}>
            No trained agents are available. Start the server with a valid checkpoint.
          </p>
        )}
        <button
          className={styles.primaryButton}
          onClick={startGame}
          disabled={loading || noAgents}
        >
          {loading ? "Loading…" : "New Game"}
        </button>
        <StatsPanel
          stats={stats}
          agentLabels={Object.fromEntries(AGENT_OPTIONS.map((o) => [o.value, o.label]))}
          onClear={() => setStats(clearStats())}
        />
      </div>
    );
  }

  // ── Error screen ──────────────────────────────────────────────────────────
  if (appPhase === "error") {
    return (
      <div className={styles.startScreen}>
        <h2 className={styles.errorTitle}>Something went wrong</h2>
        <p className={styles.errorMsg}>{error}</p>
        <button
          className={styles.primaryButton}
          onClick={() => {
            setAppPhase("start");
            setError(null);
            setSession(null);
            setDisplayedState(null);
          }}
        >
          Back to Start
        </button>
      </div>
    );
  }

  // ── Game board ────────────────────────────────────────────────────────────
  if (!displayedState) return null;

  const state = displayedState;
  const disabled = loading || animating;
  const isTerminal = state.phase === "terminal" && !animating;
  const isHumanTurn = state.current_player === 0 && !disabled;
  const isBidding = state.phase === "bidding" && isHumanTurn;
  const isPlaying = state.phase === "playing" && isHumanTurn;

  let statusText: string;
  if (loading) {
    statusText = "Waiting…";
  } else if (animating && animatingPlayer !== null) {
    statusText = animatingPlayer === 0 ? "You played." : `Agent ${animatingPlayer} is playing…`;
  } else if (isTerminal) {
    statusText = "Round over!";
  } else if (isBidding) {
    statusText = "Your turn to bid.";
  } else if (isPlaying) {
    statusText = "Your turn to play.";
  } else {
    statusText = "Waiting for opponents…";
  }

  return (
    <div className={styles.board}>
      <section className={styles.section}>
        <Scoreboard state={state} />
      </section>

      <section className={styles.section}>
        <TrickArea state={state} />
      </section>

      <section className={styles.section}>
        {isBidding ? (
          <BidControls
            legalActions={state.legal_actions}
            disabled={disabled}
            onBid={handleAction}
          />
        ) : isTerminal ? (
          <div className={styles.terminalBlock}>
            <p className={styles.terminalMsg}>
              {state.scores
                ? `Final scores: ${state.scores
                    .map((s, i) => `${i === 0 ? "You" : `Agent ${i}`}: ${s}`)
                    .join("  |  ")}`
                : "Round complete."}
            </p>
            <button
              className={styles.primaryButton}
              onClick={playAgain}
              disabled={loading}
            >
              {loading ? "Loading…" : "Play Again"}
            </button>
          </div>
        ) : (
          <p className={styles.statusText}>{statusText}</p>
        )}
      </section>

      <section className={styles.section}>
        <Hand
          cards={state.players[0].hand}
          legalActions={state.legal_actions}
          isMyTurn={isPlaying}
          disabled={disabled}
          onPlay={handleAction}
        />
        {session && stats[session.agent] && (() => {
          const s = stats[session.agent];
          const agentLabel = AGENT_OPTIONS.find((o) => o.value === session.agent)?.label ?? session.agent.toUpperCase();
          const userAvg = (s.userScoreSum / s.gamesPlayed).toFixed(2);
          const agentAvg = (s.agentScoreSum / s.gamesPlayed).toFixed(2);
          const bidPct = Math.round((s.userCorrectBids / s.gamesPlayed) * 100);
          return (
            <>
              <div className={styles.statsLine}>
                <span>{`vs ${agentLabel} — ${s.gamesPlayed} ${s.gamesPlayed === 1 ? "game" : "games"}  |  your avg: ${userAvg}  |  opp avg: ${agentAvg}  |  correct bid: ${bidPct}%`}</span>
                <button className={styles.resetBtn} onClick={() => setStats(clearStats())}>Reset</button>
              </div>
              <GameHistory agentLabel={agentLabel} stats={s} />
            </>
          );
        })()}
      </section>
    </div>
  );
}
