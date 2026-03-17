import { useState, useCallback } from "react";
import { createSession, deleteSession, streamAction } from "./api";
import type { GameState, SessionResponse } from "./api";
import Scoreboard from "./components/Scoreboard";
import TrickArea from "./components/TrickArea";
import BidControls from "./components/BidControls";
import Hand from "./components/Hand";
import styles from "./App.module.css";

type AppPhase = "start" | "playing" | "error";

const AGENT_MOVE_DELAY_MS = 700;
const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

export default function App() {
  const [appPhase, setAppPhase] = useState<AppPhase>("start");
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
    try {
      await applySession(await createSession());
      setAppPhase("playing");
    } catch (e) {
      handleError(e);
    }
  }, [applySession]);

  const playAgain = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      if (session) await deleteSession(session.session_id);
      await applySession(await createSession());
    } catch (e) {
      handleError(e);
    }
  }, [session, applySession]);

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
    return (
      <div className={styles.startScreen}>
        <h1 className={styles.title}>Oh Hell</h1>
        <p className={styles.subtitle}>A trick-taking card game</p>
        <button
          className={styles.primaryButton}
          onClick={startGame}
          disabled={loading}
        >
          {loading ? "Loading…" : "New Game"}
        </button>
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
      </section>
    </div>
  );
}
