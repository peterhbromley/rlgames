const STORAGE_KEY = "oh_hell_stats_v2";

export interface GameRecord {
  hand: number;              // 1-based game number
  userBid: number;
  userTricksWon: number;
  userScore: number;
  agentAvgScore: number;
  userBidCorrect: boolean;
}

export interface AgentStats {
  gamesPlayed: number;
  userScoreSum: number;      // cumulative sum of player 0's score
  agentScoreSum: number;     // cumulative sum of mean(agents 1..N) score per game
  userCorrectBids: number;   // games where player 0 hit their bid exactly
  agentCorrectBidSum: number; // sum of (fraction of agents who hit bid) per game
  history: GameRecord[];     // last 20 games, most recent first
}

export type StatsMap = Record<string, AgentStats>;

export function loadStats(): StatsMap {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as StatsMap) : {};
  } catch {
    return {};
  }
}

function saveStats(stats: StatsMap): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(stats));
}

/** Record a completed game and return the updated stats map. */
export function recordGame(
  agentType: string,
  scores: number[],
  players: { bid: number | null; tricks_won: number }[],
): StatsMap {
  const all = loadStats();
  const prev: AgentStats = all[agentType] ?? {
    gamesPlayed: 0,
    userScoreSum: 0,
    agentScoreSum: 0,
    userCorrectBids: 0,
    agentCorrectBidSum: 0,
    history: [],
  };

  const userScore = scores[0];
  const agentScores = scores.slice(1);
  const agentAvgScore = agentScores.reduce((a, b) => a + b, 0) / agentScores.length;

  const userBid = players[0].bid ?? 0;
  const userTricksWon = players[0].tricks_won;
  const userBidCorrect = players[0].bid !== null && players[0].bid === userTricksWon;

  const agentPlayers = players.slice(1);
  const agentsBidCorrect = agentPlayers.filter(
    (p) => p.bid !== null && p.bid === p.tricks_won
  ).length;
  const agentCorrectFrac = agentPlayers.length > 0 ? agentsBidCorrect / agentPlayers.length : 0;

  const gameNum = prev.gamesPlayed + 1;
  const record: GameRecord = {
    hand: gameNum,
    userBid,
    userTricksWon,
    userScore,
    agentAvgScore,
    userBidCorrect,
  };

  const history = [record, ...prev.history].slice(0, 20);

  all[agentType] = {
    gamesPlayed: gameNum,
    userScoreSum: prev.userScoreSum + userScore,
    agentScoreSum: prev.agentScoreSum + agentAvgScore,
    userCorrectBids: prev.userCorrectBids + (userBidCorrect ? 1 : 0),
    agentCorrectBidSum: prev.agentCorrectBidSum + agentCorrectFrac,
    history,
  };

  saveStats(all);
  return all;
}

export function clearStats(): StatsMap {
  saveStats({});
  return {};
}
