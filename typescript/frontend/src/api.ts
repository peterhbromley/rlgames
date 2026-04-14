const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export interface Card {
  id: number;
  rank: string;
  suit: string;
  label: string;
}

export interface LegalAction {
  id: number;
  label: string;
  type: "bid" | "play";
}

export interface PlayerState {
  bid: number | null;
  tricks_won: number;
  hand: Card[];
}

export interface TrickEntry {
  player: number;
  card: Card;
}

export interface GameState {
  phase: "bidding" | "playing" | "terminal";
  current_player: number;
  num_players: number;
  num_tricks: number;
  trump_card: Card | null;
  players: PlayerState[];
  current_trick: TrickEntry[];
  scores: number[] | null;
  legal_actions: LegalAction[];
}

export interface Transition {
  player: number;
  state: GameState;
}

export interface SessionResponse {
  session_id: string;
  game: string;
  agent: string;
  state: GameState;
  transitions: Transition[];
}

export interface AgentInfo {
  game: string;
  agent: string;
}

export async function getAgents(): Promise<AgentInfo[]> {
  const res = await fetch(`${API_BASE}/agents`);
  if (!res.ok) throw new Error(`Failed to get agents: ${res.status}`);
  return res.json();
}

export async function createSession(
  agent: string = "nfsp"
): Promise<SessionResponse> {
  const res = await fetch(`${API_BASE}/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ game: "oh_hell", agent, human_players: [0] }),
  });
  if (!res.ok) throw new Error(`Failed to create session: ${res.status}`);
  return res.json();
}

export async function getSession(id: string): Promise<SessionResponse> {
  const res = await fetch(`${API_BASE}/sessions/${id}`);
  if (!res.ok) throw new Error(`Failed to get session: ${res.status}`);
  return res.json();
}

export type StreamEvent =
  | { type: "transition"; player: number; state: GameState }
  | { type: "final"; player: null; state: GameState };

export async function* streamAction(
  id: string,
  actionId: number
): AsyncGenerator<StreamEvent> {
  const res = await fetch(`${API_BASE}/sessions/${id}/actions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action_id: actionId }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status}: ${text}`);
  }

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE events are separated by double newlines.
      const parts = buffer.split("\n\n");
      buffer = parts.pop()!; // last element is the incomplete chunk

      for (const part of parts) {
        const line = part.split("\n").find((l) => l.startsWith("data: "));
        if (line) yield JSON.parse(line.slice(6)) as StreamEvent;
      }
    }
  } finally {
    reader.cancel();
  }
}

export async function deleteSession(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/sessions/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`Failed to delete session: ${res.status}`);
}
