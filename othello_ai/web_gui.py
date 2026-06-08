from __future__ import annotations

import argparse
import errno
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .agents import Agent, create_agent
from .game import BLACK, WHITE, Move, OthelloState, move_to_coord, player_name


def available_agent_options(current_spec: str | None = None) -> list[dict[str, str]]:
    options = [
        {"value": "random", "label": "Aleatorio"},
        {"value": "greedy", "label": "Voraz"},
        {"value": "heuristic", "label": "Heurístico"},
        {"value": "uct:100", "label": "UCT 100"},
        {"value": "uct:300", "label": "UCT 300"},
    ]
    if Path("models/value_net_experiment.npz").exists():
        options.append({"value": "uctnn:models/value_net_experiment.npz:100", "label": "UCT con red neuronal"})

    if current_spec and current_spec not in {option["value"] for option in options}:
        options.append({"value": current_spec, "label": current_spec})
    return options


HTML = r"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Otelo UCT</title>
  <style>
    :root {
      --bg: #17201b;
      --panel: #f4f0e7;
      --board: #236b43;
      --board-dark: #1d5637;
      --line: #123323;
      --black: #111111;
      --white: #f6f2e9;
      --accent: #d9b44a;
      --text: #162019;
      --muted: #5f6b61;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background: var(--bg);
      color: var(--text);
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    main {
      width: min(1120px, 100vw);
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(320px, 680px) minmax(260px, 360px);
      gap: 24px;
      align-items: center;
      padding: 28px;
    }

    .board-wrap {
      width: min(680px, calc(100vw - 56px));
      aspect-ratio: 1;
      display: grid;
      grid-template-columns: repeat(8, 1fr);
      grid-template-rows: repeat(8, 1fr);
      border: 3px solid var(--line);
      background: var(--board);
      box-shadow: 0 18px 45px rgba(0, 0, 0, 0.35);
    }

    .cell {
      position: relative;
      border: 1px solid rgba(18, 51, 35, 0.85);
      background: var(--board);
      display: grid;
      place-items: center;
      cursor: default;
      user-select: none;
    }

    .cell:nth-child(2n) { background: var(--board-dark); }
    .cell.legal { cursor: pointer; }
    .cell.legal::after {
      content: "";
      width: 28%;
      height: 28%;
      border-radius: 999px;
      background: rgba(217, 180, 74, 0.8);
      box-shadow: 0 0 0 5px rgba(217, 180, 74, 0.18);
    }
    .cell.legal:hover::after {
      width: 36%;
      height: 36%;
      background: var(--accent);
    }

    .disc {
      width: 72%;
      height: 72%;
      border-radius: 999px;
      box-shadow: inset 0 4px 7px rgba(255,255,255,.22),
                  inset 0 -7px 12px rgba(0,0,0,.35),
                  0 5px 12px rgba(0,0,0,.35);
    }

    .disc.black { background: var(--black); }
    .disc.white {
      background: var(--white);
      box-shadow: inset 0 5px 8px rgba(255,255,255,.75),
                  inset 0 -6px 10px rgba(0,0,0,.15),
                  0 5px 12px rgba(0,0,0,.32);
    }

    aside {
      background: var(--panel);
      border-radius: 8px;
      padding: 22px;
      box-shadow: 0 18px 45px rgba(0, 0, 0, 0.28);
    }

    h1 {
      margin: 0 0 14px;
      font-size: 26px;
      line-height: 1.15;
      letter-spacing: 0;
    }

    .status {
      min-height: 58px;
      margin: 10px 0 18px;
      color: var(--muted);
      line-height: 1.35;
    }

    .score {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin: 18px 0;
    }

    .score div {
      border: 1px solid rgba(22, 32, 25, 0.16);
      border-radius: 8px;
      padding: 12px;
      background: #fffaf0;
    }

    .score strong {
      display: block;
      font-size: 26px;
      line-height: 1;
      margin-top: 6px;
    }

    aside button {
      width: 100%;
      border: 0;
      border-radius: 8px;
      padding: 12px 14px;
      margin-top: 10px;
      background: #1f6d45;
      color: white;
      font: inherit;
      font-weight: 650;
      cursor: pointer;
    }

    aside button.secondary { background: #59665c; }
    aside button:disabled {
      opacity: 0.45;
      cursor: not-allowed;
    }

    label {
      display: block;
      margin-top: 12px;
      font-size: 13px;
      font-weight: 700;
      color: var(--muted);
    }

    select {
      width: 100%;
      min-height: 42px;
      margin-top: 6px;
      border: 1px solid rgba(22, 32, 25, 0.2);
      border-radius: 8px;
      padding: 9px 10px;
      background: #fffaf0;
      color: var(--text);
      font: inherit;
    }

    .meta {
      margin-top: 18px;
      font-size: 13px;
      color: var(--muted);
      line-height: 1.5;
    }

    .moves {
      margin-top: 14px;
      max-height: 130px;
      overflow: auto;
      font-size: 13px;
      color: var(--muted);
      border-top: 1px solid rgba(22, 32, 25, 0.12);
      padding-top: 10px;
    }

    .modal[hidden] { display: none; }
    .modal {
      position: fixed;
      inset: 0;
      display: grid;
      place-items: center;
      padding: 24px;
      background: rgba(0, 0, 0, 0.46);
      z-index: 10;
    }

    .dialog {
      width: min(360px, 100%);
      background: var(--panel);
      border-radius: 8px;
      padding: 22px;
      box-shadow: 0 18px 45px rgba(0, 0, 0, 0.35);
    }

    .dialog h2 {
      margin: 0 0 10px;
      font-size: 24px;
      line-height: 1.15;
      letter-spacing: 0;
    }

    .dialog p {
      margin: 0 0 14px;
      color: var(--muted);
      line-height: 1.4;
    }

    .dialog button {
      width: 100%;
      border: 0;
      border-radius: 8px;
      padding: 12px 14px;
      margin-top: 10px;
      background: #1f6d45;
      color: white;
      font: inherit;
      font-weight: 650;
      cursor: pointer;
    }

    .dialog button.secondary { background: #59665c; }

    @media (max-width: 880px) {
      main {
        grid-template-columns: 1fr;
        align-content: start;
      }
      .board-wrap { margin: 0 auto; }
    }
  </style>
</head>
<body>
  <main>
    <section class="board-wrap" id="board" aria-label="Tablero de Otelo"></section>
    <aside>
      <h1>Otelo UCT</h1>
      <div class="status" id="status">Cargando partida...</div>
      <div class="score">
        <div>Negras<strong id="black-score">2</strong></div>
        <div>Blancas<strong id="white-score">2</strong></div>
      </div>
      <label for="agent-select">Rival</label>
      <select id="agent-select"></select>
      <label for="human-select">Color</label>
      <select id="human-select">
        <option value="black">Negras</option>
        <option value="white">Blancas</option>
      </select>
      <button id="pass-button" disabled>Pasar turno</button>
      <button class="secondary" id="new-button">Nueva partida</button>
      <div class="meta" id="meta"></div>
      <div class="moves" id="moves"></div>
    </aside>
  </main>

  <div class="modal" id="result-modal" hidden>
    <div class="dialog" role="dialog" aria-modal="true" aria-labelledby="result-title">
      <h2 id="result-title">Partida terminada</h2>
      <p id="result-message"></p>
      <button id="modal-new-button">Jugar otra partida</button>
      <button class="secondary" id="modal-close-button">Cerrar</button>
    </div>
  </div>

  <script>
    const boardEl = document.getElementById("board");
    const statusEl = document.getElementById("status");
    const blackScoreEl = document.getElementById("black-score");
    const whiteScoreEl = document.getElementById("white-score");
    const passButton = document.getElementById("pass-button");
    const newButton = document.getElementById("new-button");
    const agentSelect = document.getElementById("agent-select");
    const humanSelect = document.getElementById("human-select");
    const metaEl = document.getElementById("meta");
    const movesEl = document.getElementById("moves");
    const resultModal = document.getElementById("result-modal");
    const resultTitle = document.getElementById("result-title");
    const resultMessage = document.getElementById("result-message");
    const modalNewButton = document.getElementById("modal-new-button");
    const modalCloseButton = document.getElementById("modal-close-button");
    let currentState = null;
    let busy = false;
    let shownResultKey = null;

    function coord(row, col) {
      return String.fromCharCode("a".charCodeAt(0) + col) + (row + 1);
    }

    function sleep(ms) {
      return new Promise(resolve => setTimeout(resolve, ms));
    }

    async function api(path, body = null) {
      const options = body ? {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body)
      } : {};
      const response = await fetch(path, options);
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || response.statusText);
      }
      return await response.json();
    }

    function isLegal(row, col) {
      return currentState.legal_moves.some(m => m.row === row && m.col === col);
    }

    function fillAgentSelect(state) {
      const selected = agentSelect.value || state.agent_spec;
      agentSelect.innerHTML = "";
      for (const option of state.agent_options) {
        const item = document.createElement("option");
        item.value = option.value;
        item.textContent = option.label;
        agentSelect.appendChild(item);
      }
      agentSelect.value = state.agent_options.some(option => option.value === selected)
        ? selected
        : state.agent_spec;
    }

    function render(state) {
      currentState = state;
      boardEl.innerHTML = "";
      blackScoreEl.textContent = state.score.black;
      whiteScoreEl.textContent = state.score.white;
      statusEl.textContent = state.status;
      fillAgentSelect(state);
      humanSelect.value = state.human_player;
      passButton.disabled = !state.can_pass || busy;
      newButton.disabled = busy;
      agentSelect.disabled = busy;
      humanSelect.disabled = busy;
      metaEl.textContent = `Turno: ${state.current_player} | Rival: ${state.agent}`;
      movesEl.innerHTML = state.history.slice(-8).map(item => `<div>${item}</div>`).join("");

      for (let row = 0; row < 8; row++) {
        for (let col = 0; col < 8; col++) {
          const cell = document.createElement("button");
          cell.className = "cell";
          cell.setAttribute("aria-label", coord(row, col));
          const value = state.board[row][col];
          if (value === 1 || value === -1) {
            const disc = document.createElement("div");
            disc.className = "disc " + (value === 1 ? "black" : "white");
            cell.appendChild(disc);
          } else if (isLegal(row, col) && !busy) {
            cell.classList.add("legal");
            cell.addEventListener("click", () => play(row, col));
          }
          boardEl.appendChild(cell);
        }
      }

      if (state.game_over) {
        showResult(state);
      }
    }

    function showResult(state) {
      const key = `${state.score.black}-${state.score.white}-${state.winner}`;
      if (shownResultKey === key) return;
      shownResultKey = key;
      if (state.winner === "black") {
        resultTitle.textContent = "Ganan las negras";
      } else if (state.winner === "white") {
        resultTitle.textContent = "Ganan las blancas";
      } else {
        resultTitle.textContent = "Empate";
      }
      resultMessage.textContent = `Resultado final: negras ${state.score.black}, blancas ${state.score.white}.`;
      resultModal.hidden = false;
    }

    async function refresh() {
      render(await api("/api/state"));
      await advanceAIIfNeeded();
    }

    async function advanceAIIfNeeded() {
      if (!currentState || currentState.game_over || !currentState.needs_ai_move || busy) return;
      busy = true;
      render({...currentState, status: "La máquina está pensando..."});
      await sleep(1000);
      try {
        const state = await api("/api/ai", {});
        busy = false;
        render(state);
        await advanceAIIfNeeded();
      } catch (error) {
        busy = false;
        alert(error.message);
        await refresh();
      }
    }

    async function play(row, col) {
      if (busy || !currentState || currentState.game_over) return;
      busy = true;
      statusEl.textContent = "Aplicando jugada...";
      passButton.disabled = true;
      try {
        const state = await api("/api/move", {row, col});
        busy = false;
        render(state);
        await advanceAIIfNeeded();
      } catch (error) {
        busy = false;
        alert(error.message);
        await refresh();
      }
    }

    passButton.addEventListener("click", async () => {
      if (busy) return;
      busy = true;
      statusEl.textContent = "Pasando turno...";
      try {
        const state = await api("/api/pass", {});
        busy = false;
        render(state);
        await advanceAIIfNeeded();
      } catch (error) {
        busy = false;
        alert(error.message);
        await refresh();
      }
    });

    async function newGame() {
      busy = true;
      shownResultKey = null;
      resultModal.hidden = true;
      statusEl.textContent = "Preparando partida...";
      try {
        const state = await api("/api/new", {
          agent: agentSelect.value,
          human: humanSelect.value
        });
        busy = false;
        render(state);
        await advanceAIIfNeeded();
      } catch (error) {
        busy = false;
        alert(error.message);
        await refresh();
      }
    }

    newButton.addEventListener("click", newGame);
    modalNewButton.addEventListener("click", newGame);
    modalCloseButton.addEventListener("click", () => {
      resultModal.hidden = true;
    });

    refresh();
  </script>
</body>
</html>
"""


class WebGame:
    def __init__(self, agent_spec: str, human_player: int, seed: int | None) -> None:
        self.agent_spec = agent_spec
        self.human_player = human_player
        self.seed = seed
        self.lock = threading.Lock()
        self.agent: Agent
        self.state: OthelloState
        self.history: list[str]
        self.reset()

    def reset(self, agent_spec: str | None = None, human_player: int | None = None) -> dict[str, Any]:
        with self.lock:
            if agent_spec is not None:
                self.agent_spec = agent_spec
            if human_player is not None:
                self.human_player = human_player
            self.agent = create_agent(self.agent_spec, seed=self.seed)
            self.state = OthelloState.new_game()
            self.history = []
            return self._snapshot_locked()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return self._snapshot_locked()

    def human_move(self, move: Move) -> dict[str, Any]:
        with self.lock:
            if self.state.is_terminal():
                raise ValueError("The game is already finished.")
            if self.state.current_player != self.human_player:
                raise ValueError("It is not the human player's turn.")
            if move not in self.state.actions():
                raise ValueError("Illegal move.")

            self.history.append(f"Humano juega {move_to_coord(move)}")
            self.state = self.state.apply_move(move)
            return self._snapshot_locked()

    def ai_move(self) -> dict[str, Any]:
        with self.lock:
            if self.state.is_terminal():
                return self._snapshot_locked()
            if self.state.current_player == self.human_player:
                raise ValueError("It is not the AI player's turn.")

            while not self.state.is_terminal() and self.state.current_player != self.human_player:
                move = self.agent.select_move(self.state)
                if move not in self.state.actions():
                    raise ValueError(f"{self.agent.name} selected illegal move {move_to_coord(move)}")
                self.history.append(f"{self.agent.name} juega {move_to_coord(move)}")
                self.state = self.state.apply_move(move)

            return self._snapshot_locked()

    def _snapshot_locked(self) -> dict[str, Any]:
        black, white = self.state.score()
        legal = self.state.legal_moves() if self.state.current_player == self.human_player else []
        winner = self.state.winner() if self.state.is_terminal() else None
        can_pass = (
            not self.state.is_terminal()
            and self.state.current_player == self.human_player
            and self.state.actions() == [None]
        )
        return {
            "board": self.state.board.astype(int).tolist(),
            "score": {"black": black, "white": white},
            "current_player": player_name(self.state.current_player),
            "human_player": player_name(self.human_player),
            "agent": self.agent.name,
            "agent_spec": self.agent_spec,
            "agent_options": available_agent_options(self.agent_spec),
            "legal_moves": [{"row": row, "col": col, "coord": move_to_coord((row, col))} for row, col in legal],
            "can_pass": can_pass,
            "needs_ai_move": not self.state.is_terminal() and self.state.current_player != self.human_player,
            "game_over": self.state.is_terminal(),
            "winner": player_name(winner) if winner in (BLACK, WHITE) else ("draw" if winner == 0 else None),
            "status": self._status_locked(winner),
            "history": self.history,
        }

    def _status_locked(self, winner: int | None) -> str:
        if self.state.is_terminal():
            black, white = self.state.score()
            if winner == BLACK:
                return f"Game over. Black wins {black}-{white}."
            if winner == WHITE:
                return f"Game over. White wins {white}-{black}."
            return f"Game over. Draw {black}-{white}."
        if self.state.current_player == self.human_player:
            if self.state.actions() == [None]:
                return "No tienes movimientos legales. Pasa turno."
            return "Tu turno."
        return "La máquina está pensando..."


class OthelloRequestHandler(BaseHTTPRequestHandler):
    game: WebGame

    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            self._send_html(HTML)
        elif self.path == "/api/state":
            self._send_json(self.game.snapshot())
        elif self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        try:
            if self.path == "/api/new":
                body = self._read_json()
                human = body.get("human")
                human_player = None
                if human == "black":
                    human_player = BLACK
                elif human == "white":
                    human_player = WHITE
                payload = self.game.reset(body.get("agent"), human_player)
            elif self.path == "/api/pass":
                payload = self.game.human_move(None)
            elif self.path == "/api/move":
                body = self._read_json()
                payload = self.game.human_move((int(body["row"]), int(body["col"])))
            elif self.path == "/api/ai":
                payload = self.game.ai_move()
            else:
                self.send_error(404)
                return
            self._send_json(payload)
        except Exception as exc:
            self._send_text(str(exc), status=400)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw) if raw else {}

    def _send_html(self, html: str) -> None:
        encoded = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_text(self, text: str, status: int = 200) -> None:
        encoded = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def create_server(preferred_port: int, handler: type[BaseHTTPRequestHandler]) -> ThreadingHTTPServer:
    if preferred_port == 0:
        return ThreadingHTTPServer(("127.0.0.1", 0), handler)

    for port in range(preferred_port, preferred_port + 50):
        try:
            return ThreadingHTTPServer(("127.0.0.1", port), handler)
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
    raise OSError(f"No free local port found from {preferred_port} to {preferred_port + 49}.")


def server_port(server: ThreadingHTTPServer) -> int:
    return int(server.server_address[1])


def find_port(preferred: int) -> int:
    if preferred == 0:
        server = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
        try:
            return server_port(server)
        finally:
            server.server_close()
    return preferred


def run_gui(agent_spec: str = "uct:300", human: str = "black", seed: int | None = None, port: int = 8765, open_browser: bool = True) -> None:
    human_player = BLACK if human == "black" else WHITE
    game = WebGame(agent_spec=agent_spec, human_player=human_player, seed=seed)
    actual_port = find_port(port)

    class Handler(OthelloRequestHandler):
        pass

    Handler.game = game
    server = create_server(actual_port, Handler)
    actual_port = server_port(server)
    url = f"http://127.0.0.1:{actual_port}/"
    if actual_port != port and port != 0:
        print(f"El puerto {port} estaba ocupado; usando {actual_port}.")
    print(f"Interfaz gráfica de Otelo: {url}")
    print("Pulsa Ctrl+C para parar el servidor.")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the graphical Othello interface.")
    parser.add_argument("--agent", default="uct:300", help="random, greedy, heuristic, uct:N, uctnn:model.npz:N")
    parser.add_argument("--human", choices=["black", "white"], default="black")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    run_gui(args.agent, args.human, args.seed, args.port, not args.no_browser)


if __name__ == "__main__":
    main()
