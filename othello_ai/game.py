from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

EMPTY = 0
BLACK = 1
WHITE = -1
PASS = None

Move = tuple[int, int] | None

BOARD_SIZE = 8
DIRECTIONS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def opponent(player: int) -> int:
    if player not in (BLACK, WHITE):
        raise ValueError(f"Invalid player: {player}")
    return -player


def player_name(player: int) -> str:
    return "black" if player == BLACK else "white"


def move_to_coord(move: Move) -> str:
    if move is PASS:
        return "pass"
    row, col = move
    return f"{chr(ord('a') + col)}{row + 1}"


def coord_to_move(text: str) -> Move:
    value = text.strip().lower()
    if value in {"pass", "p"}:
        return PASS
    if len(value) == 2 and value[0].isalpha() and value[1].isdigit():
        col = ord(value[0]) - ord("a")
        row = int(value[1]) - 1
        return row, col
    parts = value.replace(",", " ").split()
    if len(parts) == 2 and all(part.lstrip("-").isdigit() for part in parts):
        row, col = int(parts[0]), int(parts[1])
        if row >= 1 and col >= 1:
            row -= 1
            col -= 1
        return row, col
    raise ValueError("Use coordinates like d3, or row col.")


def initial_board() -> np.ndarray:
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    board[3, 3] = WHITE
    board[3, 4] = BLACK
    board[4, 3] = BLACK
    board[4, 4] = WHITE
    return board


@dataclass(frozen=True)
class OthelloState:
    board: np.ndarray
    current_player: int = BLACK
    consecutive_passes: int = 0

    @classmethod
    def new_game(cls) -> "OthelloState":
        return cls(initial_board(), BLACK, 0)

    def __post_init__(self) -> None:
        if self.board.shape != (BOARD_SIZE, BOARD_SIZE):
            raise ValueError("The board must have shape 8x8.")
        if self.current_player not in (BLACK, WHITE):
            raise ValueError("The current player must be BLACK or WHITE.")
        self.board.setflags(write=False)

    def copy_mutable_board(self) -> np.ndarray:
        return np.array(self.board, copy=True)

    def inside(self, row: int, col: int) -> bool:
        return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE

    def legal_moves(self, player: int | None = None) -> list[tuple[int, int]]:
        player = self.current_player if player is None else player
        moves: list[tuple[int, int]] = []
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self.board[row, col] == EMPTY and self.captures(row, col, player):
                    moves.append((row, col))
        return moves

    def actions(self) -> list[Move]:
        if self.is_terminal():
            return []
        moves = self.legal_moves(self.current_player)
        return moves if moves else [PASS]

    def captures(self, row: int, col: int, player: int | None = None) -> list[tuple[int, int]]:
        player = self.current_player if player is None else player
        if not self.inside(row, col) or self.board[row, col] != EMPTY:
            return []

        captured: list[tuple[int, int]] = []
        rival = opponent(player)

        for dr, dc in DIRECTIONS:
            path: list[tuple[int, int]] = []
            r, c = row + dr, col + dc
            while self.inside(r, c) and self.board[r, c] == rival:
                path.append((r, c))
                r += dr
                c += dc
            if path and self.inside(r, c) and self.board[r, c] == player:
                captured.extend(path)

        return captured

    def apply_move(self, move: Move) -> "OthelloState":
        legal = self.legal_moves(self.current_player)
        if move is PASS:
            if legal:
                raise ValueError("Passing is only legal when no moves are available.")
            return OthelloState(self.board.copy(), opponent(self.current_player), self.consecutive_passes + 1)

        row, col = move
        captured = self.captures(row, col, self.current_player)
        if not captured:
            raise ValueError(f"Illegal move: {move_to_coord(move)}")

        board = self.copy_mutable_board()
        board[row, col] = self.current_player
        for r, c in captured:
            board[r, c] = self.current_player
        return OthelloState(board, opponent(self.current_player), 0)

    def is_full(self) -> bool:
        return not np.any(self.board == EMPTY)

    def is_terminal(self) -> bool:
        if self.is_full() or self.consecutive_passes >= 2:
            return True
        if self.legal_moves(BLACK) or self.legal_moves(WHITE):
            return False
        return True

    def score(self) -> tuple[int, int]:
        black = int(np.sum(self.board == BLACK))
        white = int(np.sum(self.board == WHITE))
        return black, white

    def winner(self) -> int:
        black, white = self.score()
        if black > white:
            return BLACK
        if white > black:
            return WHITE
        return EMPTY

    def result_for(self, player: int) -> float:
        winner = self.winner()
        if winner == EMPTY:
            return 0.0
        return 1.0 if winner == player else -1.0

    def disc_difference_for(self, player: int) -> float:
        own = int(np.sum(self.board == player))
        rival = int(np.sum(self.board == opponent(player)))
        total = own + rival
        if total == 0:
            return 0.0
        return (own - rival) / total

    def mobility_for(self, player: int) -> float:
        own = len(self.legal_moves(player))
        rival = len(self.legal_moves(opponent(player)))
        total = own + rival
        if total == 0:
            return 0.0
        return (own - rival) / total

    def corner_difference_for(self, player: int) -> float:
        corners = ((0, 0), (0, 7), (7, 0), (7, 7))
        own = sum(1 for r, c in corners if self.board[r, c] == player)
        rival = sum(1 for r, c in corners if self.board[r, c] == opponent(player))
        total = own + rival
        if total == 0:
            return 0.0
        return (own - rival) / total

    def heuristic_for(self, player: int) -> float:
        return float(
            0.45 * self.disc_difference_for(player)
            + 0.35 * self.mobility_for(player)
            + 0.20 * self.corner_difference_for(player)
        )

    def features(self, player: int | None = None) -> np.ndarray:
        player = self.current_player if player is None else player
        own = (self.board == player).astype(np.float32)
        rival = (self.board == opponent(player)).astype(np.float32)
        return np.concatenate([own.ravel(), rival.ravel()])

    def matrix_012(self) -> np.ndarray:
        matrix = np.zeros_like(self.board, dtype=np.int8)
        matrix[self.board == WHITE] = 1
        matrix[self.board == BLACK] = 2
        return matrix

    def key(self) -> tuple[bytes, int, int]:
        return self.board.tobytes(), self.current_player, self.consecutive_passes

    def render(self, legal_moves: Iterable[tuple[int, int]] | None = None) -> str:
        legal = set(legal_moves or [])
        black, white = self.score()
        rows = ["  a b c d e f g h"]
        for row in range(BOARD_SIZE):
            cells: list[str] = []
            for col in range(BOARD_SIZE):
                value = self.board[row, col]
                if value == BLACK:
                    cells.append("B")
                elif value == WHITE:
                    cells.append("W")
                elif (row, col) in legal:
                    cells.append("*")
                else:
                    cells.append(".")
            rows.append(f"{row + 1} " + " ".join(cells))
        rows.append(f"Turn: {player_name(self.current_player)} | B={black} W={white}")
        return "\n".join(rows)
