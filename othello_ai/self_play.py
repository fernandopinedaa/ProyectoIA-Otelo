from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .agents import create_agent, play_game
from .game import OthelloState


def symmetric_boards(board: np.ndarray) -> list[np.ndarray]:
    boards: list[np.ndarray] = []
    for k in range(4):
        rotated = np.rot90(board, k)
        boards.append(rotated)
        boards.append(np.fliplr(rotated))
    return boards


def state_features(board: np.ndarray, player: int) -> np.ndarray:
    state = OthelloState(board.astype(np.int8, copy=True), player)
    return state.features()


def generate_dataset(
    games: int,
    agent_spec: str,
    output: str | Path,
    seed: int = 0,
    augment: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[float] = []

    for game_idx in range(games):
        black = create_agent(agent_spec, seed=seed + 2 * game_idx)
        white = create_agent(agent_spec, seed=seed + 2 * game_idx + 1)
        final_state, records = play_game(black, white, seed=seed + game_idx, record_states=True)

        for state in records:
            label = final_state.result_for(state.current_player)
            boards = symmetric_boards(state.board) if augment else [state.board]
            for board in boards:
                features.append(state_features(board, state.current_player))
                labels.append(label)

        black_score, white_score = final_state.score()
        print(
            f"game={game_idx + 1:04d}/{games} "
            f"states={len(records):03d} score=B{black_score}-W{white_score}"
        )

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        x=x,
        y=y,
        games=np.array([games], dtype=np.int32),
        agent=np.array([agent_spec]),
        augmented=np.array([int(augment)], dtype=np.int8),
    )
    print(f"saved {len(x)} examples to {output}")
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Othello value data through self-play.")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--agent", default="uct:80", help="Agent used by both players.")
    parser.add_argument("--output", default="data/processed/selfplay.npz")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    generate_dataset(
        games=args.games,
        agent_spec=args.agent,
        output=args.output,
        seed=args.seed,
        augment=not args.no_augment,
    )


if __name__ == "__main__":
    main()
