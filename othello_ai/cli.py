from __future__ import annotations

import argparse

from .agents import create_agent
from .game import BLACK, WHITE, Move, OthelloState, coord_to_move, move_to_coord, player_name


def ask_human_move(state: OthelloState) -> Move:
    actions = state.actions()
    if actions == [None]:
        print("No legal moves. Passing turn.")
        return None

    legal = {move_to_coord(move) for move in actions}
    while True:
        raw = input(f"Move for {player_name(state.current_player)} {sorted(legal)}: ")
        try:
            move = coord_to_move(raw)
        except ValueError as exc:
            print(exc)
            continue
        if move in actions:
            return move
        print("Illegal move.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Othello against an AI agent.")
    parser.add_argument("--agent", default="uct:300", help="random, greedy, heuristic, uct:N, uctnn:model.npz:N")
    parser.add_argument("--human", choices=["black", "white"], default="black")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    human_player = BLACK if args.human == "black" else WHITE
    ai = create_agent(args.agent, seed=args.seed)
    state = OthelloState.new_game()

    while not state.is_terminal():
        print()
        print(state.render(state.legal_moves()))
        if state.current_player == human_player:
            move = ask_human_move(state)
        else:
            move = ai.select_move(state)
            print(f"{ai.name} plays {move_to_coord(move)}")
        state = state.apply_move(move)

    print()
    print(state.render())
    black, white = state.score()
    winner = state.winner()
    if winner == BLACK:
        print(f"Black wins {black}-{white}.")
    elif winner == WHITE:
        print(f"White wins {white}-{black}.")
    else:
        print(f"Draw {black}-{white}.")


if __name__ == "__main__":
    main()
