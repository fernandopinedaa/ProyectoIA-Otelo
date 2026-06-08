from __future__ import annotations

import argparse
from dataclasses import dataclass

from .agents import create_agent, play_game
from .game import BLACK, WHITE


@dataclass
class TournamentResult:
    wins_a: int = 0
    wins_b: int = 0
    draws: int = 0
    disc_diff_a: int = 0


def run_match(agent_a_spec: str, agent_b_spec: str, games: int, seed: int = 0) -> TournamentResult:
    result = TournamentResult()

    for game_idx in range(games):
        a_is_black = game_idx % 2 == 0
        agent_a = create_agent(agent_a_spec, seed=seed + 10 * game_idx)
        agent_b = create_agent(agent_b_spec, seed=seed + 10 * game_idx + 1)
        black_agent = agent_a if a_is_black else agent_b
        white_agent = agent_b if a_is_black else agent_a

        final_state, _ = play_game(black_agent, white_agent)
        black_score, white_score = final_state.score()
        winner = final_state.winner()

        if winner == 0:
            result.draws += 1
        elif (winner == BLACK and a_is_black) or (winner == WHITE and not a_is_black):
            result.wins_a += 1
        else:
            result.wins_b += 1

        diff_black = black_score - white_score
        result.disc_diff_a += diff_black if a_is_black else -diff_black
        print(
            f"game={game_idx + 1:03d}/{games} "
            f"A={'black' if a_is_black else 'white'} "
            f"score=B{black_score}-W{white_score}"
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an Othello tournament between two agents.")
    parser.add_argument("--agent-a", default="uct:100")
    parser.add_argument("--agent-b", default="greedy")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    result = run_match(args.agent_a, args.agent_b, args.games, args.seed)
    avg_diff = result.disc_diff_a / args.games if args.games else 0.0
    print()
    print("summary")
    print(f"agent_a={args.agent_a}")
    print(f"agent_b={args.agent_b}")
    print(f"wins_a={result.wins_a}")
    print(f"wins_b={result.wins_b}")
    print(f"draws={result.draws}")
    print(f"avg_disc_diff_a={avg_diff:.2f}")


if __name__ == "__main__":
    main()
