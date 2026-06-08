from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .neural import ValueNetwork
from .self_play import generate_dataset
from .tournament import TournamentResult, run_match


def result_row(agent_a: str, agent_b: str, games: int, result: TournamentResult, seconds: float) -> dict[str, object]:
    return {
        "agent_a": agent_a,
        "agent_b": agent_b,
        "games": games,
        "wins_a": result.wins_a,
        "wins_b": result.wins_b,
        "draws": result.draws,
        "avg_disc_diff_a": result.disc_diff_a / games if games else 0.0,
        "seconds": seconds,
    }


def write_markdown(path: Path, payload: dict[str, object]) -> None:
    dataset = payload["dataset"]
    training = payload["training"]
    tournaments = payload["tournaments"]

    lines = [
        "# Resultados experimentales",
        "",
        "Resultados generados automaticamente con:",
        "",
        "```bash",
        str(payload["command"]),
        "```",
        "",
        "## Dataset",
        "",
        f"- Partidas de autojuego: {dataset['games']}",
        f"- Agente de autojuego: `{dataset['agent']}`",
        f"- Ejemplos tras simetrias: {dataset['examples']}",
        f"- Fichero: `{dataset['path']}`",
        "",
        "## Entrenamiento",
        "",
        f"- Modelo: `{training['model_path']}`",
        f"- Capas ocultas: `{training['hidden_sizes']}`",
        f"- Epocas: {training['epochs']}",
        f"- Loss entrenamiento final: {training['final_train_loss']:.5f}",
        f"- Loss validacion final: {training['final_val_loss']:.5f}",
        "",
        "## Torneos",
        "",
        "| Agente A | Agente B | Partidas | Victorias A | Victorias B | Empates | Dif. media A | Tiempo (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in tournaments:
        lines.append(
            f"| `{row['agent_a']}` | `{row['agent_b']}` | {row['games']} | "
            f"{row['wins_a']} | {row['wins_b']} | {row['draws']} | "
            f"{row['avg_disc_diff_a']:.2f} | {row['seconds']:.1f} |"
        )

    lines.extend(
        [
            "",
            "Nota: las partidas alternan el color inicial para reducir el sesgo de mover primero.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete Othello experiment suite.")
    parser.add_argument("--selfplay-games", type=int, default=12)
    parser.add_argument("--selfplay-agent", default="uct:12")
    parser.add_argument("--dataset", default="data/processed/selfplay_experiment.npz")
    parser.add_argument("--model", default="models/value_net_experiment.npz")
    parser.add_argument("--hidden", default="64,32")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--tournament-games", type=int, default=8)
    parser.add_argument("--uct-iters", type=int, default=16)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--json-output", default="docs/experiment_results.json")
    parser.add_argument("--md-output", default="docs/experiment_results.md")
    args = parser.parse_args()

    hidden_sizes = tuple(int(part) for part in args.hidden.split(",") if part.strip())

    x, y = generate_dataset(
        games=args.selfplay_games,
        agent_spec=args.selfplay_agent,
        output=args.dataset,
        seed=args.seed,
        augment=True,
    )

    network = ValueNetwork(hidden_sizes=hidden_sizes, seed=args.seed)
    history = network.train(
        x,
        y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        verbose=True,
    )
    network.save(args.model)

    uct = f"uct:{args.uct_iters}"
    uct_stronger = f"uct:{args.uct_iters * 2}"
    uctnn = f"uctnn:{args.model}:{args.uct_iters}"
    matchups = [
        ("greedy", "random"),
        ("heuristic", "greedy"),
        (uct, "greedy"),
        (uct_stronger, uct),
        (uctnn, "random"),
        (uctnn, "greedy"),
        (uctnn, uct),
    ]

    tournaments: list[dict[str, object]] = []
    for agent_a, agent_b in matchups:
        start = perf_counter()
        result = run_match(agent_a, agent_b, games=args.tournament_games, seed=args.seed)
        elapsed = perf_counter() - start
        tournaments.append(result_row(agent_a, agent_b, args.tournament_games, result, elapsed))

    payload: dict[str, object] = {
        "command": (
            "python3 -m othello_ai.experiments "
            f"--selfplay-games {args.selfplay_games} "
            f"--selfplay-agent {args.selfplay_agent} "
            f"--epochs {args.epochs} "
            f"--tournament-games {args.tournament_games} "
            f"--uct-iters {args.uct_iters} "
            f"--seed {args.seed}"
        ),
        "dataset": {
            "games": args.selfplay_games,
            "agent": args.selfplay_agent,
            "examples": int(len(x)),
            "path": args.dataset,
            "label_mean": float(np.mean(y)),
            "label_std": float(np.std(y)),
        },
        "training": {
            "model_path": args.model,
            "hidden_sizes": hidden_sizes,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "final_train_loss": history.train_loss[-1],
            "final_val_loss": history.val_loss[-1],
            "train_loss": history.train_loss,
            "val_loss": history.val_loss,
        },
        "tournaments": tournaments,
    }

    json_path = Path(args.json_output)
    md_path = Path(args.md_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(md_path, payload)

    print(f"saved JSON results to {json_path}")
    print(f"saved Markdown results to {md_path}")


if __name__ == "__main__":
    main()
