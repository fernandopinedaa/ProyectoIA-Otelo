from __future__ import annotations

import argparse

import numpy as np

from .neural import ValueNetwork


def parse_hidden(text: str) -> tuple[int, ...]:
    values = tuple(int(part) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("At least one hidden layer is required.")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a value network for Othello.")
    parser.add_argument("--dataset", default="data/processed/selfplay.npz")
    parser.add_argument("--output", default="models/value_net.npz")
    parser.add_argument("--hidden", default="128,64")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dataset = np.load(args.dataset)
    x = dataset["x"]
    y = dataset["y"]

    network = ValueNetwork(hidden_sizes=parse_hidden(args.hidden), seed=args.seed)
    network.train(
        x,
        y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        verbose=True,
    )
    network.save(args.output)
    print(f"saved model to {args.output}")


if __name__ == "__main__":
    main()
