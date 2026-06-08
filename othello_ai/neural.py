from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .game import OthelloState


def _xavier(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float32)


@dataclass
class TrainingHistory:
    train_loss: list[float]
    val_loss: list[float]


class ValueNetwork:
    """Small fully connected value network trained with NumPy backpropagation."""

    def __init__(
        self,
        input_size: int = 128,
        hidden_sizes: Iterable[int] = (128, 64),
        seed: int | None = None,
    ) -> None:
        self.input_size = input_size
        self.hidden_sizes = tuple(hidden_sizes)
        rng = np.random.default_rng(seed)
        layer_sizes = (input_size, *self.hidden_sizes, 1)
        self.weights = [_xavier(rng, layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]
        self.biases = [np.zeros((1, layer_sizes[i + 1]), dtype=np.float32) for i in range(len(layer_sizes) - 1)]

    def forward(self, x: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        activations = [x.astype(np.float32)]
        pre_activations: list[np.ndarray] = []
        current = activations[0]
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = current @ w + b
            pre_activations.append(z)
            current = np.maximum(z, 0.0)
            activations.append(current)
        z = current @ self.weights[-1] + self.biases[-1]
        pre_activations.append(z)
        output = np.tanh(z)
        activations.append(output)
        return activations, pre_activations

    def predict(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        activations, _ = self.forward(x.astype(np.float32))
        return activations[-1].reshape(-1)

    def predict_state(self, state: OthelloState) -> float:
        return float(self.predict(state.features())[0])

    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 30,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        validation_split: float = 0.15,
        seed: int | None = None,
        verbose: bool = True,
    ) -> TrainingHistory:
        x = x.astype(np.float32)
        y = y.astype(np.float32).reshape(-1, 1)
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")
        if len(x) == 0:
            raise ValueError("empty dataset")

        rng = np.random.default_rng(seed)
        order = rng.permutation(len(x))
        x = x[order]
        y = y[order]
        split_at = int(len(x) * (1.0 - validation_split))
        split_at = min(max(split_at, 1), len(x))
        x_train, y_train = x[:split_at], y[:split_at]
        x_val, y_val = x[split_at:], y[split_at:]

        history = TrainingHistory(train_loss=[], val_loss=[])

        for epoch in range(1, epochs + 1):
            permutation = rng.permutation(len(x_train))
            for start in range(0, len(x_train), batch_size):
                idx = permutation[start : start + batch_size]
                self._train_batch(x_train[idx], y_train[idx], learning_rate)

            train_loss = self.loss(x_train, y_train)
            val_loss = self.loss(x_val, y_val) if len(x_val) else train_loss
            history.train_loss.append(train_loss)
            history.val_loss.append(val_loss)
            if verbose:
                print(f"epoch={epoch:03d} train_loss={train_loss:.5f} val_loss={val_loss:.5f}")

        return history

    def _train_batch(self, x: np.ndarray, y: np.ndarray, learning_rate: float) -> None:
        activations, pre_activations = self.forward(x)
        predictions = activations[-1]
        batch_size = max(1, len(x))

        delta = (2.0 * (predictions - y) / batch_size) * (1.0 - predictions**2)
        grad_w: list[np.ndarray] = []
        grad_b: list[np.ndarray] = []

        for layer in reversed(range(len(self.weights))):
            a_prev = activations[layer]
            grad_w.insert(0, a_prev.T @ delta)
            grad_b.insert(0, np.sum(delta, axis=0, keepdims=True))
            if layer > 0:
                delta = delta @ self.weights[layer].T
                delta = delta * (pre_activations[layer - 1] > 0.0)

        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * grad_w[i].astype(np.float32)
            self.biases[i] -= learning_rate * grad_b[i].astype(np.float32)

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        if len(x) == 0:
            return 0.0
        pred = self.predict(x).reshape(-1, 1)
        return float(np.mean((pred - y.reshape(-1, 1)) ** 2))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, np.ndarray] = {
            "input_size": np.array([self.input_size], dtype=np.int32),
            "hidden_sizes": np.array(self.hidden_sizes, dtype=np.int32),
        }
        for i, weight in enumerate(self.weights):
            data[f"w{i}"] = weight
        for i, bias in enumerate(self.biases):
            data[f"b{i}"] = bias
        np.savez_compressed(path, **data)

    @classmethod
    def load(cls, path: str | Path) -> "ValueNetwork":
        data = np.load(path)
        input_size = int(data["input_size"][0])
        hidden_sizes = tuple(int(v) for v in data["hidden_sizes"])
        network = cls(input_size=input_size, hidden_sizes=hidden_sizes)
        network.weights = [data[f"w{i}"].astype(np.float32) for i in range(len(hidden_sizes) + 1)]
        network.biases = [data[f"b{i}"].astype(np.float32) for i in range(len(hidden_sizes) + 1)]
        return network
