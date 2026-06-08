from __future__ import annotations

from dataclasses import dataclass, field
from math import log, sqrt
from random import Random
from typing import Callable

from .game import Move, OthelloState

Evaluator = Callable[[OthelloState], float]


@dataclass
class Node:
    state: OthelloState
    parent: "Node | None" = None
    move: Move = None
    untried_moves: list[Move] = field(default_factory=list)
    children: list["Node"] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0

    def __post_init__(self) -> None:
        if not self.untried_moves:
            self.untried_moves = list(self.state.actions())

    @property
    def mean_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def is_fully_expanded(self) -> bool:
        return not self.untried_moves


class UCTSearch:
    """Monte Carlo Tree Search with UCT selection for two-player zero-sum games."""

    def __init__(
        self,
        iterations: int = 500,
        exploration: float = sqrt(2.0),
        rollout_limit: int = 128,
        evaluator: Evaluator | None = None,
        seed: int | None = None,
    ) -> None:
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        self.iterations = iterations
        self.exploration = exploration
        self.rollout_limit = rollout_limit
        self.evaluator = evaluator
        self.rng = Random(seed)

    def search(self, state: OthelloState) -> Move:
        actions = state.actions()
        if not actions:
            return None
        if len(actions) == 1:
            return actions[0]

        root = Node(state=state)

        for _ in range(self.iterations):
            node = self._tree_policy(root)
            value = self._evaluate(node.state)
            self._backpropagate(node, value)

        best = max(root.children, key=lambda child: (child.visits, -child.mean_value))
        return best.move

    def _tree_policy(self, node: Node) -> Node:
        while not node.state.is_terminal():
            if node.untried_moves:
                return self._expand(node)
            node = self._best_child(node)
        return node

    def _expand(self, node: Node) -> Node:
        idx = self.rng.randrange(len(node.untried_moves))
        move = node.untried_moves.pop(idx)
        child = Node(state=node.state.apply_move(move), parent=node, move=move)
        node.children.append(child)
        return child

    def _best_child(self, node: Node) -> Node:
        parent_log = log(max(1, node.visits))

        def uct_score(child: Node) -> float:
            if child.visits == 0:
                return float("inf")
            exploitation = -child.mean_value
            exploration = self.exploration * sqrt(parent_log / child.visits)
            return exploitation + exploration

        return max(node.children, key=uct_score)

    def _evaluate(self, state: OthelloState) -> float:
        if state.is_terminal():
            return state.result_for(state.current_player)
        if self.evaluator is not None:
            value = float(self.evaluator(state))
            return max(-1.0, min(1.0, value))
        return self._default_policy(state)

    def _default_policy(self, state: OthelloState) -> float:
        current = state
        depth = 0
        while not current.is_terminal() and depth < self.rollout_limit:
            actions = current.actions()
            move = self.rng.choice(actions)
            current = current.apply_move(move)
            depth += 1
        if current.is_terminal():
            return current.result_for(current.current_player)
        return current.heuristic_for(current.current_player)

    def _backpropagate(self, node: Node, value: float) -> None:
        current: Node | None = node
        propagated = value
        while current is not None:
            current.visits += 1
            current.value_sum += propagated
            propagated = -propagated
            current = current.parent
