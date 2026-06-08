from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Protocol

from .game import BLACK, Move, OthelloState, move_to_coord
from .mcts import UCTSearch
from .neural import ValueNetwork


class Agent(Protocol):
    name: str

    def select_move(self, state: OthelloState) -> Move:
        ...


@dataclass
class RandomAgent:
    seed: int | None = None
    name: str = "random"

    def __post_init__(self) -> None:
        self.rng = Random(self.seed)

    def select_move(self, state: OthelloState) -> Move:
        actions = state.actions()
        return self.rng.choice(actions) if actions else None


@dataclass
class GreedyAgent:
    seed: int | None = None
    name: str = "greedy"

    def __post_init__(self) -> None:
        self.rng = Random(self.seed)

    def select_move(self, state: OthelloState) -> Move:
        actions = state.actions()
        if not actions:
            return None
        scored = []
        for move in actions:
            if move is None:
                scored.append((0, move))
            else:
                scored.append((len(state.captures(*move, state.current_player)), move))
        best_score = max(score for score, _ in scored)
        best = [move for score, move in scored if score == best_score]
        return self.rng.choice(best)


@dataclass
class HeuristicAgent:
    seed: int | None = None
    name: str = "heuristic"

    def __post_init__(self) -> None:
        self.rng = Random(self.seed)

    def select_move(self, state: OthelloState) -> Move:
        actions = state.actions()
        if not actions:
            return None
        scored: list[tuple[float, Move]] = []
        for move in actions:
            child = state.apply_move(move)
            scored.append((-child.heuristic_for(child.current_player), move))
        best_score = max(score for score, _ in scored)
        best = [move for score, move in scored if score == best_score]
        return self.rng.choice(best)


@dataclass
class UCTAgent:
    iterations: int = 500
    exploration: float = 2**0.5
    rollout_limit: int = 128
    seed: int | None = None
    name: str = "uct"

    def __post_init__(self) -> None:
        self.rng = Random(self.seed)

    def select_move(self, state: OthelloState) -> Move:
        search_seed = self.rng.randrange(2**63)
        search = UCTSearch(
            iterations=self.iterations,
            exploration=self.exploration,
            rollout_limit=self.rollout_limit,
            seed=search_seed,
        )
        return search.search(state)


@dataclass
class NeuralUCTAgent:
    network: ValueNetwork
    iterations: int = 500
    exploration: float = 2**0.5
    seed: int | None = None
    name: str = "uct_nn"

    def __post_init__(self) -> None:
        self.rng = Random(self.seed)

    def select_move(self, state: OthelloState) -> Move:
        search_seed = self.rng.randrange(2**63)
        search = UCTSearch(
            iterations=self.iterations,
            exploration=self.exploration,
            evaluator=lambda s: self.network.predict_state(s),
            seed=search_seed,
        )
        return search.search(state)


def create_agent(spec: str, seed: int | None = None) -> Agent:
    parts = spec.split(":")
    kind = parts[0].lower()
    if kind == "random":
        return RandomAgent(seed=seed)
    if kind == "greedy":
        return GreedyAgent(seed=seed)
    if kind == "heuristic":
        return HeuristicAgent(seed=seed)
    if kind == "uct":
        iterations = int(parts[1]) if len(parts) > 1 and parts[1] else 500
        return UCTAgent(iterations=iterations, seed=seed, name=f"uct-{iterations}")
    if kind == "uctnn":
        if len(parts) < 2:
            raise ValueError("Use uctnn:path/to/model.npz[:iterations]")
        network = ValueNetwork.load(parts[1])
        iterations = int(parts[2]) if len(parts) > 2 and parts[2] else 500
        return NeuralUCTAgent(network=network, iterations=iterations, seed=seed, name=f"uctnn-{iterations}")
    raise ValueError(f"Unknown agent spec: {spec}")


def play_game(
    black_agent: Agent,
    white_agent: Agent,
    seed: int | None = None,
    record_states: bool = False,
    max_turns: int = 200,
) -> tuple[OthelloState, list[OthelloState]]:
    state = OthelloState.new_game()
    records: list[OthelloState] = []
    agents = {BLACK: black_agent, -BLACK: white_agent}
    rng = Random(seed)

    for _ in range(max_turns):
        if state.is_terminal():
            break
        agent = agents[state.current_player]
        move = agent.select_move(state)
        if move not in state.actions():
            raise ValueError(f"{agent.name} selected illegal move {move_to_coord(move)}")
        state = state.apply_move(move)
        if record_states and not state.is_terminal():
            records.append(state)
        if seed is not None:
            rng.random()

    return state, records
