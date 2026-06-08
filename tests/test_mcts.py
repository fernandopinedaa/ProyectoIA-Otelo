import unittest

import numpy as np

from othello_ai.agents import RandomAgent, UCTAgent, play_game
from othello_ai.game import OthelloState
from othello_ai.neural import ValueNetwork


class MCTSTest(unittest.TestCase):
    def test_uct_selects_legal_initial_move(self):
        state = OthelloState.new_game()
        agent = UCTAgent(iterations=20, seed=7)
        move = agent.select_move(state)
        self.assertIn(move, state.actions())

    def test_random_game_reaches_terminal_state(self):
        final_state, _ = play_game(RandomAgent(seed=1), RandomAgent(seed=2), record_states=True)
        self.assertTrue(final_state.is_terminal())
        self.assertLessEqual(sum(final_state.score()), 64)

    def test_value_network_roundtrip(self):
        network = ValueNetwork(hidden_sizes=(16,), seed=1)
        x = np.zeros((4, 128), dtype=np.float32)
        y = np.array([1, -1, 0, 1], dtype=np.float32)
        before = network.loss(x, y)
        network.train(x, y, epochs=1, batch_size=2, learning_rate=1e-2, verbose=False)
        after = network.loss(x, y)
        self.assertLessEqual(after, before + 0.1)


if __name__ == "__main__":
    unittest.main()
