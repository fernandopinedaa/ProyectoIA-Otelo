import unittest

import numpy as np

from othello_ai.game import BLACK, EMPTY, WHITE, OthelloState, coord_to_move, move_to_coord


class OthelloRulesTest(unittest.TestCase):
    def test_initial_legal_moves_for_black(self):
        state = OthelloState.new_game()
        self.assertEqual(
            set(state.legal_moves(BLACK)),
            {(2, 3), (3, 2), (4, 5), (5, 4)},
        )

    def test_apply_move_flips_disc(self):
        state = OthelloState.new_game()
        next_state = state.apply_move((2, 3))

        self.assertEqual(next_state.board[2, 3], BLACK)
        self.assertEqual(next_state.board[3, 3], BLACK)
        self.assertEqual(next_state.current_player, WHITE)
        self.assertEqual(next_state.score(), (4, 1))

    def test_pass_is_only_legal_without_moves(self):
        state = OthelloState.new_game()
        with self.assertRaises(ValueError):
            state.apply_move(None)

        board = np.zeros((8, 8), dtype=np.int8)
        board[0, 0] = BLACK
        board[0, 1] = WHITE
        pass_state = OthelloState(board, WHITE)
        self.assertEqual(pass_state.actions(), [None])
        self.assertEqual(pass_state.apply_move(None).current_player, BLACK)

    def test_terminal_winner_and_result(self):
        board = np.full((8, 8), BLACK, dtype=np.int8)
        board[0, 0] = WHITE
        state = OthelloState(board, BLACK)
        self.assertTrue(state.is_terminal())
        self.assertEqual(state.winner(), BLACK)
        self.assertEqual(state.result_for(BLACK), 1.0)
        self.assertEqual(state.result_for(WHITE), -1.0)

    def test_coordinate_conversion(self):
        self.assertEqual(coord_to_move("d3"), (2, 3))
        self.assertEqual(move_to_coord((2, 3)), "d3")


if __name__ == "__main__":
    unittest.main()
