
import unittest
import numpy as np
from game.utils import slide_and_merge

class TestLogic(unittest.TestCase):
    def test_simple_merge_right(self):
        # [2, 2, 0, 0] -> [0, 0, 0, 4]
        board = np.zeros((4, 4), dtype=np.int32)
        board[0] = [2, 2, 0, 0]
        res, score = slide_and_merge(board, 'RIGHT')
        np.testing.assert_array_equal(res[0], [0, 0, 0, 4])
        self.assertEqual(score, 4)

    def test_recursive_merge_prevention_right(self):
        # [4, 2, 2, 0] -> [0, 0, 4, 4] (Not [0, 0, 0, 8])
        board = np.zeros((4, 4), dtype=np.int32)
        board[0] = [4, 2, 2, 0]
        res, score = slide_and_merge(board, 'RIGHT')
        np.testing.assert_array_equal(res[0], [0, 0, 4, 4])
        self.assertEqual(score, 4) # Only 2+2=4 merged

    def test_recursive_merge_prevention_left(self):
        # [0, 2, 2, 4] -> [4, 4, 0, 0]
        board = np.zeros((4, 4), dtype=np.int32)
        board[0] = [0, 2, 2, 4]
        res, score = slide_and_merge(board, 'LEFT')
        np.testing.assert_array_equal(res[0], [4, 4, 0, 0])
        self.assertEqual(score, 4)

    def test_double_merge_right(self):
        # [2, 2, 2, 2] -> [0, 0, 4, 4]
        board = np.zeros((4, 4), dtype=np.int32)
        board[0] = [2, 2, 2, 2]
        res, score = slide_and_merge(board, 'RIGHT')
        np.testing.assert_array_equal(res[0], [0, 0, 4, 4])
        self.assertEqual(score, 8)

    def test_gap_merge_right(self):
        # [2, 0, 2, 0] -> [0, 0, 0, 4]
        board = np.zeros((4, 4), dtype=np.int32)
        board[0] = [2, 0, 2, 0]
        res, score = slide_and_merge(board, 'RIGHT')
        np.testing.assert_array_equal(res[0], [0, 0, 0, 4])
        self.assertEqual(score, 4)

    def test_full_board_up(self):
        # [2, 4, 8, 16]
        # [2, 4, 8, 16]
        # [0, 0, 0, 0 ]
        # [0, 0, 0, 0 ]
        # -> UP
        # [4, 8, 16, 32]
        board = np.zeros((4, 4), dtype=np.int32)
        board[0] = [2, 4, 8, 16]
        board[1] = [2, 4, 8, 16]
        res, score = slide_and_merge(board, 'UP')
        expected = np.array([4, 8, 16, 32])
        np.testing.assert_array_equal(res[0], expected)
        np.testing.assert_array_equal(res[1], [0, 0, 0, 0])
        self.assertEqual(score, 4+8+16+32)

if __name__ == '__main__':
    unittest.main()

