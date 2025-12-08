import numpy as np
from numpy.random import default_rng

from game_2048.game_env import GameEnv
from game_2048 import Board
from game_2048.board import encode_board_flat


def test_gameenv_reset_spawns_two_tiles_and_is_deterministic():
    env = GameEnv(board_size=4, seed=42)
    b, info = env.reset()
    flat = b.to_list()
    nonzero = sum(1 for r in flat for v in r if v != 0)
    assert nonzero == 2

    # With same seed, a new env should produce same initial positions
    env2 = GameEnv(board_size=4, seed=42)
    b2, _ = env2.reset()
    assert b2 == b


def test_slide_left_and_right_behavior():
    # set a board with a merge possibility in the first row
    arr = np.zeros((4, 4), dtype=np.int32)
    arr[0] = [2, 2, 0, 0]
    board = Board(arr)
    env = GameEnv(board)

    left_board, left_score = env.slide("LEFT")
    # moving left (reverse True in merge_line) should produce [4,0,0,0]
    assert left_score == 4
    assert left_board[0, 0] == 4

    # reset board to same initial and slide RIGHT
    env._board = Board(arr)
    right_board, right_score = env.slide("RIGHT")
    assert right_score == 4
    # moving right should place merged tile at end of row
    assert right_board[0, 3] == 4


def test_legal_moves_detection():
    # full board with no merges
    board = Board(np.array([[2, 4], [8, 16]], dtype=np.int32))
    env = GameEnv(board)
    assert env.legal_moves(env._board) == []

    # a board with a possible merge should list moves
    env._board = Board(np.array([[2, 2], [4, 8]], dtype=np.int32))
    moves = env.legal_moves(env._board)
    assert isinstance(moves, list)
    assert "LEFT" in moves or "RIGHT" in moves or "UP" in moves or "DOWN" in moves


def test_reset_spawns_two_tiles():
    rng = np.random.default_rng(12345)
    env = GameEnv(board_size=4, rng=rng)
    board, info = env.reset()
    flat = encode_board_flat(board)
    nonzero = [v for v in flat if v != 0]
    assert len(nonzero) == 2
    # spawned values should be powers of two (2 or 4)
    assert all((v == 2 or v == 4) for v in nonzero)
