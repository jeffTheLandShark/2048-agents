import numpy as np
from numpy.random import default_rng

from game_2048.utils import merge_line, spawn_random_tile, _get_empty_cells
from game_2048 import Board


def test_merge_line_simple_no_reverse():
    line = [2, 2, 0, 0]
    result = merge_line(line, reverse=False)
    assert result.score_gained == 4
    assert result.merged_line == [0, 0, 0, 4]


def test_merge_line_with_reverse():
    # reverse True should merge toward the left/start
    line = [2, 2, 0, 0]
    result = merge_line(line, reverse=True)
    assert result.score_gained == 4
    assert result.merged_line == [4, 0, 0, 0]


def test_drop_and_get_empty_cells_and_spawn():
    b = Board(np.zeros((2, 2), dtype=np.int32))
    empty = _get_empty_cells(b)
    assert len(empty) == 4

    rng = default_rng(0)
    b2, loc = spawn_random_tile(b, rng)
    # After spawn, exactly one cell should be non-zero
    flat = b2.to_list()
    nonzero = sum(1 for r in flat for v in r if v != 0)
    assert nonzero == 1
    # Spawned value must be 2 or 4
    vals = {v for r in flat for v in r}
    vals.discard(0)
    assert vals <= {2, 4}


def test_spawn_on_full_board_returns_negative_location():
    b = Board(np.array([[2, 4], [8, 16]], dtype=np.int32))
    rng = default_rng(1)
    b2, loc = spawn_random_tile(b, rng)
    assert loc.row == -1 and loc.col == -1


def test_merge_line_basic():
    line = [2, 2, 0, 2]
    result = merge_line(line, reverse=False)
    assert result.merged_line == [0, 0, 4, 2]
    assert result.score_gained == 4


def test_spawn_random_tile_places_in_empty_cell():
    # small board with one empty cell at (1,1)
    board = Board(np.array([[2, 4], [8, 0]], dtype=np.int32))
    rng = np.random.default_rng(0)
    new_board, loc = spawn_random_tile(board, rng, value=2, probability_4=0.0)
    # With probability_4=0.0, spawned value must be `value` (2)
    assert new_board[loc.row, loc.col] == 2
    assert (loc.row, loc.col) == (1, 1)
