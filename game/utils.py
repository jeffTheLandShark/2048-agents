"""Utility functions for 2048 game: tile merging, random spawn, RNG helpers."""

from typing import List, Optional, Tuple
import numpy.random as random
import numpy as np
from game import MergeResult, SpawnLocation, Position, Board


def merge_line(line: List[int], reverse: bool = False) -> MergeResult:
    """
    Merge tiles in a single line (row or column) according to 2048 rules.

    Tiles slide toward one end and merge if adjacent tiles have the same value.
    Each tile can only merge once per move.

    Args:
        line: List of tile values (may contain zeros for empty cells).

    Returns:
        MergeResult with merged_line and score_gained.
    """
    # Remove zeros and prepare for merging
    merge_line = _drop_zeros(line)
    if reverse:
        merge_line.reverse()
    score_gained = 0
    while _merges_possible(merge_line):
        for i in range(len(merge_line) - 1):
            if merge_line[i] == merge_line[i + 1]:
                # Merge tiles
                new_value = merge_line[i] * 2
                score_gained += new_value
                merge_line[i] = new_value
                merge_line[i + 1] = 0
        merge_line = _drop_zeros(merge_line)

    merge_line = [0] * (len(line) - len(merge_line)) + merge_line

    if reverse:
        merge_line.reverse()

    return MergeResult(
        merged_line=merge_line,
        score_gained=score_gained,
    )


def _drop_zeros(line: List[int]) -> List[int]:
    """
    Remove zeros from the line, simulating tile sliding.

    Args:
        line: List of tile values (may contain zeros).
    Returns:
        List of tile values with zeros removed.
    """
    return [tile for tile in line if tile != 0]


def _merges_possible(line: List[int]) -> bool:
    """
    Check if any merges are possible in the given line.

    Args:
        line: List of tile values (no zeros).
    Returns:
        True if merges are possible, False otherwise.
    """
    return any(line[i] == line[i + 1] for i in range(len(line) - 1))


def spawn_random_tile(
    board: Board, rng: random.Generator, value: int = 2, probability_4: float = 0.1
) -> Tuple[Board, SpawnLocation]:
    """
    Spawn a random tile (2 or 4) in an empty cell.

    Args:
        board: Current board state.
        rng: Random number generator.
        value: Base tile value (typically 2).
        probability_4: Probability of spawning a 4 instead of value (default 0.1).

    Returns:
        Tuple of (updated_board, SpawnLocation) where SpawnLocation contains row and col.
    """
    empty_cells = _get_empty_cells(board)
    if not empty_cells:
        return board, SpawnLocation(-1, -1)  # No empty cells to spawn

    spawn_idx = rng.choice(len(empty_cells))
    spawn_cell = empty_cells[spawn_idx]  # random empty cell
    tile_value = value * 2 if rng.random() < probability_4 else value
    board[spawn_cell.as_tuple] = tile_value  # Set tile at (row, col) with tile_value
    return board, SpawnLocation(spawn_cell.row, spawn_cell.col)


def _get_empty_cells(board: Board) -> List[Position]:
    """
    Get list of coordinates for all empty cells on the board.

    Args:
        board: Board state as Board instance.

    Returns:
        List of Position objects for empty cells.
    """
    return [
        Position(row, col)
        for row in range(board.size)
        for col in range(board.size)
        if board[row, col] == 0
    ]
