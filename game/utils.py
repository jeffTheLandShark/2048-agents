"""Utility functions for 2048 game: tile merging, random spawn, RNG helpers."""

from typing import List, Tuple, Optional
import numpy.random as random


def merge_line(line: List[int]) -> Tuple[List[int], int]:
    """
    Merge tiles in a single line (row or column) according to 2048 rules.

    Tiles slide toward one end and merge if adjacent tiles have the same value.
    Each tile can only merge once per move.

    Args:
        line: List of tile values (may contain zeros for empty cells).

    Returns:
        Tuple of (merged_line, score_gained) where:
        - merged_line: Line after merging and sliding
        - score_gained: Points earned from merges in this line
    """
    raise NotImplementedError


def spawn_random_tile(
    board: List[List[int]],
    rng: random.Generator,
    value: int = 2,
    probability_4: float = 0.1
) -> Tuple[List[List[int]], Tuple[int, int]]:
    """
    Spawn a random tile (2 or 4) in an empty cell.

    Args:
        board: Current board state.
        rng: Random number generator.
        value: Base tile value (typically 2).
        probability_4: Probability of spawning a 4 instead of value (default 0.1).

    Returns:
        Tuple of (updated_board, (row, col)) where (row, col) is spawn location.
    """
    raise NotImplementedError


def get_empty_cells(board: List[List[int]]) -> List[Tuple[int, int]]:
    """
    Get list of coordinates for all empty cells on the board.

    Args:
        board: Board state as 2D list.

    Returns:
        List of (row, col) tuples for empty cells.
    """
    raise NotImplementedError


def create_rng(seed: Optional[int] = None) -> random.Generator:
    """
    Create a numpy random number generator with optional seed.

    Args:
        seed: Optional random seed for deterministic behavior.

    Returns:
        numpy.random.Generator instance.
    """
    raise NotImplementedError

