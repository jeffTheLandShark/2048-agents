"""Heuristic feature computation functions for 2048 board evaluation."""

from typing import List
from heuristics import HeuristicFeatures
from game import Board


def compute_empty_tiles(board: Board) -> int:
    """
    Compute number of empty tiles on the board.

    Args:
        board: Board state as Board instance. Use board.array for vectorized operations.

    Returns:
        Count of empty (zero) tiles.
    """
    raise NotImplementedError


def compute_monotonicity(board: Board) -> float:
    """
    Compute monotonicity score (preference for increasing/decreasing sequences).

    Measures how well tiles are arranged in monotonic order (increasing or decreasing).
    Higher values indicate better organization.

    Args:
        board: Board state as Board instance. Use board.array for vectorized operations.

    Returns:
        Monotonicity score (typically normalized to [0, 1]).
    """
    raise NotImplementedError


def compute_smoothness(board: Board) -> float:
    """
    Compute smoothness score (penalty for adjacent tiles with different values).

    Lower values indicate smoother board (fewer adjacent differences).

    Args:
        board: Board state as Board instance. Use board.array for vectorized operations.

    Returns:
        Smoothness score (typically normalized).
    """
    raise NotImplementedError


def compute_merge_potential(board: Board) -> float:
    """
    Compute merge potential (likelihood of creating merges).

    Measures how many adjacent tiles have the same value and could merge.

    Args:
        board: Board state as Board instance. Use board.array for vectorized operations.

    Returns:
        Merge potential score (typically normalized).
    """
    raise NotImplementedError


def compute_max_tile(board: Board) -> int:
    """
    Get the maximum tile value on the board.

    Args:
        board: Board state as Board instance. Use board.array for vectorized operations.

    Returns:
        Maximum tile value (e.g., 2, 4, 8, ..., 2048).
    """
    raise NotImplementedError


def compute_sum_tiles(board: Board) -> int:
    """
    Compute sum of all tile values on the board.

    Args:
        board: Board state as Board instance. Use board.array for vectorized operations.

    Returns:
        Sum of all tile values.
    """
    raise NotImplementedError


def compute_all_features(board: Board) -> HeuristicFeatures:
    """
    Compute all heuristic features for a board state.

    Args:
        board: Board state as Board instance. Use board.array for vectorized operations.

    Returns:
        HeuristicFeatures dictionary with all feature values.
    """
    raise NotImplementedError

