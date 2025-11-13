"""Heuristic feature computation functions for 2048 board evaluation."""

from typing import List, Dict


def compute_empty_tiles(board: List[List[int]]) -> int:
    """
    Compute number of empty tiles on the board.

    Args:
        board: Board state as 2D list of tile values.

    Returns:
        Count of empty (zero) tiles.
    """
    raise NotImplementedError


def compute_monotonicity(board: List[List[int]]) -> float:
    """
    Compute monotonicity score (preference for increasing/decreasing sequences).

    Measures how well tiles are arranged in monotonic order (increasing or decreasing).
    Higher values indicate better organization.

    Args:
        board: Board state as 2D list of tile values.

    Returns:
        Monotonicity score (typically normalized to [0, 1]).
    """
    raise NotImplementedError


def compute_smoothness(board: List[List[int]]) -> float:
    """
    Compute smoothness score (penalty for adjacent tiles with different values).

    Lower values indicate smoother board (fewer adjacent differences).

    Args:
        board: Board state as 2D list of tile values.

    Returns:
        Smoothness score (typically normalized).
    """
    raise NotImplementedError


def compute_merge_potential(board: List[List[int]]) -> float:
    """
    Compute merge potential (likelihood of creating merges).

    Measures how many adjacent tiles have the same value and could merge.

    Args:
        board: Board state as 2D list of tile values.

    Returns:
        Merge potential score (typically normalized).
    """
    raise NotImplementedError


def compute_max_tile(board: List[List[int]]) -> int:
    """
    Get the maximum tile value on the board.

    Args:
        board: Board state as 2D list of tile values.

    Returns:
        Maximum tile value (e.g., 2, 4, 8, ..., 2048).
    """
    raise NotImplementedError


def compute_sum_tiles(board: List[List[int]]) -> int:
    """
    Compute sum of all tile values on the board.

    Args:
        board: Board state as 2D list of tile values.

    Returns:
        Sum of all tile values.
    """
    raise NotImplementedError


def compute_all_features(board: List[List[int]]) -> Dict[str, float]:
    """
    Compute all heuristic features for a board state.

    Args:
        board: Board state as 2D list of tile values.

    Returns:
        Dictionary mapping feature names to their values.
    """
    raise NotImplementedError

