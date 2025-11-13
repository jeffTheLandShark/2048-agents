"""Utility functions for 2048 game: tile merging, random spawn, RNG helpers."""

from typing import List, Optional, Tuple
import numpy.random as random
from game import MergeResult, SpawnLocation, Position, Board


def merge_line(line: List[int]) -> MergeResult:
    """
    Merge tiles in a single line (row or column) according to 2048 rules.

    Tiles slide toward one end and merge if adjacent tiles have the same value.
    Each tile can only merge once per move.

    Args:
        line: List of tile values (may contain zeros for empty cells).

    Returns:
        MergeResult with merged_line and score_gained.
    """
    raise NotImplementedError


def spawn_random_tile(
    board: Board,
    rng: random.Generator,
    value: int = 2,
    probability_4: float = 0.1
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
    raise NotImplementedError


def get_empty_cells(board: Board) -> List[Position]:
    """
    Get list of coordinates for all empty cells on the board.

    Args:
        board: Board state as Board instance.

    Returns:
        List of Position objects for empty cells.
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

