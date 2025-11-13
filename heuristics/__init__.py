"""Heuristic evaluation functions for 2048."""

from typing import TypedDict


class HeuristicFeatures(TypedDict):
    """Heuristic feature values for a board state."""
    empty: float
    monotonicity: float
    smoothness: float
    merge_potential: float
    max_tile: float
    sum_tiles: float
