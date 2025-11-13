"""Game core module for 2048 game logic."""

from dataclasses import dataclass
from typing import TypedDict, List, Dict, Optional

from .board import Board

__all__ = [
    'Board',
    'Position',
    'SpawnLocation',
    'MergeResult',
    'ResetInfo',
    'StepInfo',
]


@dataclass
class Position:
    """Represents a board position (row, column)."""
    row: int
    col: int


@dataclass
class SpawnLocation:
    """Represents a tile spawn location."""
    row: int
    col: int


@dataclass
class MergeResult:
    """Result of merging a line of tiles."""
    merged_line: List[int]
    score_gained: int


class ResetInfo(TypedDict, total=False):
    """Information dictionary returned by GameEnv.reset()."""
    score: int
    tile_counts: Dict[str, int]
    heuristics: Dict[str, float]


class StepInfo(TypedDict, total=False):
    """Information dictionary returned by GameEnv.step()."""
    score: int
    tile_counts: Dict[str, int]
    heuristics: Dict[str, float]
    action_taken: str
