"""Game core module for 2048 game logic."""

from dataclasses import dataclass
from typing import TypedDict, List, Dict, Optional
from enum import Enum

from .board import Board, encode_board_log2, decode_board_log2

__all__ = [
    "Board",
    "Action",
    "Position",
    "SpawnLocation",
    "MergeResult",
    "ResetInfo",
    "StepInfo",
    "encode_board_log2",
    "decode_board_log2",
]


@dataclass
class Position:
    """Represents a board position (row, column)."""

    row: int
    col: int

    @property
    def as_tuple(self) -> tuple:
        """Return position as (row, col) tuple."""
        return (self.row, self.col)


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


class Action(Enum):
    """Enumeration of possible game actions."""

    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"

    # map action to direction
    @classmethod
    def direction(cls, action: "Action") -> Position:
        """Get Direction enum corresponding to Action."""
        mapping = {
            cls.UP: Position(-1, 0),
            cls.DOWN: Position(1, 0),
            cls.LEFT: Position(0, -1),
            cls.RIGHT: Position(0, 1),
        }
        return mapping[action]


# Delayed imports to avoid circular dependencies
def __getattr__(name: str):
    if name == "GameEnv":
        from .game_env import GameEnv

        return GameEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
