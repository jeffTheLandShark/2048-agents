"""Logging and data processing utilities."""

from typing import TypedDict, List, Dict, Optional, Any
from .stats_logger import StatsLogger


class TileCounts(TypedDict, total=False):
    """Tile count distribution (keys are tile values as strings)."""
    pass  # Dynamic keys for tile values (2, 4, 8, etc.)


class GameSummary(TypedDict):
    """Summary statistics for a completed game."""
    final_score: int
    highest_tile: int
    game_length: int
    final_tile_counts: Dict[str, int]
    final_heuristics: Dict[str, float]


class StepLog(TypedDict, total=False):
    """Log entry for a single game step."""
    t: int
    board: List[int]  # Flattened log2 encoding
    action: Optional[str]
    reward: float
    score: int
    tile_counts: Dict[str, int]
    heuristics: Dict[str, float]
    done: bool


class GameLog(TypedDict, total=False):
    """Complete game log entry (JSONL format)."""
    game_id: str
    agent: str
    board_size: int
    seed: Optional[int]
    config: Dict[str, Any]
    steps: List[StepLog]
    summary: GameSummary


class ExperimentSummary(TypedDict, total=False):
    """Summary statistics for an experiment (multiple games)."""
    mean_score: float
    std_score: float
    mean_highest_tile: float
    games_reached_2048: int
    games_reached_4096: int
    games_reached_8192: int
    total_games: int


__all__ = [
    'StatsLogger',
    'TileCounts',
    'GameSummary',
    'StepLog',
    'GameLog',
    'ExperimentSummary',
]
