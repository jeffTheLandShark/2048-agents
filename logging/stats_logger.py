"""Statistics logger for per-game logging with full state capture."""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json


class StatsLogger:
    """
    Logger for 2048 game statistics.

    Writes JSONL format (one JSON object per game) for append efficiency.
    Each game log contains full state history for deterministic replays.
    """

    def __init__(
        self,
        log_file: Path,
        agent_name: str,
        board_size: int = 4,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize statistics logger.

        Args:
            log_file: Path to JSONL log file (will be created/opened for appending).
            agent_name: Name of the agent being logged.
            board_size: Size of the game board.
            config: Optional configuration dictionary to include in logs.
        """
        raise NotImplementedError

    def start_game(self, game_id: str, seed: Optional[int] = None) -> None:
        """
        Start logging a new game.

        Args:
            game_id: Unique identifier for this game.
            seed: Optional random seed used for this game.
        """
        raise NotImplementedError

    def log_step(
        self,
        t: int,
        board: List[List[int]],
        action: Optional[str],
        reward: float,
        score: int,
        tile_counts: Dict[str, int],
        heuristics: Dict[str, float],
        done: bool
    ) -> None:
        """
        Log a single game step.

        Args:
            t: Time step (0-indexed).
            board: Board state as 2D list.
            action: Action taken (None for initial state).
            reward: Reward received this step.
            score: Cumulative score after this step.
            tile_counts: Dictionary mapping tile values to counts.
            heuristics: Dictionary of heuristic feature values.
            done: Whether game is over.
        """
        raise NotImplementedError

    def end_game(self, summary: Dict[str, Any]) -> None:
        """
        Finalize and write game log to file.

        Args:
            summary: Dictionary containing game summary statistics:
                     - final_score: Final game score
                     - highest_tile: Highest tile reached
                     - game_length: Number of steps
                     - final_tile_counts: Final tile distribution
                     - final_heuristics: Final heuristic values
        """
        raise NotImplementedError

    def flush(self) -> None:
        """Flush any buffered writes to disk."""
        raise NotImplementedError

    def close(self) -> None:
        """Close log file and cleanup resources."""
        raise NotImplementedError

