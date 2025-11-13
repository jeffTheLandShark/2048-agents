"""Statistics logger for per-game logging with full state capture."""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from logging import GameSummary
from heuristics import HeuristicFeatures
from game import Board


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
        board: Board,
        action: Optional[str],
        reward: float,
        score: int,
        tile_counts: Dict[str, int],
        heuristics: HeuristicFeatures,
        done: bool
    ) -> None:
        """
        Log a single game step.

        Args:
            t: Time step (0-indexed).
            board: Board state as Board instance. Will be converted to flattened encoding for logging.
            action: Action taken (None for initial state).
            reward: Reward received this step.
            score: Cumulative score after this step.
            tile_counts: Dictionary mapping tile values to counts.
            heuristics: HeuristicFeatures dictionary with feature values.
            done: Whether game is over.
        """
        raise NotImplementedError

    def end_game(self, summary: GameSummary) -> None:
        """
        Finalize and write game log to file.

        Args:
            summary: GameSummary dictionary containing game summary statistics:
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

