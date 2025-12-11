"""Statistics logger for per-game logging with full state capture."""

from typing import List, Dict, Any, Optional, TYPE_CHECKING
from pathlib import Path
import json
import re
from stats_logging.etl import load_jsonl_logs
from game_2048 import Board, encode_board_log2

if TYPE_CHECKING:
    from stats_logging import GameSummary, GameLog, StepLog
    from heuristics import HeuristicFeatures
else:
    # Delayed imports to avoid circular dependency
    GameSummary = Dict[str, Any]
    GameLog = Dict[str, Any]
    StepLog = Dict[str, Any]
    HeuristicFeatures = None


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
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize statistics logger.

        Args:
            log_file: Path to JSONL log file (will be created/opened for appending).
            agent_name: Name of the agent being logged.
            board_size: Size of the game board.
            config: Optional configuration dictionary to include in logs.
        """
        self.log_file = Path(log_file)
        self.agent_name = agent_name
        self.board_size = board_size
        self.config = config or {}

        # Ensure directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Open file in append mode
        self._file = open(self.log_file, "a", encoding="utf-8")

        # Current game state
        self._current_game: Optional[GameLog] = None

    def _get_next_game_number(self, seed: Optional[int]) -> int:
        """
        Find the next game number by reading existing logs.

        Parses existing game IDs in the format: {agent_name}_{number}
        and returns the next available number.

        Args:
            seed: Optional seed value (ignored for numbering).

        Returns:
            Next game number (1 if no existing games found).
        """
        if not self.log_file.exists():
            return 1

        # Pattern: agent_name_123
        pattern = re.compile(rf"^{re.escape(self.agent_name)}_(\d+)$")
        max_number = 0

        try:
            for game_log in load_jsonl_logs(self.log_file):
                game_id = game_log.get("game_id", "")
                match = pattern.match(game_id)
                if match:
                    number = int(match.group(1))
                    max_number = max(max_number, number)
        except Exception:
            # If there's any error reading the file, start from 1
            return 1

        return max_number + 1

    def start_game(
        self, game_id: Optional[str] = None, seed: Optional[int] = None
    ) -> None:
        """
        Start logging a new game.

        Args:
            game_id: Unique identifier for this game. If None, auto-generated.
            seed: Optional random seed used for this game.
        """
        if game_id is None:
            next_num = self._get_next_game_number(seed)
            game_id = f"{self.agent_name}_{next_num}"

        self._current_game = GameLog(
            game_id=game_id,
            agent=self.agent_name,
            board_size=self.board_size,
            seed=seed,
            config=self.config,
            steps=[],
            summary=None,  # Will be set at end_game
        )

    def log_step(
        self,
        t: int,
        board: Board,
        action: Optional[str],
        reward: float,
        score: int,
        tile_counts: Dict[str, int],
        heuristics: Dict[str, float],  # HeuristicFeatures dictionary
        done: bool,
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
            heuristics: Dictionary with heuristic feature values.
            done: Whether game is over.
        """
        if self._current_game is None:
            raise RuntimeError(
                "Cannot log step: no game started. Call start_game() first."
            )

        step_log = StepLog(
            t=t,
            board=encode_board_log2(board),
            action=action,
            reward=reward,
            score=score,
            tile_counts=tile_counts,
            heuristics=heuristics,
            done=done,
        )

        self._current_game["steps"].append(step_log)

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
        if self._current_game is None:
            raise RuntimeError("Cannot end game: no game started.")

        self._current_game["summary"] = summary

        # Write to file as a single line JSON
        json.dump(self._current_game, self._file)
        self._file.write("\n")
        self.flush()

        self._current_game = None

    def flush(self) -> None:
        """Flush any buffered writes to disk."""
        if self._file:
            self._file.flush()

    def close(self) -> None:
        """Close log file and cleanup resources."""
        if self._file:
            self._file.close()
            self._file = None
