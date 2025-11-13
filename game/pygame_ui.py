"""Pygame UI for human play and replay visualization."""

from typing import Optional, List
from pathlib import Path
from game import StepInfo, Board


class PygameUI:
    """
    Pygame-based UI for 2048 game.

    Supports:
    - Human play mode (keyboard input)
    - Replay mode (from raw JSONL logs)
    - Visual rendering of board states
    """

    def __init__(
        self,
        board_size: int = 4,
        window_size: int = 600,
        fps: int = 60
    ) -> None:
        """
        Initialize pygame UI.

        Args:
            board_size: Size of the game board (nxn).
            window_size: Window size in pixels (square).
            fps: Frames per second for rendering.
        """
        raise NotImplementedError

    def render(self, board: Board, score: int, info: Optional[StepInfo] = None) -> None:
        """
        Render current board state.

        Args:
            board: Current board state as Board instance. Convert to list for rendering if needed using board.to_list().
            score: Current game score.
            info: Optional StepInfo with additional metadata to display.
        """
        raise NotImplementedError

    def handle_input(self) -> Optional[str]:
        """
        Handle keyboard input for human play.

        Returns:
            Action string ("UP", "DOWN", "LEFT", "RIGHT") or None if no input.
        """
        raise NotImplementedError

    def run_human_play(self, env) -> None:
        """
        Run interactive human play loop.

        Args:
            env: GameEnv instance to play with.
        """
        raise NotImplementedError

    def run_replay(self, log_file: Path, speed: float = 1.0) -> None:
        """
        Run replay visualization from JSONL log file.

        Args:
            log_file: Path to JSONL log file containing game data.
            speed: Playback speed multiplier (1.0 = normal speed).
        """
        raise NotImplementedError

    def close(self) -> None:
        """Close pygame window and cleanup resources."""
        raise NotImplementedError

