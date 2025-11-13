"""Core 2048 game environment with deterministic rules and transitions."""

from typing import List, Tuple, Optional, Dict, Any
from abc import ABC
import numpy.random as random


class GameEnv:
    """
    Pure game logic for 2048: board transitions, legal moves, scoring, random tile spawn.

    Provides deterministic behavior under a provided RNG seed.
    No pygame dependencies.
    """

    def __init__(
        self,
        board_size: int = 4,
        seed: Optional[int] = None,
        rng: Optional[random.Generator] = None
    ) -> None:
        """
        Initialize the game environment.

        Args:
            board_size: Size of the game board (n×n). Defaults to 4.
            seed: Random seed for deterministic behavior. If None, uses system randomness.
            rng: Optional RNG generator. If provided, seed is ignored.
        """
        raise NotImplementedError

    def reset(self) -> Tuple[List[List[int]], Dict[str, Any]]:
        """
        Reset the game to initial state.

        Returns:
            Tuple of (initial_board, info_dict) where info_dict contains metadata.
        """
        raise NotImplementedError

    def step(self, action: str) -> Tuple[List[List[int]], float, bool, Dict[str, Any]]:
        """
        Execute one game step.

        Args:
            action: One of "UP", "DOWN", "LEFT", "RIGHT".

        Returns:
            Tuple of (next_state, reward, done, info):
            - next_state: Board state after action and random spawn
            - reward: Score gained from this move
            - done: Whether game is over
            - info: Additional metadata (score, tile counts, etc.)
        """
        raise NotImplementedError

    def legal_moves(self, board: Optional[List[List[int]]] = None) -> List[str]:
        """
        Get list of legal moves from current or given board state.

        Args:
            board: Optional board state. If None, uses current state.

        Returns:
            List of legal action strings ("UP", "DOWN", "LEFT", "RIGHT").
        """
        raise NotImplementedError

    def is_game_over(self, board: Optional[List[List[int]]] = None) -> bool:
        """
        Check if game is over (no legal moves available).

        Args:
            board: Optional board state. If None, uses current state.

        Returns:
            True if game is over, False otherwise.
        """
        raise NotImplementedError

    def get_score(self) -> int:
        """
        Get current game score.

        Returns:
            Current cumulative score.
        """
        raise NotImplementedError

    def get_board(self) -> List[List[int]]:
        """
        Get current board state.

        Returns:
            Current board as 2D list of tile values.
        """
        raise NotImplementedError

    def seed(self, seed: Optional[int] = None) -> None:
        """
        Set random seed for deterministic behavior.

        Args:
            seed: Random seed value. If None, resets to non-deterministic.
        """
        raise NotImplementedError

