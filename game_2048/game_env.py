"""
Core Game Environment and Logic for 2048.

This module defines the GameEnv class, which encapsulates the rules, state transitions,
and scoring mechanisms of the 2048 game. It is designed to be deterministic (given a seed)
and efficient, avoiding dependencies on rendering libraries like Pygame for core logic.

Key functionalities:
- Board initialization and resetting.
- Deterministic tile spawning.
- Move validation and execution (sliding/merging).
- Game state tracking (score, game over condition).
"""

from typing import List, Tuple, Optional, TYPE_CHECKING
from abc import ABC
import numpy.random as random
import numpy as np
from .board import Board
from .utils import spawn_random_tile, slide_and_merge

# Import types from parent module (__init__.py) using TYPE_CHECKING to avoid circular import
if TYPE_CHECKING:
    from game_2048 import ResetInfo, StepInfo, Action, SpawnLocation, Position
else:
    # At runtime, import from parent module after it's fully initialized
    import sys

    _game_module = sys.modules.get("game_2048")
    if _game_module:
        ResetInfo = _game_module.ResetInfo
        StepInfo = _game_module.StepInfo
        Action = _game_module.Action
        SpawnLocation = _game_module.SpawnLocation
        Position = _game_module.Position


class GameEnv:
    """
    Pure game logic for 2048: board transitions, legal moves, scoring, random tile spawn.

    Provides deterministic behavior under a provided RNG seed.
    No pygame dependencies.
    """

    def __init__(
        self,
        board: Board | None = None,
        board_size: int = 4,
        seed: Optional[int] = None,
        rng: Optional[random.Generator] = None,
    ) -> None:
        """
        Initialize the game environment.

        Args:
            board_size: Size of the game board (nxn). Defaults to 4.
            seed: Random seed for deterministic behavior. If None, uses system randomness.
            rng: Optional RNG generator. If provided, seed is ignored.
        """
        self._rewards: list[int] = []

        if rng is not None:
            self._rng = rng
        else:
            self.seed(seed)

        if board is not None:
            self._board = board
            self.board_size = board.array.shape[0]
        else:
            self._board: Board = Board(
                np.zeros((board_size, board_size), dtype=np.int32)
            )
            self.board_size = board_size

    def reset(self) -> Tuple[Board, ResetInfo]:
        """
        Reset the game to initial state.

        Returns:
            Tuple of (initial_board, ResetInfo) where ResetInfo contains metadata.
        """
        self._board = Board(
            np.zeros((self.board_size, self.board_size), dtype=np.int32)
        )
        self._rewards = []  # Reset score tracking
        # Spawn two initial tiles
        self._board, _ = spawn_random_tile(self._board, self._rng)
        self._board, _ = spawn_random_tile(self._board, self._rng)
        info: ResetInfo = {
            "score": 0,
            "tile_counts": {},  # Placeholder for tile counts
            "heuristics": {},  # Placeholder for heuristics
        }
        return self._board, info

    def slide(self, action: str) -> tuple["Board", int]:
        """
        Slide the board in the given direction without merging tiles.

        Args:
            action: One of "UP", "DOWN", "LEFT", "RIGHT".

        Returns:
            New Board instance after sliding.
        """
        new_array, score_gained = slide_and_merge(self._board.array, action)
        return Board(new_array), score_gained

    def step(self, action: str) -> Tuple[Board, float, bool, StepInfo]:
        """
        Execute one game step.

        Args:
            action: One of "UP", "DOWN", "LEFT", "RIGHT".

        Returns:
            Tuple of (next_state, reward, done, info):
            - next_state: Board state after action and random spawn
            - reward: Score gained from this move
            - done: Whether game is over
            - info: StepInfo with additional metadata (score, tile counts, etc.)
        """
        next_state, reward = self.slide(action)
        self._board = next_state
        self._rewards.append(int(reward))  # Track reward for score calculation

        # Spawn random tile
        self._board, _ = spawn_random_tile(self._board, self._rng)

        # Check for game over AFTER spawning the new tile
        done = self.is_game_over(self._board)

        info = StepInfo(
            score=self.get_score(),
            tile_counts={},  # Placeholder for tile counts
            heuristics={},  # Placeholder for heuristics
            action_taken=action,
        )
        return self._board, float(reward), done, info

    def legal_moves(self, board: Optional[Board] = None) -> List[str]:
        """
        Get list of legal moves from current or given board state.

        Args:
            board: Optional board state. If None, uses current state.

        Returns:
            List of legal action strings ("UP", "DOWN", "LEFT", "RIGHT").
        """
        if board is None:
            board = self._board
        legal_actions = []
        for action in Action:
            next_state, _ = self.slide(action.value)
            if not np.array_equal(next_state.array, board.array):
                legal_actions.append(action.value)
        return legal_actions

    def is_game_over(self, board: Optional[Board] = None) -> bool:
        """
        Check if game is over (no legal moves available).

        Args:
            board: Optional board state. If None, uses current state.

        Returns:
            True if game is over, False otherwise.
        """
        return self.legal_moves(board) == []

    def get_score(self) -> int:
        """
        Get current game score.

        Returns:
            Current cumulative score.
        """
        return sum(self._rewards)

    def get_board(self) -> Board:
        """
        Get current board state.

        Returns:
            Current board as Board instance.
        """
        return self._board

    def seed(self, seed: Optional[int] = None) -> None:
        """
        Set random seed for deterministic behavior.

        Args:
            seed: Random seed value. If None, resets to non-deterministic.
        """
        self._rng = random.default_rng(seed)

    def copy(self) -> "GameEnv":
        """
        Create a deep copy of the game environment.

        Returns:
            New GameEnv instance with copied state and RNG.
        """
        new_env = GameEnv(
            board=self._board.copy(),
            board_size=self.board_size,
            rng=self._rng,
        )
        new_env._rewards = self._rewards.copy()
        return new_env

    def get_move_count(self) -> int:
        """
        Get the number of moves made so far.

        Returns:
            Number of moves taken in the game.
        """
        return len(self._rewards)
