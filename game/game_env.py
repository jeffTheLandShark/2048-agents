"""Core 2048 game environment with deterministic rules and transitions."""

from typing import List, Tuple, Optional
from abc import ABC
import numpy.random as random
import numpy as np
from game import ResetInfo, StepInfo, Board, Action, SpawnLocation, Position
from game.utils import spawn_random_tile, merge_line


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
        rng: Optional[random.Generator] = None,
    ) -> None:
        """
        Initialize the game environment.

        Args:
            board_size: Size of the game board (nxn). Defaults to 4.
            seed: Random seed for deterministic behavior. If None, uses system randomness.
            rng: Optional RNG generator. If provided, seed is ignored.
        """
        self.board_size = board_size
        if rng is not None:
            self._rng = rng
        else:
            self.seed(seed)
        self._board: Board = Board(np.zeros((board_size, board_size), dtype=np.int32))
        self._rewards: list[int] = []

    def reset(self) -> Tuple[Board, ResetInfo]:
        """
        Reset the game to initial state.

        Returns:
            Tuple of (initial_board, ResetInfo) where ResetInfo contains metadata.
        """
        self._board = Board(
            np.zeros((self.board_size, self.board_size), dtype=np.int32)
        )
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
        movement = Action.direction(Action[action])  # Get movement vector
        if (movement.row == 0 and movement.col == 0) or (
            movement.row != 0 and movement.col != 0
        ):  # if no movement or diagonal
            raise ValueError(f"Invalid action for sliding: {action}")

        axis: bool = movement.col == 0  # True for vertical, False for horizontal
        reverse = (movement.row if axis else movement.col) < 0  # True if moving left/up
        print("Axis:", axis, "Movement:", movement, "Reverse:", reverse)

        axis_iter = (
            [self._board.array[:, col].tolist() for col in range(self._board.size)]
            if axis
            else self._board.to_list()
        )  # Get rows or columns

        score_gained = 0
        new_lines = []
        for line in axis_iter:
            merged = merge_line(line, reverse=reverse)
            new_lines.append(merged.merged_line)
            score_gained += merged.score_gained
        new_array = (
            np.array(new_lines, dtype=np.int32).T
            if axis
            else np.array(new_lines, dtype=np.int32)
        )

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
        done = self.is_game_over(next_state)
        info = StepInfo(
            score=self.get_score(),
            tile_counts={},  # Placeholder for tile counts
            heuristics={},  # Placeholder for heuristics
            action_taken=action,
        )
        self._board = next_state
        if not done:
            self._board, _ = spawn_random_tile(self._board, self._rng)
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
