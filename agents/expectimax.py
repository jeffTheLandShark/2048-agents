"""Expectimax agent implementation for 2048."""

import time
import numpy as np
from typing import List, Dict, Optional, Tuple
from agents import Agent
from game_2048 import Board
from heuristics.evaluator import HeuristicEvaluator
from game_2048.utils import slide_and_merge

# Default weights (guessed/estimated)
DEFAULT_WEIGHTS = {
    "empty": 30.0,
    "monotonicity": 35.0,
    "smoothness": 5.0,
    "merge_potential": 10.0,
    "max_tile": 1.0,
    "sum_tiles": 0.0,
}

BEST_WEIGHTS = {
    "empty": 9.2,
    "monotonicity": 3.3,
    "smoothness": 3.2,
    "merge_potential": 6.4,
    "max_tile": 0,
    "sum_tiles": 0,
}


class ExpectimaxAgent(Agent):
    """
    Expectimax search agent with configurable heuristic weights.

    Uses iterative deepening and weighted heuristic evaluation.
    Supports both fixed (best-known) weights and GA-optimized weights.
    """

    def __init__(
        self,
        depth_limit: int = 7,
        time_limit_ms: Optional[int] = None,
        heuristic_weights: Optional[Dict[str, float]] = None,
        use_iterative_deepening: bool = True,
        sample_empty_cells: bool = False,
        max_empty_samples: int = 6,
        sample_depth_cutoff: int = 2,
        sample_rng_seed: int = 0,
    ) -> None:
        """
        Initialize Expectimax agent.

        Args:
            depth_limit: Maximum search depth.
            time_limit_ms: Optional time limit per move in milliseconds.
            heuristic_weights: Dictionary mapping heuristic names to weights.
                              If None, uses default best-known weights.
            use_iterative_deepening: Whether to use iterative deepening search.
        """
        weights = (
            heuristic_weights if heuristic_weights is not None else DEFAULT_WEIGHTS
        )
        self.evaluator = HeuristicEvaluator(weights)
        self.depth_limit = depth_limit
        self.time_limit_ms = time_limit_ms
        self.use_iterative_deepening = use_iterative_deepening
        # Optional sampling to cap branching factor in deep chance nodes
        self.sample_empty_cells = sample_empty_cells
        self.max_empty_samples = max_empty_samples
        self.sample_depth_cutoff = sample_depth_cutoff
        self._sample_rng = np.random.default_rng(sample_rng_seed)
        self._start_time = 0.0

    def choose_action(self, state: Board, legal_moves: List[str]) -> str:
        """
        Choose action using Expectimax search.

        Args:
            state: Current board state as Board instance.
            legal_moves: List of legal action strings.

        Returns:
            Best action according to Expectimax search.
        """
        if not legal_moves:
            return "UP"  # Should not happen if game checks for game over

        self._start_time = time.time()
        best_action = legal_moves[0]

        # Iterative deepening
        current_depth = 1
        max_depth = self.depth_limit

        # If using iterative deepening, start from depth 1.
        # Otherwise jump straight to depth_limit.
        start_depth = 1 if self.use_iterative_deepening else max_depth

        for depth in range(start_depth, max_depth + 1):
            try:
                score, action = self._maximize(state, depth)
                if action:
                    best_action = action
            except TimeoutError:
                break

            if self._check_timeout():
                break

            if not self.use_iterative_deepening:
                break

        return best_action

    def set_heuristic_weights(self, weights: Dict[str, float]) -> None:
        """
        Update heuristic weights (e.g., from GA optimization).

        Args:
            weights: Dictionary mapping heuristic names to weights.
        """
        self.evaluator.set_weights(weights)

    def _check_timeout(self) -> bool:
        """Check if time limit has been exceeded."""
        if self.time_limit_ms is None:
            return False
        elapsed = (time.time() - self._start_time) * 1000
        return elapsed >= self.time_limit_ms

    def _maximize(self, board: Board, depth: int) -> Tuple[float, Optional[str]]:
        """
        Max node: Player's turn. Returns (score, best_action).
        """
        if self._check_timeout():
            raise TimeoutError("Time limit exceeded")

        if depth == 0:
            return self.evaluator.evaluate(board), None

        # We need to know legal moves.
        # Instead of calling game_2048.legal_moves which creates Boards,
        # we can try all 4 directions and see if board changes.
        # Since we are in the search loop, optimizing this is key.
        # However, to keep it clean, let's iterate through possible actions.

        best_score = float("-inf")
        best_action = None

        # Actions: UP, DOWN, LEFT, RIGHT
        # We can iterate them directly.
        actions = ["UP", "DOWN", "LEFT", "RIGHT"]

        moves_found = False

        for action in actions:
            # Simulate move
            # slide_and_merge returns (new_array, reward)
            # Note: Expectimax usually optimizes Final Score or Position Value?
            # Here we optimize Heuristic Value of the resulting state.
            # Some implementations add the immediate reward to the heuristic score.
            # Let's just stick to heuristic evaluation of the leaf for now,
            # or heuristic + reward.
            # Usually heuristic includes score-like components.
            # Let's use Heuristic(next_state).

            new_array, _ = slide_and_merge(board.array, action)

            if np.array_equal(new_array, board.array):
                continue

            moves_found = True

            # Create new board object for next step
            # Use copy=False-ish logic? Board(array) makes a copy of array if we don't be careful?
            # Board(new_array) is fine.
            next_board = Board(new_array)

            # Call Chance node
            score = self._chance(next_board, depth - 1)

            if score > best_score:
                best_score = score
                best_action = action

        if not moves_found:
            # No legal moves, terminal state
            return self.evaluator.evaluate(board), None

        return best_score, best_action

    def _chance(self, board: Board, depth: int) -> float:
        """
        Chance node: Random tile spawn. Returns weighted average score.
        """
        if self._check_timeout():
            raise TimeoutError("Time limit exceeded")

        if depth == 0:
            return self.evaluator.evaluate(board)

        # Find empty cells
        empty_rows, empty_cols = np.where(board.array == 0)
        empty_cells = list(zip(empty_rows, empty_cols))

        if not empty_cells:
            return self.evaluator.evaluate(board)

        # Optionally sample empty cells to cap branching factor at deeper depths.
        sampled_cells = empty_cells
        if (
            self.sample_empty_cells
            and depth > self.sample_depth_cutoff
            and len(empty_cells) > self.max_empty_samples
        ):
            sampled_indices = self._sample_rng.choice(
                len(empty_cells), size=self.max_empty_samples, replace=False
            )
            sampled_cells = [empty_cells[i] for i in sampled_indices]

        # 2 tile (0.9 prob), 4 tile (0.1 prob)
        avg_score = 0.0
        prob_2 = 0.9
        prob_4 = 0.1

        num_cells = len(sampled_cells)
        prob_cell = 1.0 / num_cells

        for r, c in sampled_cells:
            # Spawn 2
            board.array[r, c] = 2
            score_2, _ = self._maximize(board, depth - 1)
            avg_score += score_2 * prob_2 * prob_cell

            # Spawn 4
            board.array[r, c] = 4
            score_4, _ = self._maximize(board, depth - 1)
            avg_score += score_4 * prob_4 * prob_cell

            # Revert
            board.array[r, c] = 0

        return avg_score
