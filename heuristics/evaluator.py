"""Heuristic evaluator for computing weighted board state scores."""

import numpy as np
from typing import Dict
from game_2048 import Board
from heuristics.features import compute_all_features


class HeuristicEvaluator:
    """
    Weighted heuristic evaluator for 2048 board states.

    Computes H(s) = Σ w_i * f_i where w_i are weights and f_i are feature values.
    Used by Expectimax (both fixed and GA-optimized variants).
    """

    def __init__(self, weights: Dict[str, float]) -> None:
        """
        Initialize evaluator with heuristic weights.

        Args:
            weights: Dictionary mapping feature names to their weights.
                    Example: {"empty": 2.7, "monotonicity": 1.0, "smoothness": 0.1}
        """
        self.weights = weights

    def evaluate(self, board: Board) -> float:
        """
        Evaluate board state using weighted heuristic features.

        Args:
            board: Board state as Board instance.

        Returns:
            Heuristic score H(s) = Σ w_i * f_i.
        """
        features = compute_all_features(board)
        score = 0.0
        for name, value in features.items():
            if name in self.weights:
                # Apply transformation for specific features if needed
                val = value
                if name == "max_tile" and value > 0:
                    val = float(np.log2(value))
                if name == "empty" and value > 0:
                    val = float(value/board.size**2)

                score += self.weights[name] * val
        # Bonus if max tile is in any corner (position [size-1, size-1])
        max_tile_value = features["max_tile"]
        corners = [(0, 0), (board.size - 1, 0), (0, board.size - 1), (board.size - 1, board.size - 1)]
        for corner in corners:
            if max_tile_value == board.array[corner] and max_tile_value > 0:
                score += 3.0
                break # only one corner bonus
        return score

    def get_weights(self) -> Dict[str, float]:
        """
        Get current heuristic weights.

        Returns:
            Dictionary of current weights.
        """
        return self.weights.copy()

    def set_weights(self, weights: Dict[str, float]) -> None:
        """
        Update heuristic weights (e.g., from GA optimization).

        Args:
            weights: Dictionary mapping feature names to new weights.
        """
        self.weights = weights.copy()
