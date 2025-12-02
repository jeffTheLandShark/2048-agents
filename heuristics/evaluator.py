"""Heuristic evaluator for computing weighted board state scores."""

from typing import List, Dict
from game import Board


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
        raise NotImplementedError

    def evaluate(self, board: Board) -> float:
        """
        Evaluate board state using weighted heuristic features.

        Args:
            board: Board state as Board instance.

        Returns:
            Heuristic score H(s) = Σ w_i * f_i.
        """
        raise NotImplementedError

    def get_weights(self) -> Dict[str, float]:
        """
        Get current heuristic weights.

        Returns:
            Dictionary of current weights.
        """
        raise NotImplementedError

    def set_weights(self, weights: Dict[str, float]) -> None:
        """
        Update heuristic weights (e.g., from GA optimization).

        Args:
            weights: Dictionary mapping feature names to new weights.
        """
        raise NotImplementedError

