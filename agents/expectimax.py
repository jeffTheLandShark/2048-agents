"""Expectimax agent implementation for 2048."""

from typing import List, Dict, Optional
from agents.base import Agent


class ExpectimaxAgent(Agent):
    """
    Expectimax search agent with configurable heuristic weights.

    Uses iterative deepening and weighted heuristic evaluation.
    Supports both fixed (best-known) weights and GA-optimized weights.
    """

    def __init__(
        self,
        depth_limit: int = 5,
        time_limit_ms: Optional[int] = None,
        heuristic_weights: Optional[Dict[str, float]] = None,
        use_iterative_deepening: bool = True
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
        raise NotImplementedError

    def choose_action(self, state: List[List[int]], legal_moves: List[str]) -> str:
        """
        Choose action using Expectimax search.

        Args:
            state: Current board state as 2D list.
            legal_moves: List of legal action strings.

        Returns:
            Best action according to Expectimax search.
        """
        raise NotImplementedError

    def set_heuristic_weights(self, weights: Dict[str, float]) -> None:
        """
        Update heuristic weights (e.g., from GA optimization).

        Args:
            weights: Dictionary mapping heuristic names to weights.
        """
        raise NotImplementedError

