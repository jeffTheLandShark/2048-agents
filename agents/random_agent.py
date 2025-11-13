"""Random agent baseline for 2048 evaluation."""

from typing import List, Optional
import numpy.random as random
from agents.base import Agent


class RandomAgent(Agent):
    """
    Simple random agent that selects uniformly from legal moves.

    Used as a baseline for evaluation and comparison.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """
        Initialize random agent.

        Args:
            seed: Optional random seed for reproducibility.
        """
        raise NotImplementedError

    def choose_action(self, state: List[List[int]], legal_moves: List[str]) -> str:
        """
        Choose a random legal action.

        Args:
            state: Current board state (unused by random agent).
            legal_moves: List of legal action strings.

        Returns:
            Randomly selected action from legal_moves.
        """
        raise NotImplementedError

