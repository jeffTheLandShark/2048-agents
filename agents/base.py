"""Base agent interface for 2048 AI agents."""

from abc import ABC, abstractmethod
from typing import List, Any
from game import Board


class Agent(ABC):
    """
    Abstract base class for all 2048 agents.

    All agents must implement choose_action to select moves based on game state.
    """

    @abstractmethod
    def choose_action(self, state: Board, legal_moves: List[str]) -> str:
        """
        Choose an action given the current game state.

        Args:
            state: Current board state as Board instance.
            legal_moves: List of legal action strings ("UP", "DOWN", "LEFT", "RIGHT").

        Returns:
            Selected action string (must be in legal_moves).

        Raises:
            ValueError: If returned action is not in legal_moves.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        Reset agent state between games.

        Called at the start of each new game to clear any internal state.
        """
        pass

