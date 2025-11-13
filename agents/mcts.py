"""Monte-Carlo Tree Search (MCTS) agent implementation for 2048."""

from typing import List, Optional
from agents import Agent
from game import Board


class MCTSAgent(Agent):
    """
    MCTS agent using UCT (Upper Confidence Bound for Trees) algorithm.

    Supports random rollouts or heuristic-guided rollouts.
    """

    def __init__(
        self,
        num_simulations: int = 1000,
        exploration_constant: float = 1.414,
        rollout_policy: str = "random",
        time_limit_ms: Optional[int] = None
    ) -> None:
        """
        Initialize MCTS agent.

        Args:
            num_simulations: Number of MCTS simulations per move.
            exploration_constant: UCT exploration constant (C in UCB1 formula).
            rollout_policy: Policy for rollouts ("random" or "heuristic").
            time_limit_ms: Optional time limit per move in milliseconds.
        """
        raise NotImplementedError

    def choose_action(self, state: Board, legal_moves: List[str]) -> str:
        """
        Choose action using MCTS search.

        Args:
            state: Current board state as Board instance.
            legal_moves: List of legal action strings.

        Returns:
            Best action according to MCTS search.
        """
        raise NotImplementedError

