"""Monte-Carlo Tree Search (MCTS) agent implementation for 2048."""

from typing import List, Optional
from dataclasses import dataclass
import random
import time

from agents import Agent
from game_2048.game_env import GameEnv
from game_2048 import Board


@dataclass
class MCTSNode:
    state: GameEnv
    parent: Optional["MCTSNode"]
    children: List["MCTSNode"]
    visits: int
    wins: float
    untried_actions: List[str]
    prev_move: str = ""


class MCTSAgent(Agent):
    """
    MCTS agent using UCT (Upper Confidence Bound for Trees) algorithm.

    Supports random rollouts or heuristic-guided rollouts.
    """

    def __init__(
        self,
        num_simulations: int = 10,
        exploration_constant: float = 1.414,  # sqrt(2)
        rollout_policy: str = "random",
        depth_limit: int = 100,
        time_limit_ms: Optional[int] = None,
    ) -> None:
        """
        Initialize MCTS agent.

        Args:
            num_simulations: Number of MCTS simulations per move.
            exploration_constant: UCT exploration constant (C in UCB1 formula).
            rollout_policy: Policy for rollouts ("random" or "heuristic").
            time_limit_ms: Optional time limit per move in milliseconds.
        """
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.rollout_policy = rollout_policy
        self.depth_limit = depth_limit
        self.time_limit_ms = time_limit_ms
        self._start_time = 0.0

    def choose_action(self, state: Board, legal_moves: List[str]) -> str:
        """
        Choose action using MCTS search.

        Args:
            state: Current board state as Board instance.
            legal_moves: List of legal action strings.

        Returns:
            Best action according to MCTS search.
        """
        if not legal_moves:
            raise ValueError("No legal moves available.")

        board_env = GameEnv(state)
        root: MCTSNode = MCTSNode(
            state=board_env,
            parent=None,
            children=[],
            visits=0,
            wins=0.0,
            untried_actions=legal_moves.copy(),
        )

        self._start_time = time.time()

        for _ in range(self.num_simulations):
            node = root
            board_env = board_env.copy()

            # Selection
            while node.untried_actions == [] and node.children != []:
                # select
                node = self._selection(node)
                # step in the environment
                board_env.step(node.prev_move)

            # Expansion
            if node.untried_actions:
                # expand
                node = self._expansion(node)
                # step in the environment
                board_env.step(node.prev_move)

            # Simulation
            reward = self._simulation(board_env)

            # Backpropagation
            self._backpropagation(node, reward)

        # Choose the best action from the root's children
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.prev_move

    def _selection(self, node: MCTSNode) -> MCTSNode:
        """Select a child node using UCT."""
        total_simulations = sum(child.visits for child in node.children)
        uct_values = [
            self._uct_value(total_simulations, child.wins, child.visits)
            for child in node.children
        ]
        best_index = uct_values.index(max(uct_values))
        return node.children[best_index]

    def _expansion(self, node: MCTSNode) -> MCTSNode:
        """Expand the node by adding a new child."""
        action = node.untried_actions.pop()
        new_state = node.state.copy()
        new_state.step(action)
        child_node = MCTSNode(
            state=new_state,
            parent=node,
            children=[],
            visits=0,
            wins=0.0,
            untried_actions=new_state.legal_moves(),
            prev_move=action,
        )
        node.children.append(child_node)
        return child_node

    def _simulation(self, state: GameEnv) -> float:
        """Perform a rollout from the given state and return the reward."""

        current_state = state.copy()
        while (
            not current_state.is_game_over()
            and current_state.get_move_count() < self.depth_limit
        ):
            legal_moves = current_state.legal_moves()
            if not legal_moves:
                break
            if self.rollout_policy == "random":
                action = random.choice(legal_moves)
            else:
                # Placeholder for heuristic policy
                action = legal_moves[0]  # Replace with actual heuristic
            current_state.step(action)

        return current_state.get_score()

    def _backpropagation(self, node: MCTSNode, reward: float) -> None:
        """Backpropagate the reward up the tree."""
        next_node = node
        while next_node is not None:
            node = next_node
            node.visits += 1
            node.wins += reward
            next_node = node.parent if node else None

    def _uct_value(
        self, total_simulations: int, node_wins: float, node_visits: int
    ) -> float:
        """Calculate the UCT value for a node.
        UCT = (wins / visits) + C * sqrt(ln(total_simulations) / visits)
        """
        if node_visits == 0:
            return float("inf")
        exploitation = node_wins / node_visits
        exploration = (
            self.exploration_constant
            * ((2 * (total_simulations).bit_length()) / node_visits) ** 0.5
        )
        return exploitation + exploration

    def _check_timeout(self) -> bool:
        """Check if time limit has been exceeded."""
        if self.time_limit_ms is None:
            return False
        elapsed = (time.time() - self._start_time) * 1000
        return elapsed >= self.time_limit_ms
