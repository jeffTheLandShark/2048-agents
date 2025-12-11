"""Monte-Carlo Tree Search (MCTS) agent implementation for 2048."""

from typing import List, Optional
from dataclasses import dataclass
import random
import time
import math

from agents import Agent
from game_2048.game_env import GameEnv
from game_2048 import Board
from heuristics.evaluator import HeuristicEvaluator


@dataclass
class MCTSNode:
    state: GameEnv
    parent: Optional["MCTSNode"]
    children: List["MCTSNode"]
    visits: int
    total_reward: float
    untried_actions: List[str]
    prev_move: str = ""


class MCTSAgent(Agent):
    """
    MCTS agent using UCT (Upper Confidence Bound for Trees) algorithm.

    Supports random rollouts or heuristic-guided rollouts.
    """

    def __init__(
        self,
        num_simulations: int = 1000,
        exploration_constant: float = 1.414,  # sqrt(2)
        rollout_policy: str = "random",
        depth_limit: int = 3,
        time_limit_ms: Optional[int] = 100,
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
        self._start_time = 0.0
        self._evaluator = HeuristicEvaluator()
        self.time_limit_ms = time_limit_ms

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
            total_reward=0.0,
            untried_actions=legal_moves.copy(),
        )

        self._start_time = time.time()

        for _ in range(self.num_simulations):
            if self._check_timeout():
                break
            node = root
            node_board_env = board_env.copy()
            # Selection
            while node.untried_actions == [] and node.children != []:
                node = self._selection(node)
                node_board_env.step(node.prev_move)
            # Expansion
            if node.untried_actions:
                node = self._expansion(node)
                node_board_env.step(node.prev_move)
            # Simulation
            reward = self._simulation(node_board_env)
            # Backpropagation
            self._backpropagation(node, reward)

        # Choose the best action from the root's children (highest average reward)
        best_child = max(
            root.children,
            key=lambda c: (
                (c.total_reward / c.visits) if c.visits > 0 else float("-inf")
            ),
        )
        return best_child.prev_move

    def _selection(self, node: MCTSNode) -> MCTSNode:
        """Select a child node using UCT."""
        uct_values = [
            self._uct_value(node.visits, child.total_reward, child.visits)
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
            total_reward=0.0,
            untried_actions=new_state.legal_moves(),
            prev_move=action,
        )
        node.children.append(child_node)
        return child_node

    def _simulation(self, state: GameEnv) -> float:
        """Perform a rollout from the given state and return the reward."""
        current_state = state.copy()
        rollout_steps = 0
        while not current_state.is_game_over() and rollout_steps < self.depth_limit:
            legal_moves = current_state.legal_moves()
            if not legal_moves:
                break
            if self.rollout_policy == "heuristic":
                action = self._choose_heuristic_action(legal_moves, current_state)
            else:
                action = random.choice(legal_moves)
            current_state.step(action)
            rollout_steps += 1
        return self._evaluator.evaluate(current_state.get_board())

    def _choose_heuristic_action(self, legal_moves: List[str], state: GameEnv) -> str:
        """Choose the best action according to the heuristic evaluator."""
        best_score = float("-inf")
        best_action = legal_moves[0]
        for action in legal_moves:
            temp_state = state.copy()
            temp_state.step(action)
            score = self._evaluator.evaluate(temp_state.get_board())
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _backpropagation(self, node: MCTSNode, reward: float) -> None:
        """Backpropagate the reward up the tree."""
        next_node = node
        while next_node is not None:
            node = next_node
            node.visits += 1
            node.total_reward += reward
            next_node = node.parent if node else None

    def _uct_value(
        self, parent_visits: int, node_total_reward: float, node_visits: int
    ) -> float:
        """Calculate the UCT value for a node.
        UCT = (total_reward / visits) + C * sqrt(ln(parent_visits) / visits)
        """
        if node_visits == 0:
            return float("inf")
        exploitation = node_total_reward / node_visits
        exploration = self.exploration_constant * math.sqrt(
            math.log(parent_visits + 1) / node_visits
        )
        return exploitation + exploration

    def _check_timeout(self) -> bool:
        """Check if time limit has been exceeded."""
        if self.time_limit_ms is None:
            return False
        elapsed = (time.time() - self._start_time) * 1000
        return elapsed >= self.time_limit_ms
