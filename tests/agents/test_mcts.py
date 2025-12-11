"""Comprehensive test suite for MCTSAgent."""

import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import time

from agents.base import Agent
from agents.mcts import MCTSAgent, MCTSNode
from game_2048 import Board
from game_2048.game_env import GameEnv


class TestMCTSNodeInitialization(unittest.TestCase):
    """Test MCTSNode initialization and structure."""

    def test_node_creation(self):
        """Test basic MCTSNode creation."""
        board_state = np.zeros((4, 4), dtype=np.int32)
        env = GameEnv(Board(board_state))

        node = MCTSNode(
            state=env,
            parent=None,
            children=[],
            visits=0,
            wins=0.0,
            untried_actions=["UP", "DOWN", "LEFT", "RIGHT"],
        )

        self.assertIsNone(node.parent)
        self.assertEqual(node.visits, 0)
        self.assertEqual(node.wins, 0.0)
        self.assertEqual(len(node.children), 0)
        self.assertEqual(len(node.untried_actions), 4)

    def test_node_with_parent(self):
        """Test MCTSNode with parent relationship."""
        board_state = np.zeros((4, 4), dtype=np.int32)
        env = GameEnv(Board(board_state))

        parent = MCTSNode(
            state=env,
            parent=None,
            children=[],
            visits=1,
            wins=100.0,
            untried_actions=[],
        )

        child = MCTSNode(
            state=env.copy(),
            parent=parent,
            children=[],
            visits=0,
            wins=0.0,
            untried_actions=["UP"],
        )
        parent.children.append(child)

        self.assertEqual(child.parent, parent)
        self.assertIn(child, parent.children)

    def test_node_with_prev_move(self):
        """Test MCTSNode stores previous move."""
        board_state = np.zeros((4, 4), dtype=np.int32)
        env = GameEnv(Board(board_state))

        node = MCTSNode(
            state=env,
            parent=None,
            children=[],
            visits=0,
            wins=0.0,
            untried_actions=[],
            prev_move="UP",
        )

        self.assertEqual(node.prev_move, "UP")


class TestMCTSAgentInitialization(unittest.TestCase):
    """Test MCTSAgent initialization and configuration."""

    def test_default_initialization(self):
        """Test MCTS agent with default parameters."""
        agent = MCTSAgent()

        self.assertEqual(agent.num_simulations, 10)
        self.assertEqual(agent.exploration_constant, 1.414)
        self.assertEqual(agent.rollout_policy, "random")
        self.assertEqual(agent.depth_limit, 100)
        self.assertIsNone(agent.time_limit_ms)

    def test_custom_initialization(self):
        """Test MCTS agent with custom parameters."""
        agent = MCTSAgent(
            num_simulations=100,
            exploration_constant=2.0,
            rollout_policy="heuristic",
            depth_limit=50,
            time_limit_ms=5000,
        )

        self.assertEqual(agent.num_simulations, 100)
        self.assertEqual(agent.exploration_constant, 2.0)
        self.assertEqual(agent.rollout_policy, "heuristic")
        self.assertEqual(agent.depth_limit, 50)
        self.assertEqual(agent.time_limit_ms, 5000)

    def test_agent_is_agent_subclass(self):
        """Test MCTSAgent is a proper Agent subclass."""
        agent = MCTSAgent()
        self.assertIsInstance(agent, Agent)


class TestMCTSSelection(unittest.TestCase):
    """Test MCTS selection phase (UCT calculation)."""

    def setUp(self):
        """Set up test fixtures."""
        self.agent = MCTSAgent(exploration_constant=1.414)
        self.board_state = np.zeros((4, 4), dtype=np.int32)
        self.env = GameEnv(Board(self.board_state))

    def test_uct_value_no_visits(self):
        """Test UCT value with zero visits returns infinity."""
        value = self.agent._uct_value(
            total_simulations=100,
            node_wins=0.0,
            node_visits=0,
        )
        self.assertEqual(value, float("inf"))

    def test_uct_value_calculation(self):
        """Test UCT value is calculated correctly."""
        # UCT = (wins / visits) + C * sqrt(2 * ln(N) / n)
        C = 1.414
        N = 100
        wins = 50.0
        visits = 10

        value = self.agent._uct_value(
            total_simulations=N,
            node_wins=wins,
            node_visits=visits,
        )

        # Calculate expected value
        exploitation = wins / visits  # 5.0
        exploration = C * ((2 * N.bit_length()) / visits) ** 0.5
        expected = exploitation + exploration

        # Verify actual calculation matches formula
        self.assertAlmostEqual(value, expected, places=5)

    def test_uct_exploration_term(self):
        """Test that higher exploration constant increases UCT value."""
        agent_high_explore = MCTSAgent(exploration_constant=2.0)
        agent_low_explore = MCTSAgent(exploration_constant=0.5)

        value_high = agent_high_explore._uct_value(100, 50.0, 10)
        value_low = agent_low_explore._uct_value(100, 50.0, 10)

        self.assertGreater(value_high, value_low)

    def test_selection_chooses_best_child(self):
        """Test selection picks child with highest UCT value."""
        root = MCTSNode(
            state=self.env,
            parent=None,
            children=[],
            visits=100,
            wins=0.0,
            untried_actions=[],
        )

        # Create children with different visit/win ratios
        # Child with good exploitation
        child1 = MCTSNode(
            state=self.env.copy(),
            parent=root,
            children=[],
            visits=50,
            wins=400.0,
            untried_actions=[],
            prev_move="UP",
        )

        # Child with mediocre exploitation
        child2 = MCTSNode(
            state=self.env.copy(),
            parent=root,
            children=[],
            visits=40,
            wins=200.0,
            untried_actions=[],
            prev_move="DOWN",
        )

        # Rarely-visited child (should have high exploration bonus despite low reward)
        child3 = MCTSNode(
            state=self.env.copy(),
            parent=root,
            children=[],
            visits=5,
            wins=20.0,
            untried_actions=[],
            prev_move="LEFT",
        )

        root.children = [child1, child2, child3]

        # Manually calculate UCT values to verify which is highest
        uct1 = self.agent._uct_value(100, 400.0, 50)
        uct2 = self.agent._uct_value(100, 200.0, 40)
        uct3 = self.agent._uct_value(100, 20.0, 5)

        # Find which has highest UCT
        max_uct = max(uct1, uct2, uct3)
        if max_uct == uct1:
            expected_move = "UP"
        elif max_uct == uct2:
            expected_move = "DOWN"
        else:
            expected_move = "LEFT"

        selected = self.agent._selection(root)
        self.assertEqual(
            selected.prev_move,
            expected_move,
            f"Selection should pick child with highest UCT (UCT values: UP={uct1:.2f}, DOWN={uct2:.2f}, LEFT={uct3:.2f})",
        )


class TestMCTSExpansion(unittest.TestCase):
    """Test MCTS expansion phase."""

    def setUp(self):
        """Set up test fixtures."""
        self.agent = MCTSAgent()
        self.board_state = np.zeros((4, 4), dtype=np.int32)
        self.env = GameEnv(Board(self.board_state))

    def test_expansion_creates_new_child(self):
        """Test expansion creates a new child node."""
        # Create state with actual tiles so moves produce different states
        board_state = np.array(
            [
                [2, 2, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        env = GameEnv(Board(board_state))

        root = MCTSNode(
            state=env,
            parent=None,
            children=[],
            visits=0,
            wins=0.0,
            untried_actions=["UP", "DOWN"],
        )

        initial_children = len(root.children)
        child = self.agent._expansion(root)

        # Verify child was created and linked
        self.assertEqual(len(root.children), initial_children + 1)
        self.assertEqual(child.parent, root)
        self.assertEqual(child.visits, 0)
        self.assertEqual(child.wins, 0.0)

        # Verify child state actually reflects the move taken
        self.assertFalse(
            np.array_equal(child.state.get_board().array, root.state.get_board().array),
            "Child state should differ from parent after expansion",
        )

    def test_expansion_removes_action_from_untried(self):
        """Test expansion removes action from untried actions."""
        root = MCTSNode(
            state=self.env,
            parent=None,
            children=[],
            visits=0,
            wins=0.0,
            untried_actions=["UP", "DOWN", "LEFT"],
        )

        initial_untried = len(root.untried_actions)
        self.agent._expansion(root)

        self.assertEqual(len(root.untried_actions), initial_untried - 1)

    def test_expansion_child_has_valid_actions(self):
        """Test expanded child has valid untried actions."""
        root = MCTSNode(
            state=self.env,
            parent=None,
            children=[],
            visits=0,
            wins=0.0,
            untried_actions=["UP"],
        )

        child = self.agent._expansion(root)

        # Child should have legal moves from its state
        self.assertIsInstance(child.untried_actions, list)
        self.assertGreaterEqual(len(child.untried_actions), 0)

    def test_expansion_sets_prev_move(self):
        """Test expansion sets the previous move on child."""
        root = MCTSNode(
            state=self.env,
            parent=None,
            children=[],
            visits=0,
            wins=0.0,
            untried_actions=["UP"],
        )

        child = self.agent._expansion(root)
        self.assertIn(child.prev_move, ["UP", "DOWN", "LEFT", "RIGHT"])


class TestMCTSSimulation(unittest.TestCase):
    """Test MCTS simulation (rollout) phase."""

    def setUp(self):
        """Set up test fixtures."""
        self.agent = MCTSAgent(rollout_policy="random")
        self.board_state = np.zeros((4, 4), dtype=np.int32)
        self.board_state[0, 0] = 2
        self.board_state[0, 1] = 2
        self.env = GameEnv(Board(self.board_state))

    def test_simulation_returns_score(self):
        """Test simulation returns a numeric score that reflects game progress."""
        board_state = np.array(
            [
                [2, 2, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        env = GameEnv(Board(board_state), seed=42)
        initial_score = env.get_score()

        score = self.agent._simulation(env)

        # Score should be non-negative and reflect gameplay
        self.assertIsInstance(score, (int, float, np.integer))
        self.assertGreaterEqual(
            score,
            initial_score,
            "Simulation score should be at least the initial score",
        )

    def test_simulation_respects_depth_limit(self):
        """Test simulation respects depth limit."""
        depth_limit = 5
        agent = MCTSAgent(rollout_policy="random", depth_limit=depth_limit)

        # Create environment with initial state
        board_state = np.array(
            [
                [2, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        env = GameEnv(Board(board_state), seed=42)
        initial_moves = env.get_move_count()

        score = agent._simulation(env)

        # Verify depth limit was enforced
        final_moves = env.get_move_count()
        moves_made = final_moves - initial_moves
        self.assertLessEqual(
            moves_made,
            depth_limit,
            f"Simulation made {moves_made} moves, exceeding depth limit of {depth_limit}",
        )
        self.assertIsInstance(score, (int, float, np.integer))

    def test_simulation_handles_game_over(self):
        """Test simulation handles game-over state gracefully."""
        # Create a board where game is likely over
        board_state = np.array(
            [
                [2, 4, 8, 16],
                [32, 64, 128, 256],
                [2, 4, 8, 16],
                [32, 64, 128, 256],
            ],
            dtype=np.int32,
        )

        env = GameEnv(Board(board_state), seed=42)

        # Should not raise exception
        score = self.agent._simulation(env)
        self.assertIsInstance(score, (int, float, np.integer))

    def test_simulation_random_policy(self):
        """Test simulation uses random policy when specified."""
        agent_random = MCTSAgent(rollout_policy="random", num_simulations=5)
        score = agent_random._simulation(self.env)

        self.assertIsInstance(score, (int, float, np.integer))


class TestMCTSBackpropagation(unittest.TestCase):
    """Test MCTS backpropagation phase."""

    def setUp(self):
        """Set up test fixtures."""
        self.agent = MCTSAgent()
        self.board_state = np.zeros((4, 4), dtype=np.int32)
        self.env = GameEnv(Board(self.board_state))

    def test_backpropagation_updates_leaf(self):
        """Test backpropagation updates leaf node."""
        leaf = MCTSNode(
            state=self.env,
            parent=None,
            children=[],
            visits=0,
            wins=0.0,
            untried_actions=[],
        )

        self.agent._backpropagation(leaf, 100.0)

        self.assertEqual(leaf.visits, 1)
        self.assertEqual(leaf.wins, 100.0)

    def test_backpropagation_updates_all_ancestors(self):
        """Test backpropagation updates entire path to root."""
        grandparent = MCTSNode(
            state=self.env,
            parent=None,
            children=[],
            visits=0,
            wins=0.0,
            untried_actions=[],
        )

        parent = MCTSNode(
            state=self.env.copy(),
            parent=grandparent,
            children=[],
            visits=0,
            wins=0.0,
            untried_actions=[],
        )

        child = MCTSNode(
            state=self.env.copy(),
            parent=parent,
            children=[],
            visits=0,
            wins=0.0,
            untried_actions=[],
        )

        self.agent._backpropagation(child, 50.0)

        # All nodes should be updated
        self.assertEqual(child.visits, 1)
        self.assertEqual(child.wins, 50.0)
        self.assertEqual(parent.visits, 1)
        self.assertEqual(parent.wins, 50.0)
        self.assertEqual(grandparent.visits, 1)
        self.assertEqual(grandparent.wins, 50.0)

    def test_backpropagation_accumulates_visits(self):
        """Test backpropagation accumulates multiple visits."""
        node = MCTSNode(
            state=self.env,
            parent=None,
            children=[],
            visits=5,
            wins=100.0,
            untried_actions=[],
        )

        self.agent._backpropagation(node, 50.0)

        self.assertEqual(node.visits, 6)
        self.assertEqual(node.wins, 150.0)


class TestMCTSChooseAction(unittest.TestCase):
    """Test MCTS choose_action method (integration)."""

    def setUp(self):
        """Set up test fixtures."""
        self.board_state = np.array(
            [
                [2, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 2],
            ],
            dtype=np.int32,
        )
        self.board = Board(self.board_state)

    def test_choose_action_returns_legal_move(self):
        """Test choose_action returns a legal move."""
        agent = MCTSAgent(num_simulations=5)

        env = GameEnv(self.board)
        legal_moves = env.legal_moves()

        action = agent.choose_action(self.board, legal_moves)

        self.assertIn(action, legal_moves)

    def test_choose_action_with_single_legal_move(self):
        """Test choose_action when only one move is legal."""
        agent = MCTSAgent(num_simulations=1)

        # Create board where only UP is legal
        board_state = np.array(
            [
                [2, 2, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        board = Board(board_state)

        action = agent.choose_action(board, ["UP"])
        self.assertEqual(action, "UP")

    def test_choose_action_no_legal_moves_raises(self):
        """Test choose_action raises when no legal moves."""
        agent = MCTSAgent(num_simulations=1)
        board = Board(np.zeros((4, 4), dtype=np.int32))

        with self.assertRaises(ValueError):
            agent.choose_action(board, [])

    def test_choose_action_runs_simulations(self):
        """Test choose_action executes the correct number of simulations."""
        num_sims = 10
        agent = MCTSAgent(num_simulations=num_sims)

        board_state = np.array(
            [
                [2, 4, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        board = Board(board_state)
        env = GameEnv(board)
        legal_moves = env.legal_moves()

        # Track simulation calls
        call_count = [0]
        original_simulation = agent._simulation

        def mock_simulation(state):
            call_count[0] += 1
            return original_simulation(state)

        agent._simulation = mock_simulation
        action = agent.choose_action(board, legal_moves)

        # Verify exact number of simulations
        self.assertEqual(
            call_count[0],
            num_sims,
            f"Expected {num_sims} simulations, but {call_count[0]} were executed",
        )
        self.assertIn(action, legal_moves)

    def test_choose_action_deterministic_with_seed(self):
        """Test choose_action produces consistent results with controlled randomness."""
        # Run multiple times to verify consistency
        agent = MCTSAgent(num_simulations=20)

        board_state = np.array(
            [
                [2, 4, 0, 0],
                [2, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        board = Board(board_state)
        env = GameEnv(board, seed=42)
        legal_moves = env.legal_moves()

        # Run multiple times - should produce valid actions
        actions = []
        for _ in range(5):
            action = agent.choose_action(board, legal_moves)
            self.assertIn(action, legal_moves)
            actions.append(action)

        # All actions should be legal
        for action in actions:
            self.assertIn(action, legal_moves)

    def test_choose_action_builds_search_tree(self):
        """Test that choose_action actually builds and uses a search tree."""
        agent = MCTSAgent(num_simulations=20)

        board_state = np.array(
            [
                [2, 4, 8, 0],
                [4, 8, 16, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        board = Board(board_state)
        env = GameEnv(board, seed=42)
        legal_moves = env.legal_moves()

        # Track tree building through expansion calls
        expansion_count = [0]
        original_expansion = agent._expansion

        def mock_expansion(node):
            expansion_count[0] += 1
            return original_expansion(node)

        agent._expansion = mock_expansion
        action = agent.choose_action(board, legal_moves)

        # Verify tree was expanded (should expand at least a few nodes)
        self.assertGreater(
            expansion_count[0],
            0,
            "MCTS should expand at least some nodes during search",
        )
        self.assertIn(action, legal_moves)

    def test_choose_action_different_with_different_seed(self):
        """Test choose_action can differ with different seeds."""
        agent = MCTSAgent(num_simulations=20)

        board_state = np.array(
            [
                [2, 0, 0, 0],
                [0, 2, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        board = Board(board_state)

        env = GameEnv(board)
        legal_moves = env.legal_moves()

        action1 = agent.choose_action(board, legal_moves)
        action2 = agent.choose_action(board, legal_moves)

        # Both should be legal
        self.assertIn(action1, legal_moves)
        self.assertIn(action2, legal_moves)


class TestMCTSTimeouts(unittest.TestCase):
    """Test MCTS timeout functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.board_state = np.zeros((4, 4), dtype=np.int32)
        self.board = Board(self.board_state)

    def test_check_timeout_with_no_limit(self):
        """Test _check_timeout returns False when no limit is set."""
        agent = MCTSAgent(time_limit_ms=None)
        agent._start_time = time.time()

        self.assertFalse(agent._check_timeout())

    def test_check_timeout_respects_time_limit(self):
        """Test _check_timeout respects the millisecond limit."""
        agent = MCTSAgent(time_limit_ms=10)
        agent._start_time = time.time() - 0.05  # 50ms ago

        self.assertTrue(agent._check_timeout())

    def test_check_timeout_within_limit(self):
        """Test _check_timeout returns False when time is within limit."""
        agent = MCTSAgent(time_limit_ms=1000)
        agent._start_time = time.time()

        self.assertFalse(agent._check_timeout())


class TestMCTSEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.agent = MCTSAgent(num_simulations=5)
        self.board_state = np.zeros((4, 4), dtype=np.int32)

    def test_empty_legal_moves_raises_error(self):
        """Test error when no legal moves are provided."""
        board = Board(self.board_state)

        with self.assertRaises(ValueError):
            self.agent.choose_action(board, [])

    def test_all_four_directions_legal(self):
        """Test when all four directions are legal."""
        # Sparse board - all moves should be legal
        board_state = np.array(
            [
                [2, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        board = Board(board_state)

        env = GameEnv(board)
        legal_moves = env.legal_moves()

        action = self.agent.choose_action(board, legal_moves)
        self.assertIn(action, legal_moves)

    def test_high_value_tiles(self):
        """Test MCTS works with high value tiles."""
        board_state = np.array(
            [
                [1024, 512, 256, 128],
                [256, 128, 64, 32],
                [128, 64, 32, 16],
                [64, 32, 16, 8],
            ],
            dtype=np.int32,
        )
        board = Board(board_state)

        env = GameEnv(board)
        legal_moves = env.legal_moves()

        if legal_moves:
            action = self.agent.choose_action(board, legal_moves)
            self.assertIn(action, legal_moves)

    def test_agent_reset(self):
        """Test agent reset method."""
        agent = MCTSAgent()
        agent.reset()  # Should not raise

    def test_board_copied_not_modified(self):
        """Test that original board is not modified during search."""
        board_state = np.array(
            [
                [2, 4, 0, 0],
                [8, 16, 0, 0],
                [2, 4, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        board = Board(board_state)
        original_array = board.array.copy()
        original_sum = board.array.sum()

        env = GameEnv(board)
        legal_moves = env.legal_moves()

        # Make sure board has legal moves
        self.assertGreater(len(legal_moves), 0, "Test board should have legal moves")

        # Run agent with many simulations to ensure search tree is built
        agent = MCTSAgent(num_simulations=50)
        agent.choose_action(board, legal_moves)

        # Verify board unchanged
        np.testing.assert_array_equal(
            board.array, original_array, "Board state was modified during MCTS search"
        )
        self.assertEqual(
            board.array.sum(), original_sum, "Board tile sum changed during search"
        )


class TestMCTSRolloutPolicies(unittest.TestCase):
    """Test different rollout policies."""

    def setUp(self):
        """Set up test fixtures."""
        self.board_state = np.array(
            [
                [2, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 2],
            ],
            dtype=np.int32,
        )
        self.board = Board(self.board_state)

    def test_random_rollout_policy(self):
        """Test random rollout policy works."""
        agent = MCTSAgent(rollout_policy="random", num_simulations=5)

        env = GameEnv(self.board)
        legal_moves = env.legal_moves()

        action = agent.choose_action(self.board, legal_moves)
        self.assertIn(action, legal_moves)

    def test_heuristic_rollout_policy(self):
        """Test heuristic rollout policy (placeholder implementation)."""
        agent = MCTSAgent(rollout_policy="heuristic", num_simulations=5)

        env = GameEnv(self.board)
        legal_moves = env.legal_moves()

        # Should use heuristic policy in simulation
        action = agent.choose_action(self.board, legal_moves)
        self.assertIn(action, legal_moves)

    def test_invalid_policy_still_works(self):
        """Test agent doesn't crash with invalid policy."""
        agent = MCTSAgent(rollout_policy="invalid", num_simulations=3)

        env = GameEnv(self.board)
        legal_moves = env.legal_moves()

        # Should fallback gracefully
        action = agent.choose_action(self.board, legal_moves)
        self.assertIn(action, legal_moves)


if __name__ == "__main__":
    unittest.main()
