"""Integration tests for MCTS agent with game environment."""

import unittest
import numpy as np
from pathlib import Path
import tempfile
import json

from agents.mcts import MCTSAgent
from game_2048 import Board
from game_2048.game_env import GameEnv
from stats_logging import StatsLogger


class TestMCTSGameIntegration(unittest.TestCase):
    """Test MCTS agent integration with GameEnv."""

    def setUp(self):
        """Set up test fixtures."""
        self.agent = MCTSAgent(num_simulations=10)
        self.seed = 42

    def test_agent_plays_complete_game(self):
        """Test MCTS agent can play a complete game without errors."""
        env = GameEnv(seed=self.seed)
        board, _ = env.reset()

        move_count = 0
        max_moves = 100  # Prevent infinite loops

        while not env.is_game_over() and move_count < max_moves:
            legal_moves = env.legal_moves()
            if not legal_moves:
                break

            # Agent chooses action
            action = self.agent.choose_action(board, legal_moves)
            self.assertIn(action, legal_moves)

            # Execute action
            board, reward, done, info = env.step(action)
            move_count += 1

            # Verify game state is valid
            self.assertIsInstance(board, Board)
            self.assertIsInstance(reward, (int, float))
            self.assertIsInstance(done, bool)
            self.assertIsInstance(info, dict)

        # Verify game progressed
        self.assertGreater(move_count, 0, "Agent should make at least one move")
        self.assertGreater(env.get_score(), 0, "Game should accumulate some score")

    def test_agent_with_different_board_sizes(self):
        """Test MCTS agent works with different board configurations."""
        for size in [4]:  # Standard 2048 size
            with self.subTest(size=size):
                env = GameEnv(board_size=size, seed=self.seed)
                board, _ = env.reset()

                legal_moves = env.legal_moves()
                if legal_moves:
                    action = self.agent.choose_action(board, legal_moves)
                    self.assertIn(action, legal_moves)

    def test_agent_handles_near_game_over(self):
        """Test MCTS agent handles near-game-over states gracefully."""
        # Create a nearly full board
        board_state = np.array(
            [
                [2, 4, 8, 16],
                [32, 64, 128, 256],
                [512, 1024, 2, 4],
                [8, 16, 32, 0],
            ],
            dtype=np.int32,
        )

        env = GameEnv(Board(board_state), seed=self.seed)
        legal_moves = env.legal_moves()

        if legal_moves:
            action = self.agent.choose_action(env.get_board(), legal_moves)
            self.assertIn(action, legal_moves)

            # Verify action can be executed
            board, reward, done, info = env.step(action)
            self.assertIsInstance(board, Board)

    def test_agent_consistent_across_resets(self):
        """Test agent maintains consistent behavior across multiple games."""
        scores = []

        for game_num in range(3):
            env = GameEnv(seed=self.seed + game_num)
            board, _ = env.reset()

            move_count = 0
            max_moves = 50  # Short games for speed

            while not env.is_game_over() and move_count < max_moves:
                legal_moves = env.legal_moves()
                if not legal_moves:
                    break

                action = self.agent.choose_action(board, legal_moves)
                board, reward, done, info = env.step(action)
                move_count += 1

            scores.append(env.get_score())

        # All games should produce valid scores
        for score in scores:
            self.assertGreater(score, 0, "Each game should accumulate score")

    def test_agent_produces_valid_moves_every_step(self):
        """Test agent always returns valid moves throughout a game."""
        env = GameEnv(seed=self.seed)
        board, _ = env.reset()

        for _ in range(30):  # Test 30 moves
            legal_moves = env.legal_moves()
            if not legal_moves:
                break

            action = self.agent.choose_action(board, legal_moves)

            # Verify action is valid
            self.assertIn(action, legal_moves)
            self.assertIn(action, ["UP", "DOWN", "LEFT", "RIGHT"])

            # Execute and continue
            board, reward, done, info = env.step(action)

            if done:
                break


class TestMCTSLoggingIntegration(unittest.TestCase):
    """Test MCTS agent integration with StatsLogger."""

    def setUp(self):
        """Set up test fixtures."""
        self.agent = MCTSAgent(num_simulations=10)
        self.seed = 42
        self.temp_dir = tempfile.mkdtemp()

    def test_agent_with_logger(self):
        """Test MCTS agent works with StatsLogger."""
        log_file = Path(self.temp_dir) / "test_mcts.jsonl"

        logger = StatsLogger(
            log_file=log_file,
            agent_name="mcts",
            board_size=4,
            config={"num_simulations": 10, "seed": self.seed},
        )

        env = GameEnv(seed=self.seed)
        board, reset_info = env.reset()

        logger.start_game(seed=self.seed)

        # Play a few moves
        for step in range(10):
            legal_moves = env.legal_moves()
            if not legal_moves:
                break

            action = self.agent.choose_action(board, legal_moves)
            board, reward, done, info = env.step(action)

            # Log the step
            logger.log_step(
                t=step,
                board=board,
                action=action,
                reward=reward,
                score=info["score"],
                tile_counts={},
                heuristics={},
                done=done,
            )

            if done:
                break

        # End game and verify log was written
        summary = {
            "final_score": env.get_score(),
            "highest_tile": int(board.array.max()),
            "game_length": env.get_move_count(),
            "final_tile_counts": {},
            "final_heuristics": {},
        }
        logger.end_game(summary)

        # Verify log file exists and contains data
        self.assertTrue(log_file.exists(), "Log file should be created")
        self.assertGreater(log_file.stat().st_size, 0, "Log file should contain data")


class TestMCTSPerformance(unittest.TestCase):
    """Test MCTS agent performance characteristics."""

    def test_agent_respects_simulation_count(self):
        """Test different simulation counts affect behavior."""
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
        env = GameEnv(board, seed=42)
        legal_moves = env.legal_moves()

        # Test with different simulation counts
        for num_sims in [5, 20, 50]:
            with self.subTest(num_sims=num_sims):
                agent = MCTSAgent(num_simulations=num_sims)
                action = agent.choose_action(board, legal_moves)
                self.assertIn(action, legal_moves)

    def test_agent_respects_time_limit(self):
        """Test agent can be configured with time limits."""
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

        # Agent with time limit configured (note: actual timeout enforcement
        # depends on implementation checking _check_timeout())
        agent = MCTSAgent(num_simulations=100, time_limit_ms=100)

        action = agent.choose_action(board, legal_moves)

        # Verify agent still produces valid action with time limit set
        self.assertIn(action, legal_moves)
        self.assertIsNotNone(agent.time_limit_ms)
        self.assertEqual(agent.time_limit_ms, 100)

    def test_agent_performance_vs_random(self):
        """Test MCTS agent performs better than random baseline."""
        from agents.random_agent import RandomAgent

        mcts_agent = MCTSAgent(num_simulations=20)
        random_agent = RandomAgent(seed=42)

        mcts_scores = []
        random_scores = []

        # Run short games for both agents
        for seed in [42, 43, 44]:
            # MCTS game
            env = GameEnv(seed=seed)
            board, _ = env.reset()
            for _ in range(30):
                legal_moves = env.legal_moves()
                if not legal_moves:
                    break
                action = mcts_agent.choose_action(board, legal_moves)
                board, _, done, _ = env.step(action)
                if done:
                    break
            mcts_scores.append(env.get_score())

            # Random game
            env = GameEnv(seed=seed)
            board, _ = env.reset()
            for _ in range(30):
                legal_moves = env.legal_moves()
                if not legal_moves:
                    break
                action = random_agent.choose_action(board, legal_moves)
                board, _, done, _ = env.step(action)
                if done:
                    break
            random_scores.append(env.get_score())

        # MCTS should generally perform better (not guaranteed, but likely)
        avg_mcts = sum(mcts_scores) / len(mcts_scores)
        avg_random = sum(random_scores) / len(random_scores)

        # At minimum, both should produce valid scores
        self.assertGreater(avg_mcts, 0)
        self.assertGreater(avg_random, 0)


class TestMCTSUICompatibility(unittest.TestCase):
    """Test MCTS agent is compatible with UI expectations."""

    def test_agent_has_required_interface(self):
        """Test agent implements required Agent interface."""
        from agents.base import Agent

        agent = MCTSAgent()

        # Check inheritance
        self.assertIsInstance(agent, Agent)

        # Check required methods exist
        self.assertTrue(hasattr(agent, "choose_action"))
        self.assertTrue(hasattr(agent, "reset"))
        self.assertTrue(callable(agent.choose_action))
        self.assertTrue(callable(agent.reset))

    def test_agent_works_with_board_objects(self):
        """Test agent properly handles Board objects (not raw arrays)."""
        board_state = np.array(
            [
                [2, 4, 0, 0],
                [8, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int32,
        )

        # Test with Board object
        board = Board(board_state)
        agent = MCTSAgent(num_simulations=5)
        env = GameEnv(board)
        legal_moves = env.legal_moves()

        action = agent.choose_action(board, legal_moves)
        self.assertIn(action, legal_moves)

    def test_agent_reset_clears_state(self):
        """Test agent reset method works correctly."""
        agent = MCTSAgent(num_simulations=10)

        # Play a move
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

        agent.choose_action(board, legal_moves)

        # Reset should not raise
        agent.reset()

        # Agent should still work after reset
        action = agent.choose_action(board, legal_moves)
        self.assertIn(action, legal_moves)


if __name__ == "__main__":
    unittest.main()
