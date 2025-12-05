"""Experiment runner for executing multiple games with an agent."""

from typing import Optional, Dict, Any
from pathlib import Path
from game import GameEnv
from agents import Agent
from logging import StatsLogger, GameSummary, ExperimentSummary


def run_experiment(
    agent: Agent,
    env: GameEnv,
    num_games: int = 100,
    log_file: Optional[Path] = None,
    agent_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None
) -> ExperimentSummary:
    """
    Run an experiment: execute N games with an agent and optionally log results.

    Args:
        agent: Agent instance to use for gameplay.
        env: GameEnv instance for game logic.
        num_games: Number of games to play.
        log_file: Optional path to JSONL log file. If None, no logging.
        agent_name: Optional agent name for logging. If None, uses agent class name.
        config: Optional configuration dictionary to include in logs.
        seed: Optional random seed for reproducibility.

    Returns:
        ExperimentSummary with experiment summary statistics:
        - mean_score: Average final score
        - std_score: Standard deviation of scores
        - mean_highest_tile: Average highest tile reached
        - games_reached_2048: Number of games reaching 2048
        - etc.
    """
    raise NotImplementedError


def run_single_game(
    agent: Agent,
    env: GameEnv,
    logger: Optional[StatsLogger] = None,
    game_id: Optional[str] = None,
    seed: Optional[int] = None
) -> GameSummary:
    """
    Run a single game with an agent.

    Args:
        agent: Agent instance to use.
        env: GameEnv instance.
        logger: Optional StatsLogger for logging game data.
        game_id: Optional unique identifier for this game.
        seed: Optional random seed for this game.

    Returns:
        GameSummary with game summary:
        - final_score: Final game score
        - highest_tile: Highest tile reached
        - game_length: Number of steps
        - final_tile_counts: Final tile distribution
        - final_heuristics: Final heuristic values
    """
    raise NotImplementedError

