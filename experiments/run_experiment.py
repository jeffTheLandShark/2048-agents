"""
Headless Experiment Runner for 2048 Agents.

This script facilitates the execution of large-scale experiments to benchmark agent performance.
It supports running multiple games in parallel using multiprocessing, collecting detailed logs,
and configuring experiments via JSON files.

Usage:
    python experiments/run_experiment.py

Configuration is loaded from 'experiments/config.json'.
"""

import sys
import json
import multiprocessing as mp
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to Python path so imports work regardless of where script is run from
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from game_2048.game_env import GameEnv
from game_2048 import encode_board_log2
from agents import Agent
from agents.mcts import MCTSAgent
from agents.expectimax import ExpectimaxAgent
from agents.random_agent import RandomAgent
from stats_logging import StatsLogger, GameSummary
from heuristics.features import (
    compute_tile_counts,
    compute_all_features,
)


def _create_agent_from_config(agent_type: str, agent_config: Dict[str, Any]) -> Agent:
    """
    Create an agent instance from configuration.

    Args:
        agent_type: Type of agent ("expectimax", "mcts", "random").
        agent_config: Configuration dictionary for the agent.

    Returns:
        Agent instance.
    """
    if agent_type == "expectimax":
        return ExpectimaxAgent(**agent_config)
    elif agent_type == "mcts":
        return MCTSAgent(**agent_config)
    elif agent_type == "random":
        return RandomAgent(**agent_config)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def _run_game_core(
    agent: Agent,
    env: GameEnv,
    collect_logs: bool = False,
) -> tuple[GameSummary, list[Dict[str, Any]]]:
    """
    Core game execution logic that runs a single game.

    Args:
        agent: Agent instance to use.
        env: GameEnv instance (will be reset).
        collect_logs: If True, collects step logs with raw Board objects; if False, returns empty list.

    Returns:
        Tuple of (summary, steps_list) where:
        - summary: GameSummary dictionary
        - steps_list: List of step dictionaries with raw Board objects (empty if collect_logs=False)
          Each step dict contains: t, board (Board object), action, reward, score, tile_counts, heuristics, done
    """
    env.reset()
    board = env.get_board()
    steps = []

    # Log initial state
    tile_counts = compute_tile_counts(board)
    heuristics = compute_all_features(board)

    if collect_logs:
        steps.append({
            "t": 0,
            "board": board,  # Raw Board object
            "action": None,
            "reward": 0.0,
            "score": 0,
            "tile_counts": tile_counts,
            "heuristics": heuristics,
            "done": False,
        })

    # Run game
    step = 1
    while not env.is_game_over():
        legal_moves = env.legal_moves()
        if not legal_moves:
            break

        action = agent.choose_action(board, legal_moves)
        board, reward, done, info = env.step(action)

        tile_counts = compute_tile_counts(board)
        heuristics = compute_all_features(board)

        if collect_logs:
            steps.append({
                "t": step,
                "board": board,  # Raw Board object
                "action": action,
                "reward": reward,
                "score": info["score"],
                "tile_counts": tile_counts,
                "heuristics": heuristics,
                "done": done,
            })
        step += 1

    # Create summary
    summary = {
        "final_score": env.get_score(),
        "highest_tile": int(board.array.max()),
        "game_length": env.get_move_count(),
        "final_tile_counts": compute_tile_counts(board),
        "final_heuristics": compute_all_features(board),
    }

    return summary, steps


def _run_single_game_worker(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker function for multiprocessing that runs a single game and returns the full game log.

    Args:
        args: Dictionary containing:
            - agent_type: Type of agent
            - agent_config: Agent configuration
            - game_id: Game identifier
            - seed: Random seed
            - board_size: Board size
            - agent_name: Agent name for logging
            - config: Full experiment config

    Returns:
        Complete GameLog dictionary ready to be written to file.
    """
    # Import here to ensure proper module initialization in worker processes
    from game_2048.game_env import GameEnv
    from game_2048 import encode_board_log2
    from agents.mcts import MCTSAgent
    from agents.expectimax import ExpectimaxAgent
    from agents.random_agent import RandomAgent

    agent_type = args["agent_type"]
    agent_config = args["agent_config"]
    game_id = args["game_id"]
    seed = args["seed"]
    board_size = args["board_size"]
    agent_name = args["agent_name"]
    config = args.get("config", {})

    # Create agent
    if agent_type == "expectimax":
        agent = ExpectimaxAgent(**agent_config)
    elif agent_type == "mcts":
        agent = MCTSAgent(**agent_config)
    elif agent_type == "random":
        agent = RandomAgent(**agent_config)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    # Create environment and run game
    env = GameEnv(seed=seed, board_size=board_size)
    summary, steps = _run_game_core(agent, env, collect_logs=True)

    # Encode boards for JSON serialization
    encoded_steps = []
    for step in steps:
        encoded_step = step.copy()
        encoded_step["board"] = encode_board_log2(step["board"])
        encoded_steps.append(encoded_step)

    # Return complete game log
    return {
        "game_id": game_id,
        "agent": agent_name,
        "board_size": board_size,
        "seed": seed,
        "config": config,
        "steps": encoded_steps,
        "summary": summary,
    }


def run_single_game(
    agent: Agent,
    env: GameEnv,
    logger: Optional[StatsLogger] = None,
    game_id: Optional[str] = None,
    seed: Optional[int] = None,
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
    if logger:
        logger.start_game(game_id=game_id, seed=seed)
        # Run game and collect logs for StatsLogger
        summary, steps = _run_game_core(agent, env, collect_logs=True)

        # Write logs to StatsLogger (boards are already Board objects, no encoding needed)
        for step_log in steps:
            logger.log_step(
                step_log["t"],
                step_log["board"],  # Raw Board object
                step_log["action"],
                step_log["reward"],
                step_log["score"],
                step_log["tile_counts"],
                step_log["heuristics"],
                step_log["done"],
            )
        logger.end_game(summary)
    else:
        # Run game without collecting logs
        summary, _ = _run_game_core(agent, env, collect_logs=False)

    return summary

def run_experiment(config: dict):
    """
    Run an experiment with multiple games using multiprocessing.

    Args:
        config: Experiment configuration dictionary containing:
            - agent: Dict with "type" and "config"
            - num_games: Number of games to run
            - seed: Base seed (or None)
            - board_size: Board size (default 4)
            - logging: Dict with "log_dir" and "log_format"
            - num_workers: Number of worker processes (None for sequential)
    """
    agent_type = config["agent"]["type"]
    agent_config = config["agent"]["config"]
    num_games = config["num_games"]
    base_seed = config.get("seed")
    board_size = config.get("board_size", 4)
    num_workers = config.get("num_workers")
    # Resolve log_dir relative to project root, not current working directory
    log_dir = _project_root / config["logging"]["log_dir"]
    log_format = config["logging"].get("log_format", "jsonl")
    experiment_name = config.get("experiment_name", agent_type)  # Use experiment_name or fallback to agent_type

    # Determine log file path based on experiment name
    if log_format == "jsonl":
        log_file = log_dir / f"{experiment_name}.jsonl"
    else:
        log_file = log_dir / f"{experiment_name}.{log_format}"

    # Prepare arguments for each game
    game_args = []
    for i in range(num_games):
        game_seed = base_seed + i if base_seed is not None else None
        game_id = f"{agent_type}_game_{i}"
        game_args.append({
            "agent_type": agent_type,
            "agent_config": agent_config,
            "game_id": game_id,
            "seed": game_seed,
            "board_size": board_size,
            "agent_name": agent_type,
            "config": agent_config,  # Just the agent config, not the full agent dict
        })

    # Run games with multiprocessing or sequentially
    if num_workers and num_workers > 1:
        # Use multiprocessing
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=num_workers) as pool:
            game_logs = []
            completed = 0
            for game_log in pool.imap(_run_single_game_worker, game_args):
                game_logs.append(game_log)
                completed += 1
                summary = game_log["summary"]
                print(
                    f"Game {completed}/{num_games} ({game_log['game_id']}) completed: "
                    f"score={summary['final_score']}, "
                    f"highest_tile={summary['highest_tile']}, "
                    f"length={summary['game_length']} moves"
                )

            # Write all logs sequentially to file
            print(f"Writing {len(game_logs)} game logs to {log_file}...")
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "a", encoding="utf-8") as f:
                for game_log in game_logs:
                    json.dump(game_log, f)
                    f.write("\n")
            print(f"Experiment complete! Logs written to {log_file}")
    else:
        # Sequential execution
        print(f"Running {num_games} games sequentially...")
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger = StatsLogger(log_file, agent_type, board_size, agent_config)
        agent = _create_agent_from_config(agent_type, agent_config)

        for i in range(num_games):
            game_seed = base_seed + i if base_seed is not None else None
            env = GameEnv(seed=game_seed, board_size=board_size)
            summary = run_single_game(
                agent,
                env,
                logger=logger,
                seed=game_seed,
            )
            print(
                f"Game {i + 1}/{num_games} ({game_id}) completed: "
                f"score={summary['final_score']}, "
                f"highest_tile={summary['highest_tile']}, "
                f"length={summary['game_length']} moves"
            )

        logger.close()
        print(f"Experiment complete! Logs written to {log_file}")

def main():
    """Load config.json and run the experiment."""
    # # agent = MCTSAgent(num_simulations=100, rollout_policy="heuristic", depth_limit=20)
    # agent = MCTSAgent(num_simulations=100, depth_limit=20)
    # stats_path = "data/raw_logs/mcts_batch.jsonl"
    # logger = StatsLogger(Path(stats_path), "mcts", 4)
    # print("Running MCTS agent for 10 games...")
    # print(f"Logging to {stats_path}")

    # for i in range(10):  # Run 10 games
    #     print(f"Starting game {i}...")
    #     env = GameEnv(seed=42 + i)

    #     summary = run_single_game(agent, env, logger=logger, seed=42 + i)
    #     print(f"Game {i} summary: {summary}")
    # logger.close()
    config_path = Path(__file__).parent / "config.json"

    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    print(f"Running experiment: {config.get('experiment_name', 'unnamed')}")
    print(f"Agent: {config['agent']['type']}")
    print(f"Games: {config['num_games']}")
    print(f"Workers: {config.get('num_workers', 'sequential')}")
    print()

    run_experiment(config)


if __name__ == "__main__":
    main()
