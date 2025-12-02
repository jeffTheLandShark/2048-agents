"""
Launcher script for 2048 AI Pygame UI.

Supports three modes:
1. Human Play: Play the game manually.
2. Replay: Watch a replay from a JSONL log file.
3. Agent Play: Watch an AI agent play in real-time.

Usage:
    python run_ui.py human [--log-file LOG_FILE] [--seed SEED]
    python run_ui.py replay --log-file LOG_FILE [--game-id ID] [--speed SPEED]
    python run_ui.py agent --agent {random,expectimax,mcts} [--log-file LOG_FILE] [--seed SEED] [--delay MS]
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from game import GameEnv
from game.pygame_ui import run_human_ui, run_replay_from_log, run_agent, UIConfig
from stats_logging import StatsLogger
from agents.random_agent import RandomAgent
# Import other agents as needed/available
# from agents.expectimax import ExpectimaxAgent
# from agents.mcts import MCTSAgent


def setup_logger(log_file: Optional[str], agent_name: str, seed: Optional[int]) -> Optional[StatsLogger]:
    """Helper to setup StatsLogger with default path if log_file is not provided."""
    if not log_file:
        # Default to data/raw_logs/{agent_name}.jsonl
        log_file = f"data/raw_logs/{agent_name}.jsonl"

    log_path = Path(log_file)
    print(f"Logging game to {log_path}")
    return StatsLogger(
        log_file=log_path,
        agent_name=agent_name,
        board_size=4,
        config={"seed": seed}
    )


def main():
    parser = argparse.ArgumentParser(description="2048 AI Pygame UI Launcher")
    subparsers = parser.add_subparsers(dest="mode", help="Mode to run", required=True)

    # --- Human Play Mode ---
    human_parser = subparsers.add_parser("human", help="Play manually")
    human_parser.add_argument("--log-file", type=str, help="Path to save game logs (JSONL)")
    human_parser.add_argument("--seed", type=int, help="Random seed for deterministic game")
    human_parser.add_argument("--fps", type=int, default=60, help="Render FPS")

    # --- Replay Mode ---
    replay_parser = subparsers.add_parser("replay", help="Watch replay from log")
    replay_parser.add_argument("--log-file", type=str, required=True, help="Path to JSONL log file")
    replay_parser.add_argument("--game-id", type=str, help="Specific game ID to replay")
    replay_parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    replay_parser.add_argument("--fps", type=int, default=60, help="Render FPS")

    # --- Agent Play Mode ---
    agent_parser = subparsers.add_parser("agent", help="Watch AI agent play")
    agent_parser.add_argument("--agent", type=str, choices=["random", "expectimax", "mcts"], default="random", help="Agent to run")
    agent_parser.add_argument("--log-file", type=str, help="Path to save game logs (JSONL)")
    agent_parser.add_argument("--seed", type=int, help="Random seed")
    agent_parser.add_argument("--delay", type=int, default=200, help="Delay between moves (ms)")
    agent_parser.add_argument("--fps", type=int, default=60, help="Render FPS")

    args = parser.parse_args()

    # Common UI Config
    config = UIConfig(fps=args.fps)

    # --- Mode Execution ---
    if args.mode == "human":
        print("Starting Human Play Mode...")
        logger = setup_logger(args.log_file, "human", args.seed)
        env = GameEnv(seed=args.seed)
        run_human_ui(env=env, seed=args.seed, logger=logger, config=config)

    elif args.mode == "replay":
        print(f"Starting Replay Mode from {args.log_file}...")
        run_replay_from_log(
            log_file=Path(args.log_file),
            game_id=args.game_id,
            speed=args.speed,
            config=config
        )

    elif args.mode == "agent":
        print(f"Starting Agent Play Mode ({args.agent})...")

        # Select Agent
        if args.agent == "random":
            agent = RandomAgent(seed=args.seed)
        elif args.agent == "expectimax":
            # TODO: Instantiate Expectimax when available
            # agent = ExpectimaxAgent(...)
            print("Expectimax agent not yet fully integrated in launcher. Using RandomAgent.")
            agent = RandomAgent(seed=args.seed)
        elif args.agent == "mcts":
            # TODO: Instantiate MCTS when available
            # agent = MCTSAgent(...)
            print("MCTS agent not yet fully integrated in launcher. Using RandomAgent.")
            agent = RandomAgent(seed=args.seed)
        else:
            print(f"Unknown agent: {args.agent}")
            sys.exit(1)

        logger = setup_logger(args.log_file, args.agent, args.seed)
        env = GameEnv(seed=args.seed)

        run_agent(
            agent=agent,
            env=env,
            seed=args.seed,
            logger=logger,
            move_delay_ms=args.delay,
            config=config
        )

if __name__ == "__main__":
    main()

