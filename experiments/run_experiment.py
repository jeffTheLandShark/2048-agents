"""Experiment runner for executing multiple games with an agent."""

from typing import Optional
from pathlib import Path
from game_2048.game_env import GameEnv
from agents import Agent
from agents.mcts import MCTSAgent
from stats_logging import StatsLogger, GameSummary
from heuristics.features import (
    compute_tile_counts,
    compute_all_features,
)


# Example of what run_single_game should look like
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
    env.reset()
    board = env.get_board()

    if logger:
        logger.start_game(game_id=game_id, seed=seed)
        # Log initial state
        tile_counts = compute_tile_counts(board)
        heuristics = compute_all_features(board)
        logger.log_step(0, board, None, 0, 0, tile_counts, heuristics, False)

    step = 1
    while not env.is_game_over():
        legal_moves = env.legal_moves()
        if not legal_moves:
            break

        action = agent.choose_action(board, legal_moves)
        board, reward, done, info = env.step(action)

        if logger:
            tile_counts = compute_tile_counts(board)
            heuristics = compute_all_features(board)
            logger.log_step(
                step,
                board,
                action,
                reward,
                info["score"],
                tile_counts,
                heuristics,
                done,
            )
        step += 1

    summary = {
        "final_score": env.get_score(),
        "highest_tile": int(board.array.max()),
        "game_length": env.get_move_count(),
        "final_tile_counts": compute_tile_counts(board),
        "final_heuristics": compute_all_features(board),
    }

    if logger:
        logger.end_game(summary)

    return summary


def main():

    agent = MCTSAgent()
    stats_path = "data/raw_logs/mcts_batch.jsonl"
    logger = StatsLogger(Path(stats_path), "mcts", 4)
    print("Running MCTS agent for 10 games...")
    print(f"Logging to {stats_path}")

    for i in range(10):  # Run 10 games
        print(f"Starting game {i}...")
        env = GameEnv(seed=42 + i)

        summary = run_single_game(agent, env, logger=logger, seed=42 + i)
        print(f"Game {i} summary: {summary}")
    logger.close()


if __name__ == "__main__":
    main()
