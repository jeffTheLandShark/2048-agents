"""ETL utilities for converting raw JSONL logs to Parquet tables."""

from typing import List, Optional
from pathlib import Path
import pandas as pd
from stats_logging import GameLog


def load_jsonl_logs(log_file: Path) -> List[GameLog]:
    """
    Load all game records from a JSONL log file.

    Args:
        log_file: Path to JSONL file (one JSON object per line).

    Returns:
        List of GameLog dictionaries.
    """
    raise NotImplementedError


def create_games_summary_table(logs: List[GameLog]) -> pd.DataFrame:
    """
    Create games_summary.parquet table from raw logs.

    One row per game with summary statistics.

    Args:
        logs: List of GameLog dictionaries from JSONL.

    Returns:
        DataFrame with columns: game_id, agent, board_size, seed, final_score,
        highest_tile, game_length, reached_2048, reached_4096, etc.
    """
    raise NotImplementedError


def create_steps_table(logs: List[GameLog]) -> pd.DataFrame:
    """
    Create steps.parquet table from raw logs.

    One row per step of every game.

    Args:
        logs: List of GameLog dictionaries from JSONL.

    Returns:
        DataFrame with columns: game_id, t, action, reward, score, done,
        h_empty, h_monotonicity, h_smoothness, etc.
    """
    raise NotImplementedError


def create_tile_counts_table(logs: List[GameLog]) -> pd.DataFrame:
    """
    Create tile_counts.parquet table from raw logs.

    Normalized long-format table: one row per (game, step, tile_value).

    Args:
        logs: List of GameLog dictionaries from JSONL.

    Returns:
        DataFrame with columns: game_id, t, tile_value, count.
    """
    raise NotImplementedError


def process_logs_to_parquet(
    log_file: Path,
    output_dir: Path,
    games_summary_name: str = "games_summary.parquet",
    steps_name: str = "steps.parquet",
    tile_counts_name: str = "tile_counts.parquet"
) -> None:
    """
    Convert JSONL log file to Parquet tables.

    Args:
        log_file: Path to input JSONL file.
        output_dir: Directory to write Parquet files.
        games_summary_name: Filename for games summary table.
        steps_name: Filename for steps table.
        tile_counts_name: Filename for tile counts table.
    """
    raise NotImplementedError

