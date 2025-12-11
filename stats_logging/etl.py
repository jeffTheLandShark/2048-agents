"""ETL utilities for converting raw JSONL logs to Parquet tables."""

from typing import List, Optional, Generator, Any
from pathlib import Path
import json
import pandas as pd
from typing import TypedDict, Dict, Any, Optional, List

# Define GameLog locally to avoid circular import
GameLog = TypedDict('GameLog', {
    'game_id': str,
    'agent': str,
    'board_size': int,
    'seed': Optional[int],
    'config': Dict[str, Any],
    'steps': List[Dict[str, Any]],
    'summary': Dict[str, Any]
})


def load_jsonl_logs(log_file: Path) -> Generator[GameLog, None, None]:
    """
    Yield game records one by one from a JSONL log file.
    Memory efficient for large files.

    Args:
        log_file: Path to JSONL file (one JSON object per line).

    Yields:
        GameLog dictionaries one at a time.
    """
    log_file = Path(log_file)
    if not log_file.exists():
        return

    with open(log_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON at line {line_num + 1} in {log_file}: {e}")


def get_game_by_id(log_file: Path, game_id: str) -> Optional[GameLog]:
    """
    Find a specific game log by its game_id without loading the whole file.
    Streams through the file until the matching ID is found.

    Args:
        log_file: Path to the JSONL log file.
        game_id: The unique identifier of the game to find.

    Returns:
        The GameLog if found, else None.
    """
    for game_log in load_jsonl_logs(log_file):
        if game_log.get("game_id") == game_id:
            return game_log
    return None


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
