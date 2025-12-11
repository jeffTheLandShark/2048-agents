"""ETL utilities for converting raw JSONL logs to Parquet tables."""

from typing import List, Optional, Generator, Any, TYPE_CHECKING, TypedDict, Dict
from pathlib import Path
import json
import pandas as pd

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
                print(
                    f"Warning: Skipping invalid JSON at line {line_num + 1} in {log_file}: {e}"
                )


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
    rows = []
    for log in logs:
        summary = log.get("summary", {})
        final_tile_counts = summary.get("final_tile_counts", {})
        final_heuristics = summary.get("final_heuristics", {})
        highest_tile = summary.get("highest_tile", 0)

        row = {
            "game_id": log.get("game_id"),
            "agent": log.get("agent"),
            "board_size": log.get("board_size"),
            "seed": log.get("seed"),
            "final_score": summary.get("final_score", 0),
            "highest_tile": highest_tile,
            "game_length": summary.get("game_length", 0),
            "reached_2048": highest_tile >= 2048,
            "reached_4096": highest_tile >= 4096,
        }

        # Add final heuristics with h_ prefix
        for key, value in final_heuristics.items():
            row[f"h_{key}"] = value

        # Add final tile counts (optional - can be kept as dict or flattened)
        # For now, we'll add common tile values as columns
        for tile_value in ["2", "4", "8", "16", "32", "64", "128", "256", "512", "1024", "2048", "4096"]:
            row[f"tile_{tile_value}_count"] = final_tile_counts.get(tile_value, 0)

        rows.append(row)

    return pd.DataFrame(rows)


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
    rows = []
    for log in logs:
        game_id = log.get("game_id")
        steps = log.get("steps", [])

        for step in steps:
            heuristics = step.get("heuristics", {})

            row = {
                "game_id": game_id,
                "t": step.get("t"),
                "action": step.get("action"),
                "reward": step.get("reward", 0.0),
                "score": step.get("score", 0),
                "done": step.get("done", False),
            }

            # Add heuristics with h_ prefix
            for key, value in heuristics.items():
                row[f"h_{key}"] = value

            rows.append(row)

    return pd.DataFrame(rows)


def create_tile_counts_table(logs: List[GameLog]) -> pd.DataFrame:
    """
    Create tile_counts.parquet table from raw logs.

    Normalized long-format table: one row per (game, step, tile_value).

    Args:
        logs: List of GameLog dictionaries from JSONL.

    Returns:
        DataFrame with columns: game_id, t, tile_value, count.
    """
    rows = []
    for log in logs:
        game_id = log.get("game_id")
        steps = log.get("steps", [])

        for step in steps:
            t = step.get("t")
            tile_counts = step.get("tile_counts", {})

            for tile_value, count in tile_counts.items():
                rows.append({
                    "game_id": game_id,
                    "t": t,
                    "tile_value": tile_value,
                    "count": count,
                })

    return pd.DataFrame(rows)


def process_logs_to_parquet(
    log_file: Path,
    output_dir: Path,
    games_summary_name: str = "games_summary.parquet",
    steps_name: str = "steps.parquet",
    tile_counts_name: str = "tile_counts.parquet",
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
    log_file = Path(log_file)
    output_dir = Path(output_dir)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all logs from JSONL file
    logs = list(load_jsonl_logs(log_file))

    if not logs:
        print(f"Warning: No logs found in {log_file}")
        return

    # Create tables
    print(f"Processing {len(logs)} games from {log_file}...")

    games_summary_df = create_games_summary_table(logs)
    steps_df = create_steps_table(logs)
    tile_counts_df = create_tile_counts_table(logs)

    # Write to Parquet files
    games_summary_path = output_dir / games_summary_name
    steps_path = output_dir / steps_name
    tile_counts_path = output_dir / tile_counts_name

    games_summary_df.to_parquet(games_summary_path, index=False)
    print(f"Wrote {len(games_summary_df)} rows to {games_summary_path}")

    steps_df.to_parquet(steps_path, index=False)
    print(f"Wrote {len(steps_df)} rows to {steps_path}")

    tile_counts_df.to_parquet(tile_counts_path, index=False)
    print(f"Wrote {len(tile_counts_df)} rows to {tile_counts_path}")
