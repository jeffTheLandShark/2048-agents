"""Script to process raw JSONL logs into Parquet tables using ETL."""

from pathlib import Path
from stats_logging.etl import process_logs_to_parquet

def main():
    """Process mass_expectimax_corner.jsonl and mass_mcts.jsonl into Parquet tables."""
    raw_logs_dir = Path("data/raw_logs")
    processed_dir = Path("data/processed")

    # Files to process
    log_files = [
        # "mass_expectimax_corner.jsonl",
        # "mass_mcts.jsonl"
        # "random_games.jsonl"
        "mass_mcts_heuristic.jsonl"
    ]

    for log_file_name in log_files:
        log_file = raw_logs_dir / log_file_name

        if not log_file.exists():
            print(f"Warning: {log_file} does not exist. Skipping...")
            continue

        # Create output directory named after the log file (without extension)
        output_dir = processed_dir / log_file.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Processing: {log_file}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}")

        process_logs_to_parquet(log_file, output_dir)

        print(f"Completed processing {log_file_name}\n")

if __name__ == "__main__":
    main()

