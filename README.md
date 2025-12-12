# 2048 AI Project

## Project Overview

A flexible 2048 AI research framework supporting multiple agents (Expectimax, MCTS, Random), heuristic optimization via genetic algorithms, and comprehensive logging/analysis. This project allows for the comparison of different AI strategies and the optimization of heuristic weights to achieve high scores in the 2048 game.

## Contributors

- Project Contributors (Please fill in names)

## File Descriptions

### Core Game Logic (`game_2048/`)
- `game_2048/__init__.py`: Package initialization for game logic.
- `game_2048/game_env.py`: Core game environment implementing deterministic rules, scoring, and transitions.
- `game_2048/board.py`: Board state representation using NumPy for efficient array operations.
- `game_2048/utils.py`: Utility functions for board manipulation (sliding, merging) and random tile spawning.
- `game_2048/pygame_ui.py`: Pygame-based graphical user interface for visualizing the game.

### AI Agents (`agents/`)
- `agents/__init__.py`: Package initialization for agents.
- `agents/base.py`: Abstract base class defining the interface for all agents.
- `agents/expectimax.py`: Expectimax search agent implementation using iterative deepening and heuristics.
- `agents/mcts.py`: Monte Carlo Tree Search (MCTS) agent implementation.
- `agents/random_agent.py`: Baseline agent that selects random valid moves.

### Heuristics (`heuristics/`)
- `heuristics/__init__.py`: Package initialization for heuristics.
- `heuristics/evaluator.py`: Logic for combining individual heuristic features into a weighted score.
- `heuristics/features.py`: Implementation of stateless heuristic feature functions (monotonicity, smoothness, etc.).

### Genetic Algorithm (`ga/`)
- `ga/__init__.py`: Package initialization for genetic algorithms.
- `ga/ga_runner.py`: Runner script for optimizing heuristic weights using genetic algorithms.
- `ga/fitness.py`: Defines fitness functions for evaluating genomes in the GA.
- `ga/genome.py`: Represents a candidate solution (set of weights) in the GA.

### Experiments and Analysis (`experiments/`, `stats_logging/`, root)
- `experiments/__init__.py`: Package initialization for experiments.
- `experiments/run_experiment.py`: Main entry point for running headless experiments defined in `config.json`.
- `experiments/config.json`: Configuration file for defining experiment parameters (agents, games, logging).
- `experiments/run_experiment.sh`: Shell script wrapper for running experiments.
- `stats_logging/__init__.py`: Package initialization for logging.
- `stats_logging/stats_logger.py`: Handles logging of game events, steps, and statistics to files.
- `stats_logging/etl.py`: Extract-Transform-Load utilities for processing raw logs.
- `process_logs.py`: Script to process raw JSON logs into Parquet format for efficient analysis.
- `run_ui.py`: Entry point for launching the graphical user interface.

### Tests (`tests/`)
- `tests/*`: Unit and integration tests ensuring the correctness of game logic and agents.

## Installation and Usage

### Prerequisites
- Python 3.8+
- pip

### Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the UI
To watch an agent play or play manually:
```bash
python run_ui.py
```
This will launch the Pygame window. You can configure the agent in the `run_ui.py` file or via command line arguments if implemented.

### Running Experiments
To run headless experiments (e.g., 100 games of Expectimax):
1. Configure `experiments/config.json` with your desired settings (agent type, number of games, output path).
2. Run the experiment script:
   ```bash
   python experiments/run_experiment.py
   ```

### Running Genetic Algorithm Optimization
To start the genetic algorithm:
```bash
python ga/ga_runner.py
```
Check the script for additional command-line arguments.

### Analysis
Results are stored in the `data/` directory:
- `data/raw_logs/`: Contains JSONL files with detailed game logs.
- `data/processed/`: Contains processed Parquet files (after running `process_logs.py`).
- `data/ga_results/`: Contains the best genomes found during GA runs.

To process logs:
```bash
python process_logs.py
```
The results can be interpreted by analyzing the score distributions, max tiles, and game lengths in the processed data.
