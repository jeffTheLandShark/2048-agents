"""
Genetic Algorithm (GA) Runner for Heuristic Optimization.

This script executes the evolutionary process to optimize the weights of the Expectimax
heuristic function. It manages the population of genomes (weight sets), evaluates their
fitness by running parallel game simulations, and evolves the population over generations
using selection, crossover, and mutation.

Key features:
- Multiprocessing support for parallel fitness evaluation.
- Configurable GA parameters (population size, mutation rate, etc.).
- Tracking and saving of the best genomes.
"""

from typing import List, Callable, Optional, Dict, Any
from pathlib import Path
import numpy.random as random
import numpy as np
import pandas as pd
import json
import multiprocessing as mp

from tqdm import tqdm

from ga.genome import Genome
from ga.fitness import FitnessStrategy, MeanScoreFitness, ScorePlusBonusFitness
from agents.expectimax import ExpectimaxAgent
from game_2048.game_env import GameEnv
from heuristics.features import compute_all_features, compute_tile_counts, create_game_summary
from stats_logging import GameSummary


def _evaluate_single_genome(args: Dict[str, Any]) -> float:
    """
    Helper function for multiprocessing evaluation of a single genome.
    """
    genome_weights = args["weights"]
    games_per_genome = args["games_per_genome"]
    board_size = args["board_size"]
    depth_limit = args["depth_limit"]
    time_limit_ms = args["time_limit_ms"]
    base_seed = args["base_seed"]
    fitness_strategy = args["fitness_strategy"]
    sample_empty_cells = args["sample_empty_cells"]
    max_empty_samples = args["max_empty_samples"]
    sample_depth_cutoff = args["sample_depth_cutoff"]

    df = run_games_with_weights(
        weights=genome_weights,
        num_games=games_per_genome,
        board_size=board_size,
        depth_limit=depth_limit,
        time_limit_ms=time_limit_ms,
        base_seed=base_seed,
        sample_empty_cells=sample_empty_cells,
        max_empty_samples=max_empty_samples,
        sample_depth_cutoff=sample_depth_cutoff,
        sample_rng_seed=base_seed,
    )
    return float(fitness_strategy.compute(df))


def run_games_with_weights(
    weights: Dict[str, float],
    num_games: int,
    board_size: int = 4,
    depth_limit: int = 5,
    time_limit_ms: Optional[int] = None,
    seed: Optional[int] = None,
    base_seed: Optional[int] = None,
    sample_empty_cells: bool = True,
    max_empty_samples: int = 6,
    sample_depth_cutoff: int = 2,
    sample_rng_seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run K games with Expectimax agent using given heuristic weights.

    Args:
        weights: Dictionary mapping heuristic feature names to weights.
        num_games: Number of games to play.
        board_size: Size of the game board.
        depth_limit: Expectimax depth limit.
        time_limit_ms: Optional time limit per move.
        seed: Optional random seed (if None, uses base_seed + game_index).
        base_seed: Base seed for deterministic games (seed = base_seed + game_index).

    Returns:
        DataFrame with one row per game containing:
        - final_score: Final game score
        - highest_tile: Highest tile reached
        - game_length: Number of steps
        - final_tile_counts: Final tile distribution (as dict)
        - final_heuristics: Final heuristic values (as dict)
    """
    agent = ExpectimaxAgent(
        depth_limit=depth_limit,
        time_limit_ms=time_limit_ms,
        heuristic_weights=weights,
        use_iterative_deepening=True,
        sample_empty_cells=sample_empty_cells,
        max_empty_samples=max_empty_samples,
        sample_depth_cutoff=sample_depth_cutoff,
        sample_rng_seed=sample_rng_seed if sample_rng_seed is not None else 0,
    )

    results = []

    for game_idx in range(num_games):
        # Use deterministic seed if provided
        game_seed = seed if seed is not None else (base_seed + game_idx if base_seed is not None else None)

        env = GameEnv(board_size=board_size, seed=game_seed)
        board, _ = env.reset()

        score = 0
        step_count = 0
        done = False

        while not done:
            legal_moves = env.legal_moves(board)
            if not legal_moves:
                done = True
                break

            action = agent.choose_action(board, legal_moves)
            board, reward, done, _ = env.step(action)
            score += int(reward)
            step_count += 1

        # Compute final statistics
        tile_counts = compute_tile_counts(board)
        heuristics = compute_all_features(board)

        summary = create_game_summary(
            board,
            score,
            step_count,
            heuristics,
            tile_counts
        )

        results.append({
            'final_score': summary['final_score'],
            'highest_tile': summary['highest_tile'],
            'game_length': summary['game_length'],
            'final_tile_counts': summary['final_tile_counts'],
            'final_heuristics': summary['final_heuristics']
        })

    return pd.DataFrame(results)


class GARunner:
    """
    Genetic algorithm runner for optimizing Expectimax heuristic weights.

    Runs configurable number of generations, evaluating each genome by
    playing K games with Expectimax(genome_weights) and computing fitness.
    """

    def __init__(
        self,
        population_size: int = 50,
        num_generations: int = 100,
        mutation_rate: float = 0.1,
        mutation_strength: float = 0.1,
        crossover_rate: float = 0.7,
        elite_size: int = 5,
        games_per_genome: int = 10,
        fitness_strategy: Optional[FitnessStrategy] = None,
        initial_weights: Optional[Dict[str, float]] = None,
        board_size: int = 4,
        depth_limit: int = 5,
        time_limit_ms: Optional[int] = None,
        seed: Optional[int] = None,
        output_dir: Optional[Path] = None,
        num_workers: Optional[int] = None,
        sample_empty_cells: bool = True,
        max_empty_samples: int = 6,
        sample_depth_cutoff: int = 2,
    ) -> None:
        """
        Initialize GA runner.

        Args:
            population_size: Number of genomes in each generation.
            num_generations: Number of generations to evolve.
            mutation_rate: Probability of mutating each weight.
            mutation_strength: Standard deviation of mutation noise.
            crossover_rate: Probability of taking weight from first parent in crossover.
            elite_size: Number of top genomes to preserve each generation.
            games_per_genome: Number of games to play per genome for fitness evaluation.
            fitness_strategy: Fitness computation strategy (from ga.fitness).
            initial_weights: Optional initial weight dictionary for seeding population.
            board_size: Size of the game board.
            depth_limit: Expectimax depth limit.
            time_limit_ms: Optional time limit per move.
            seed: Optional random seed for reproducibility.
            output_dir: Optional directory to save best genomes.
        """
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.games_per_genome = games_per_genome
        self.fitness_strategy = fitness_strategy or MeanScoreFitness()
        self.board_size = board_size
        self.depth_limit = depth_limit
        self.time_limit_ms = time_limit_ms
        self.output_dir = Path(output_dir) if output_dir else None
        self.num_workers = num_workers
        self.sample_empty_cells = sample_empty_cells
        self.max_empty_samples = max_empty_samples
        self.sample_depth_cutoff = sample_depth_cutoff
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize RNG
        self.rng = random.default_rng(seed)

        # Initialize population
        if initial_weights:
            # Create one genome from initial weights, then mutate to create population
            base_genome = Genome(weights=initial_weights)
            self.population = [base_genome.copy()]
            # Fill rest with random genomes
            while len(self.population) < population_size:
                self.population.append(Genome.random(rng=self.rng))
        else:
            self.population = [Genome.random(rng=self.rng) for _ in range(population_size)]

        # Track best genome across all generations
        self.best_genome: Optional[Genome] = None
        self.best_fitness: float = float('-inf')

    def run(self) -> List[Genome]:
        """
        Execute GA evolution loop.

        Returns:
            List of genomes from final generation (sorted by fitness, best first).
        """
        # Create progress bar for generations
        gen_pbar = tqdm(
            range(self.num_generations),
            desc="Evolution",
            unit="gen"
        )

        for generation in gen_pbar:
            gen_pbar.set_description(f"Generation {generation + 1}/{self.num_generations}")

            # Evaluate population
            fitness_scores = self.evaluate_population(self.population)

            # Track best
            best_idx = np.argmax(fitness_scores)
            best_fitness_this_gen = fitness_scores[best_idx]

            if best_fitness_this_gen > self.best_fitness:
                self.best_fitness = best_fitness_this_gen
                self.best_genome = self.population[best_idx].copy()

                # Update progress bar with best fitness
                gen_pbar.set_postfix({
                    'best_fitness': f"{self.best_fitness:.2f}",
                    'gen_best': f"{best_fitness_this_gen:.2f}",
                    'gen_mean': f"{np.mean(fitness_scores):.2f}"
                })

                # Save best genome
                if self.output_dir:
                    self._save_best_genome(generation)
            else:
                # Update progress bar without new best
                gen_pbar.set_postfix({
                    'best_fitness': f"{self.best_fitness:.2f}",
                    'gen_best': f"{best_fitness_this_gen:.2f}",
                    'gen_mean': f"{np.mean(fitness_scores):.2f}"
                })

            # Select elite
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
            elite = [self.population[i].copy() for i in elite_indices]

            # Create next generation
            if generation < self.num_generations - 1:
                self.population = self.create_next_generation(
                    self.population,
                    fitness_scores,
                    elite
                )

        gen_pbar.close()

        # Sort final population by fitness
        final_fitness = self.evaluate_population(self.population)
        sorted_indices = np.argsort(final_fitness)[::-1]
        return [self.population[i] for i in sorted_indices]

    def evaluate_population(self, population: List[Genome]) -> List[float]:
        """
        Evaluate fitness for entire population (optionally in parallel).

        Args:
            population: List of genomes to evaluate.

        Returns:
            List of fitness scores (one per genome, same order).
        """
        fitness_scores: List[float] = [0.0] * len(population)

        # Seeds per genome for determinism across processes
        seeds = self.rng.integers(0, 2**31, size=len(population)) if self.rng else [None] * len(population)

        if self.num_workers and self.num_workers > 1:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=self.num_workers) as pool:
                args_iter = []
                for idx, genome in enumerate(population):
                    args_iter.append(
                        {
                            "weights": genome.weights,
                            "games_per_genome": self.games_per_genome,
                            "board_size": self.board_size,
                            "depth_limit": self.depth_limit,
                            "time_limit_ms": self.time_limit_ms,
                            "base_seed": int(seeds[idx]) if seeds[idx] is not None else None,
                            "fitness_strategy": self.fitness_strategy,
                            "sample_empty_cells": self.sample_empty_cells,
                            "max_empty_samples": self.max_empty_samples,
                            "sample_depth_cutoff": self.sample_depth_cutoff,
                        }
                    )

                for idx, fitness in tqdm(
                    enumerate(pool.imap(_evaluate_single_genome, args_iter)),
                    total=len(population),
                    desc="  Evaluating genomes",
                    unit="genome",
                    leave=False,
                ):
                    fitness_scores[idx] = fitness
        else:
            for idx, genome in tqdm(
                enumerate(population),
                total=len(population),
                desc="  Evaluating genomes",
                unit="genome",
                leave=False,
            ):
                df = run_games_with_weights(
                    weights=genome.weights,
                    num_games=self.games_per_genome,
                    board_size=self.board_size,
                    depth_limit=self.depth_limit,
                    time_limit_ms=self.time_limit_ms,
                    base_seed=int(seeds[idx]) if seeds[idx] is not None else None,
                    sample_empty_cells=self.sample_empty_cells,
                    max_empty_samples=self.max_empty_samples,
                    sample_depth_cutoff=self.sample_depth_cutoff,
                    sample_rng_seed=int(seeds[idx]) if seeds[idx] is not None else None,
                )
                fitness_scores[idx] = float(self.fitness_strategy.compute(df))

        return fitness_scores

    def select_parents(self, population: List[Genome], fitness: List[float], num_parents: int) -> List[Genome]:
        """
        Select parent genomes for reproduction using tournament selection.

        Args:
            population: Current population of genomes.
            fitness: Fitness scores for each genome.
            num_parents: Number of parents to select.

        Returns:
            List of selected parent genomes.
        """
        parents = []
        fitness_array = np.array(fitness)

        # Tournament selection with tournament size 3
        tournament_size = 3

        for _ in range(num_parents):
            # Randomly select tournament participants
            tournament_indices = self.rng.choice(len(population), size=tournament_size, replace=False)
            tournament_fitness = fitness_array[tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            parents.append(population[winner_idx])

        return parents

    def create_next_generation(
        self,
        population: List[Genome],
        fitness: List[float],
        elite: List[Genome]
    ) -> List[Genome]:
        """
        Create next generation via selection, crossover, and mutation.

        Args:
            population: Current population.
            fitness: Fitness scores for current population.
            elite: Top genomes to preserve (elitism).

        Returns:
            New population for next generation.
        """
        new_population = []

        # Add elite (preserve best genomes)
        new_population.extend(elite)

        # Generate remaining genomes through crossover and mutation
        num_to_generate = self.population_size - len(elite)

        # Select parents
        parents = self.select_parents(population, fitness, num_to_generate * 2)

        # Create offspring via crossover and mutation
        for i in range(0, len(parents) - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]

            # Crossover
            offspring = parent1.crossover(parent2, crossover_rate=self.crossover_rate, rng=self.rng)

            # Mutation
            offspring = offspring.mutate(
                mutation_rate=self.mutation_rate,
                mutation_strength=self.mutation_strength,
                rng=self.rng
            )

            new_population.append(offspring)

            # If we need one more, create another from same parents
            if len(new_population) < self.population_size:
                offspring2 = parent2.crossover(parent1, crossover_rate=self.crossover_rate, rng=self.rng)
                offspring2 = offspring2.mutate(
                    mutation_rate=self.mutation_rate,
                    mutation_strength=self.mutation_strength,
                    rng=self.rng
                )
                new_population.append(offspring2)

        # Trim to exact population size if needed
        return new_population[:self.population_size]

    def _save_best_genome(self, generation: int) -> None:
        """Save best genome to JSON file."""
        if self.best_genome is None or self.output_dir is None:
            return

        filename = self.output_dir / f"best_genome_gen_{generation + 1}.json"
        with open(filename, 'w') as f:
            json.dump({
                'generation': generation + 1,
                'fitness': self.best_fitness,
                'weights': self.best_genome.weights
            }, f, indent=2)

        # Also save as latest
        latest_file = self.output_dir / "best_genome_latest.json"
        with open(latest_file, 'w') as f:
            json.dump({
                'generation': generation + 1,
                'fitness': self.best_fitness,
                'weights': self.best_genome.weights
            }, f, indent=2)


def main():
    """Command-line interface for running genetic algorithm optimization."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run genetic algorithm to optimize Expectimax heuristic weights."
    )

    parser.add_argument(
        "--population-size",
        type=int,
        default=50,
        help="Number of genomes in each generation (default: 50)"
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=100,
        help="Number of generations to evolve (default: 100)"
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.1,
        help="Probability of mutating each weight (default: 0.1)"
    )
    parser.add_argument(
        "--mutation-strength",
        type=float,
        default=0.1,
        help="Standard deviation of mutation noise (default: 0.1)"
    )
    parser.add_argument(
        "--crossover-rate",
        type=float,
        default=0.7,
        help="Probability of taking weight from first parent in crossover (default: 0.7)"
    )
    parser.add_argument(
        "--elite-size",
        type=int,
        default=5,
        help="Number of top genomes to preserve each generation (default: 5)"
    )
    parser.add_argument(
        "--games-per-genome",
        type=int,
        default=10,
        help="Number of games to play per genome for fitness evaluation (default: 10)"
    )
    parser.add_argument(
        "--fitness-strategy",
        type=str,
        choices=["mean_score", "score_plus_bonus"],
        default="mean_score",
        help="Fitness computation strategy (default: mean_score)"
    )
    parser.add_argument(
        "--board-size",
        type=int,
        default=4,
        help="Size of the game board (default: 4)"
    )
    parser.add_argument(
        "--depth-limit",
        type=int,
        default=5,
        help="Expectimax depth limit (default: 5)"
    )
    parser.add_argument(
        "--time-limit-ms",
        type=int,
        default=None,
        help="Optional time limit per move in milliseconds"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/ga_results",
        help="Directory to save best genomes (default: data/ga_results)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker processes for genome evaluation (default: None -> sequential)"
    )
    parser.add_argument(
        "--sample-empty-cells",
        action="store_true",
        default=True,
        help="Sample empty cells at chance nodes to reduce branching (default: on for GA runs)"
    )
    parser.add_argument(
        "--max-empty-samples",
        type=int,
        default=6,
        help="Max empty cells to sample at deep chance nodes when sampling is enabled (default: 6)"
    )
    parser.add_argument(
        "--sample-depth-cutoff",
        type=int,
        default=2,
        help="Only sample empty cells when remaining depth is greater than this cutoff (default: 2)"
    )
    parser.add_argument(
        "--initial-weights",
        type=str,
        default=None,
        help="Path to JSON file with initial weights to seed population"
    )

    args = parser.parse_args()

    # Load initial weights if provided
    initial_weights = None
    if args.initial_weights:
        with open(args.initial_weights, 'r') as f:
            initial_weights = json.load(f)
            if 'weights' in initial_weights:
                initial_weights = initial_weights['weights']

    # Select fitness strategy
    if args.fitness_strategy == "mean_score":
        fitness_strategy = MeanScoreFitness()
    elif args.fitness_strategy == "score_plus_bonus":
        fitness_strategy = ScorePlusBonusFitness()
    else:
        fitness_strategy = MeanScoreFitness()

    # Create GA runner
    runner = GARunner(
        population_size=args.population_size,
        num_generations=args.generations,
        mutation_rate=args.mutation_rate,
        mutation_strength=args.mutation_strength,
        crossover_rate=args.crossover_rate,
        elite_size=args.elite_size,
        games_per_genome=args.games_per_genome,
        fitness_strategy=fitness_strategy,
        initial_weights=initial_weights,
        board_size=args.board_size,
        depth_limit=args.depth_limit,
        time_limit_ms=args.time_limit_ms,
        seed=args.seed,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        sample_empty_cells=args.sample_empty_cells,
        max_empty_samples=args.max_empty_samples,
        sample_depth_cutoff=args.sample_depth_cutoff,
    )

    print("=" * 60)
    print("Genetic Algorithm Optimization")
    print("=" * 60)
    print(f"Population size: {args.population_size}")
    print(f"Generations: {args.generations}")
    print(f"Games per genome: {args.games_per_genome}")
    print(f"Fitness strategy: {args.fitness_strategy}")
    print(f"Board size: {args.board_size}")
    print(f"Depth limit: {args.depth_limit}")
    print(f"Time limit ms: {args.time_limit_ms}")
    print(f"Num workers: {args.num_workers}")
    print(f"Sample empty cells: {args.sample_empty_cells} (max {args.max_empty_samples}, cutoff {args.sample_depth_cutoff})")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    print()

    # Run evolution
    final_population = runner.run()

    print()
    print("=" * 60)
    print("Evolution Complete!")
    print("=" * 60)
    print(f"Best fitness: {runner.best_fitness:.2f}")
    print(f"Best weights:")
    if runner.best_genome:
        for name, weight in runner.best_genome.weights.items():
            print(f"  {name}: {weight:.4f}")
    print(f"\nBest genome saved to: {args.output_dir}/best_genome_latest.json")
    print("=" * 60)


if __name__ == "__main__":
    main()

