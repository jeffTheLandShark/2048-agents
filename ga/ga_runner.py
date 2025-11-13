"""Genetic algorithm runner for optimizing heuristic weights."""

from typing import List, Callable, Optional, Dict, Any
from ga.genome import Genome


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
        crossover_rate: float = 0.7,
        elite_size: int = 5,
        games_per_genome: int = 10,
        fitness_strategy: Optional[Callable] = None,
        initial_weights: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None
    ) -> None:
        """
        Initialize GA runner.

        Args:
            population_size: Number of genomes in each generation.
            num_generations: Number of generations to evolve.
            mutation_rate: Probability of mutating each weight.
            crossover_rate: Probability of crossover vs mutation.
            elite_size: Number of top genomes to preserve each generation.
            games_per_genome: Number of games to play per genome for fitness evaluation.
            fitness_strategy: Fitness computation strategy (from ga.fitness).
            initial_weights: Optional initial weight dictionary for seeding population.
            seed: Optional random seed for reproducibility.
        """
        raise NotImplementedError

    def run(self) -> List[Genome]:
        """
        Execute GA evolution loop.

        Returns:
            List of genomes from final generation (sorted by fitness).
        """
        raise NotImplementedError

    def evaluate_population(self, population: List[Genome]) -> List[float]:
        """
        Evaluate fitness for entire population.

        Args:
            population: List of genomes to evaluate.

        Returns:
            List of fitness scores (one per genome, same order).
        """
        raise NotImplementedError

    def select_parents(self, population: List[Genome], fitness: List[float], num_parents: int) -> List[Genome]:
        """
        Select parent genomes for reproduction.

        Args:
            population: Current population of genomes.
            fitness: Fitness scores for each genome.
            num_parents: Number of parents to select.

        Returns:
            List of selected parent genomes.
        """
        raise NotImplementedError

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
        raise NotImplementedError

