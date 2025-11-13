"""Genome representation for genetic algorithm optimization."""

from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy.random as random


@dataclass
class Genome:
    """
    Genome representation for heuristic weight optimization.

    Contains a vector of floats (one weight per heuristic feature).
    Supports mutation, crossover, and normalization operations.
    """

    weights: Dict[str, float]
    """Dictionary mapping heuristic feature names to weight values."""

    def mutate(
        self,
        mutation_rate: float = 0.1,
        mutation_strength: float = 0.1,
        rng: Optional[random.Generator] = None
    ) -> "Genome":
        """
        Create a mutated copy of this genome.

        Args:
            mutation_rate: Probability of mutating each weight.
            mutation_strength: Standard deviation of mutation noise.
            rng: Optional random number generator.

        Returns:
            New Genome instance with mutated weights.
        """
        raise NotImplementedError

    def crossover(
        self,
        other: "Genome",
        crossover_rate: float = 0.7,
        rng: Optional[random.Generator] = None
    ) -> "Genome":
        """
        Create offspring genome via crossover with another genome.

        Args:
            other: Other parent genome.
            crossover_rate: Probability of taking weight from self vs other.
            rng: Optional random number generator.

        Returns:
            New Genome instance with weights from both parents.
        """
        raise NotImplementedError

    def normalize(self, method: str = "l2") -> "Genome":
        """
        Normalize genome weights using specified method.

        Args:
            method: Normalization method ("l2", "max", "min_max", etc.).

        Returns:
            New Genome instance with normalized weights.
        """
        raise NotImplementedError

    def copy(self) -> "Genome":
        """
        Create a deep copy of this genome.

        Returns:
            New Genome instance with copied weights.
        """
        raise NotImplementedError

