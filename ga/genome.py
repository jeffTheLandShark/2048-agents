"""Genome representation for genetic algorithm optimization."""

from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy.random as random
import numpy as np
from heuristics.features import AVAILABLE_FEATURES


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
        if rng is None:
            rng = random.default_rng()

        new_weights = {}
        for key, value in self.weights.items():
            if rng.random() < mutation_rate:
                # Add Gaussian noise
                noise = rng.normal(0, mutation_strength)
                new_weights[key] = value + noise
            else:
                new_weights[key] = value

        return Genome(weights=new_weights)

    def crossover(
        self,
        other: "Genome",
        crossover_rate: float = 0.7,
        rng: Optional[random.Generator] = None
    ) -> "Genome":
        """
        Create offspring genome via crossover with another genome.

        Uses uniform crossover: for each weight, randomly choose from self or other.

        Args:
            other: Other parent genome.
            crossover_rate: Probability of taking weight from self vs other.
            rng: Optional random number generator.

        Returns:
            New Genome instance with weights from both parents.
        """
        if rng is None:
            rng = random.default_rng()

        new_weights = {}
        for key in self.weights:
            if key in other.weights:
                # Uniform crossover: choose from self or other based on crossover_rate
                if rng.random() < crossover_rate:
                    new_weights[key] = self.weights[key]
                else:
                    new_weights[key] = other.weights[key]
            else:
                # If key not in other, use self's value
                new_weights[key] = self.weights[key]

        return Genome(weights=new_weights)

    def normalize(self, method: str = "l2") -> "Genome":
        """
        Normalize genome weights using specified method.

        Args:
            method: Normalization method ("l2", "max", "min_max", etc.).

        Returns:
            New Genome instance with normalized weights.
        """
        weight_values = np.array(list(self.weights.values()))

        if method == "l2":
            norm = np.linalg.norm(weight_values)
            if norm > 0:
                normalized = weight_values / norm
            else:
                normalized = weight_values
        elif method == "max":
            max_val = np.max(np.abs(weight_values))
            if max_val > 0:
                normalized = weight_values / max_val
            else:
                normalized = weight_values
        elif method == "min_max":
            min_val = np.min(weight_values)
            max_val = np.max(weight_values)
            if max_val > min_val:
                normalized = (weight_values - min_val) / (max_val - min_val)
            else:
                normalized = weight_values
        else:
            # No normalization
            normalized = weight_values

        new_weights = {key: float(val) for key, val in zip(self.weights.keys(), normalized)}
        return Genome(weights=new_weights)

    def copy(self) -> "Genome":
        """
        Create a deep copy of this genome.

        Returns:
            New Genome instance with copied weights.
        """
        return Genome(weights=self.weights.copy())

    @classmethod
    def random(
        cls,
        feature_names: Optional[List[str]] = None,
        weight_range: tuple[float, float] = (-10.0, 10.0),
        rng: Optional[random.Generator] = None
    ) -> "Genome":
        """
        Create a random genome with weights in the specified range.

        Args:
            feature_names: List of feature names. If None, uses AVAILABLE_FEATURES.
            weight_range: Tuple of (min, max) for weight values.
            rng: Optional random number generator.

        Returns:
            New Genome instance with random weights.
        """
        if rng is None:
            rng = random.default_rng()

        if feature_names is None:
            feature_names = AVAILABLE_FEATURES

        weights = {
            name: float(rng.uniform(weight_range[0], weight_range[1]))
            for name in feature_names
        }
        return cls(weights=weights)

