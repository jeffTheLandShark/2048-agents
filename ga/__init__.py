"""Genetic algorithm framework for optimizing heuristic weights."""

from ga.genome import Genome
from ga.fitness import FitnessStrategy, MeanScoreFitness, ScorePlusBonusFitness

__all__ = [
    "Genome",
    "GARunner",
    "run_games_with_weights",
    "FitnessStrategy",
    "MeanScoreFitness",
    "ScorePlusBonusFitness",
]

# Lazy import to avoid circular import warning when running ga.ga_runner as main
def __getattr__(name: str):
    if name == "GARunner" or name == "run_games_with_weights":
        from ga.ga_runner import GARunner, run_games_with_weights
        if name == "GARunner":
            return GARunner
        elif name == "run_games_with_weights":
            return run_games_with_weights
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

