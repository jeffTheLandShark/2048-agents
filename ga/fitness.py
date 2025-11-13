"""Fitness strategy implementations for GA evaluation."""

from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd


class FitnessStrategy(ABC):
    """
    Abstract base class for fitness computation strategies.

    Each strategy computes fitness from game summary statistics (e.g., from games_summary.parquet).
    """

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> float:
        """
        Compute fitness score from game summary DataFrame.

        Args:
            df: DataFrame with game summaries (one row per game).
                 Expected columns: final_score, highest_tile, game_length, etc.

        Returns:
            Fitness score (higher is better).
        """
        raise NotImplementedError

    def get_name(self) -> str:
        """
        Get name/identifier for this fitness strategy.

        Returns:
            Strategy name string.
        """
        return self.__class__.__name__


class MeanScoreFitness(FitnessStrategy):
    """
    Fitness strategy that uses mean final score across games.

    Simple baseline: fitness = mean(final_score).
    """

    def compute(self, df: pd.DataFrame) -> float:
        """
        Compute fitness as mean final score.

        Args:
            df: DataFrame with game summaries (must have 'final_score' column).

        Returns:
            Mean final score across all games.
        """
        raise NotImplementedError


class ScorePlusBonusFitness(FitnessStrategy):
    """
    Fitness strategy that combines score with achievement bonuses.

    Formula: fitness = mean(final_score) + bonus_for_2048 + bonus_for_4096 + ...
    """

    def __init__(
        self,
        bonus_2048: float = 10000.0,
        bonus_4096: float = 50000.0,
        bonus_8192: float = 200000.0
    ) -> None:
        """
        Initialize fitness strategy with bonus values.

        Args:
            bonus_2048: Bonus points for reaching 2048 tile.
            bonus_4096: Bonus points for reaching 4096 tile.
            bonus_8192: Bonus points for reaching 8192 tile.
        """
        raise NotImplementedError

    def compute(self, df: pd.DataFrame) -> float:
        """
        Compute fitness as mean score plus achievement bonuses.

        Args:
            df: DataFrame with game summaries (must have 'final_score' and 'highest_tile' columns).

        Returns:
            Mean final score plus weighted achievement bonuses.
        """
        raise NotImplementedError

