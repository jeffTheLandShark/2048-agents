"""Heuristic feature computation functions for 2048 board evaluation."""

from typing import List, TypedDict, Dict, Any
import numpy as np
from stats_logging import GameSummary, TileCounts
from game import Board


class HeuristicFeatures(TypedDict):
    """Heuristic feature values for a board state."""
    empty: float
    monotonicity: float
    smoothness: float
    merge_potential: float
    max_tile: float
    sum_tiles: float


def compute_tile_counts(board: Board) -> TileCounts:
    """
    Compute the count of each tile value on the board.

    Args:
        board: Board state as Board instance.

    Returns:
        Dictionary mapping tile values (as strings) to their counts.
    """
    unique, counts = np.unique(board.array, return_counts=True)
    # Filter out zeros if desired, or keep them. Usually we track tile counts > 0
    result = {}
    for val, count in zip(unique, counts):
        if val > 0:
            result[str(val)] = int(count)
    return result

def create_game_summary(
    board: Board,
    final_score: int,
    game_length: int,
    heuristics: Dict[str, float],
    tile_counts: Dict[str, int],
) -> GameSummary:
    """Create game summary for logging."""
    board_array = board.array
    highest_tile = int(board_array.max())

    return GameSummary(
        final_score=final_score,
        highest_tile=highest_tile,
        game_length=game_length,
        final_tile_counts=tile_counts,
        final_heuristics=heuristics,
    )


def compute_empty_tiles(board: Board) -> int:
    """
    Compute number of empty tiles on the board.

    Args:
        board: Board state as Board instance. Use board.array for vectorized operations.

    Returns:
        Count of empty (zero) tiles.
    """
    return int(np.sum(board.array == 0))


def _shift_non_zeros(arr: np.ndarray) -> np.ndarray:
    """
    Shift non-zero elements to the left (for 2D array rows) maintaining order.
    Returns a new array with non-zeros packed to the left and padded with zeros.
    """
    # Create a boolean mask where True is zero
    is_zero = arr == 0
    # argsort with stable sort puts False (0) before True (1), preserving relative order
    # This gives us indices that put non-zeros first
    shifted_indices = np.argsort(is_zero, axis=1, kind='stable')
    return np.take_along_axis(arr, shifted_indices, axis=1)


def compute_monotonicity(board: Board) -> float:
    """
    Compute monotonicity score (preference for increasing/decreasing sequences).

    Measures how well tiles are arranged in monotonic order (increasing or decreasing).
    Higher values indicate better organization.

    Args:
        board: Board state as Board instance. Use board.array for vectorized operations.

    Returns:
        Monotonicity score (typically normalized to [0, 1]).
    """
    # We need to check both rows and columns
    # To behave like 2048, we should ignore zeros (look at the "collapsed" sequence)

    # 1. Get rows and cols
    rows = board.array
    cols = board.array.T

    # 2. Shift non-zeros to the start to get the contiguous sequence
    # e.g. [2, 0, 4, 0] -> [2, 4, 0, 0]
    rows_shifted = _shift_non_zeros(rows)
    cols_shifted = _shift_non_zeros(cols)

    def calculate_axis_score(shifted_arr):
        # Get differences between adjacent elements
        # shape: (4, 3) for a 4x4 board
        diffs = shifted_arr[:, 1:] - shifted_arr[:, :-1]

        # Mask for valid pairs (where both left and right elements are non-zero)
        # We only care about the sequence of actual tiles
        # shifted_arr has zeros at the end, so we check if the *left* element is non-zero
        # checking right element != 0 ensures we don't compare last tile with padding zero
        valid_mask = (shifted_arr[:, :-1] != 0) & (shifted_arr[:, 1:] != 0)

        # Count increasing and decreasing steps
        # >= 0 includes equal values which are monotonic
        is_increasing = (diffs >= 0) & valid_mask
        is_decreasing = (diffs <= 0) & valid_mask

        count_inc = np.sum(is_increasing, axis=1)
        count_dec = np.sum(is_decreasing, axis=1)
        count_valid = np.sum(valid_mask, axis=1)

        # Avoid division by zero
        # If count_valid is 0 (0 or 1 tile), score is 1.0
        with np.errstate(divide='ignore', invalid='ignore'):
            row_scores = np.maximum(count_inc, count_dec) / count_valid
            row_scores = np.where(count_valid == 0, 1.0, row_scores)

        return row_scores

    row_scores = calculate_axis_score(rows_shifted)
    col_scores = calculate_axis_score(cols_shifted)

    # Combine all scores
    return float(np.mean(np.concatenate([row_scores, col_scores])))


def compute_smoothness(board: Board) -> float:
    """
    Compute smoothness score (preference for adjacent tiles with similar values).

    Higher values indicate smoother board (more adjacent tiles with same or similar values).

    Args:
        board: Board state as Board instance. Use board.array for vectorized operations.

    Returns:
        Smoothness score (normalized to [0, 1]).
    """
    arr = board.array

    # Prepare log values for comparison, handling zeros safely
    with np.errstate(divide='ignore'):
        log_arr = np.log2(arr)
    # We use a mask, so the value at 0 doesn't matter, but let's clean it
    log_arr[arr == 0] = 0

    def calculate_pair_scores(arr_2d, log_arr_2d):
        # Compare arr[:, i] with arr[:, i+1]
        left = arr_2d[:, :-1]
        right = arr_2d[:, 1:]

        # Only consider pairs where both are non-zero
        mask = (left != 0) & (right != 0)

        if np.sum(mask) == 0:
            return 0.0, 0

        # 1.0 point for exact match
        exact_matches = (left == right) & mask

        # 0.5 point for close match (log difference <= 1)
        # avoiding exact matches to not double count
        log_diff = np.abs(log_arr_2d[:, :-1] - log_arr_2d[:, 1:])
        close_matches = (log_diff <= 1.0) & mask & (~exact_matches)

        score = np.sum(exact_matches) * 1.0 + np.sum(close_matches) * 0.5
        count = np.sum(mask)
        return score, count

    # Horizontal
    h_score, h_count = calculate_pair_scores(arr, log_arr)
    # Vertical
    v_score, v_count = calculate_pair_scores(arr.T, log_arr.T)

    total_score = h_score + v_score
    total_pairs = h_count + v_count

    return float(total_score / total_pairs) if total_pairs > 0 else 1.0


def compute_merge_potential(board: Board) -> float:
    """
    Compute merge potential (likelihood of creating merges).

    Measures how many adjacent tiles have the same value and could merge.

    Args:
        board: Board state as Board instance. Use board.array for vectorized operations.

    Returns:
        Merge potential score (typically normalized).
    """
    arr = board.array

    def count_merges_and_pairs(arr_2d):
        left = arr_2d[:, :-1]
        right = arr_2d[:, 1:]

        # Mask for non-zero pairs
        mask = (left != 0) & (right != 0)

        # Count exact matches within the mask
        merges = np.sum((left == right) & mask)
        pairs = np.sum(mask)

        return merges, pairs

    h_merges, h_pairs = count_merges_and_pairs(arr)
    v_merges, v_pairs = count_merges_and_pairs(arr.T)

    total_merges = h_merges + v_merges
    total_pairs = h_pairs + v_pairs

    return float(total_merges / total_pairs) if total_pairs > 0 else 0.0


def compute_max_tile(board: Board) -> int:
    """
    Get the maximum tile value on the board.

    Args:
        board: Board state as Board instance. Use board.array for vectorized operations.

    Returns:
        Maximum tile value (e.g., 2, 4, 8, ..., 2048).
    """
    return int(np.max(board.array))


def compute_sum_tiles(board: Board) -> int:
    """
    Compute sum of all tile values on the board.

    Args:
        board: Board state as Board instance. Use board.array for vectorized operations.

    Returns:
        Sum of all tile values.
    """
    return int(np.sum(board.array))


def compute_all_features(board: Board) -> HeuristicFeatures:
    """
    Compute all heuristic features for a board state.

    Args:
        board: Board state as Board instance. Use board.array for vectorized operations.

    Returns:
        HeuristicFeatures dictionary with all feature values.
    """
    return HeuristicFeatures(
        empty=float(compute_empty_tiles(board)),
        monotonicity=compute_monotonicity(board),
        smoothness=compute_smoothness(board),
        merge_potential=compute_merge_potential(board),
        max_tile=float(compute_max_tile(board)),
        sum_tiles=float(compute_sum_tiles(board))
    )


# Registry of available feature names for introspection
AVAILABLE_FEATURES: List[str] = list(HeuristicFeatures.__annotations__.keys())


if __name__ == "__main__":
    # Simple test hook
    import sys

    # Mock board for testing
    class MockBoard(Board):
        pass

    try:
        # Create a sample board
        # 8 0 4 0
        # 2 4 0 0
        # 2 8 0 0
        # 0 0 0 0
        arr = np.array([
            [8, 0, 4, 0],
            [2, 4, 0, 0],
            [2, 8, 0, 0],
            [0, 0, 0, 0]
        ])
        board = MockBoard(arr)

        print(f"Testing features with board:\n{arr}")
        print("-" * 30)

        features = compute_all_features(board)
        for name, value in features.items():
            print(f"{name:<20}: {value:.4f}")

        print("-" * 30)
        print("Tile counts:")
        print(compute_tile_counts(board))

        print("-" * 30)
        print("All features computed successfully.")

    except Exception as e:
        print(f"Error testing features: {e}")
        sys.exit(1)
