"""Board encoding/decoding utilities for logging and storage."""

from typing import List


def encode_board_log2(board: List[List[int]]) -> List[int]:
    """
    Encode board state as flattened list of log2 values.

    Converts tile values to log2 representation:
    - 0 (empty) -> 0
    - 2 -> 1
    - 4 -> 2
    - 8 -> 3
    - 16 -> 4
    - etc.

    Args:
        board: Board state as 2D list of tile values.

    Returns:
        Flattened list of log2 values (row-major order).
    """
    raise NotImplementedError


def decode_board_log2(encoded: List[int], board_size: int) -> List[List[int]]:
    """
    Decode log2-encoded board back to tile values.

    Args:
        encoded: Flattened list of log2 values.
        board_size: Size of the board (n×n).

    Returns:
        Board state as 2D list of tile values.
    """
    raise NotImplementedError


def encode_board_flat(board: List[List[int]]) -> List[int]:
    """
    Encode board as flattened list of raw tile values.

    Args:
        board: Board state as 2D list of tile values.

    Returns:
        Flattened list of tile values (row-major order).
    """
    raise NotImplementedError


def decode_board_flat(encoded: List[int], board_size: int) -> List[List[int]]:
    """
    Decode flattened board back to 2D representation.

    Args:
        encoded: Flattened list of tile values.
        board_size: Size of the board (n×n).

    Returns:
        Board state as 2D list of tile values.
    """
    raise NotImplementedError

