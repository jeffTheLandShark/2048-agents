"""Board class and encoding/decoding utilities for 2048 game state."""

from typing import List, Union, Optional
import numpy as np


class Board:
    """
    Board class wrapping a numpy array for 2048 game state.

    Provides type safety, validation, and fast numpy operations.
    Use `.array` property for performance-critical code that needs direct numpy access.
    """

    def __init__(self, board: Union[List[List[int]], np.ndarray], size: Optional[int] = None) -> None:
        """
        Initialize Board from a 2D list or numpy array.

        Args:
            board: Board state as 2D list or numpy array.
            size: Optional board size. If None, inferred from board dimensions.

        Raises:
            ValueError: If board is not square or has invalid dimensions.
        """
        if isinstance(board, np.ndarray):
            self._array = board.astype(np.int32)
        else:
            self._array = np.array(board, dtype=np.int32)

        # Validate square shape
        if len(self._array.shape) != 2:
            raise ValueError(f"Board must be 2D, got shape {self._array.shape}")

        if self._array.shape[0] != self._array.shape[1]:
            raise ValueError(f"Board must be square, got shape {self._array.shape}")

        # Validate size if provided
        if size is not None and self._array.shape[0] != size:
            raise ValueError(f"Board size mismatch: expected {size}, got {self._array.shape[0]}")

    @property
    def array(self) -> np.ndarray:
        """
        Get direct access to the underlying numpy array.

        Use this for performance-critical operations (e.g., vectorized heuristics).

        Returns:
            The underlying numpy array (shape: (n, n), dtype: int32).
        """
        return self._array

    @property
    def size(self) -> int:
        """
        Get the board size (n for an nxn board).

        Returns:
            Board size (e.g., 4 for a 4x4 board).
        """
        return self._array.shape[0]

    def copy(self) -> 'Board':
        """
        Create a fast copy of the board using numpy array copy.

        Returns:
            A new Board instance with copied data.
        """
        return Board(self._array.copy())

    def __getitem__(self, key: Union[int, tuple]) -> int:
        """
        Get tile value at position (row, col).

        Args:
            key: Either an int (row) or tuple (row, col).

        Returns:
            Tile value at the specified position.
        """
        return int(self._array[key])

    def __setitem__(self, key: Union[int, tuple], value: int) -> None:
        """
        Set tile value at position (row, col).

        Args:
            key: Either an int (row) or tuple (row, col).
            value: New tile value.
        """
        self._array[key] = value

    def __eq__(self, other: object) -> bool:
        """
        Check if two boards are equal.

        Args:
            other: Another Board instance or object to compare.

        Returns:
            True if boards have the same values, False otherwise.
        """
        if not isinstance(other, Board):
            return False
        return np.array_equal(self._array, other._array)

    def to_list(self) -> List[List[int]]:
        """
        Convert board to 2D list for backward compatibility.

        Returns:
            Board state as 2D list of tile values.
        """
        return self._array.tolist()

    @classmethod
    def from_list(cls, board: List[List[int]], size: Optional[int] = None) -> 'Board':
        """
        Create Board instance from a 2D list.

        Args:
            board: Board state as 2D list.
            size: Optional board size for validation.

        Returns:
            New Board instance.
        """
        return cls(board, size=size)

    def __repr__(self) -> str:
        """String representation of the board."""
        return f"Board(size={self.size}, array=\n{self._array})"


def encode_board_log2(board: Union[Board, List[List[int]]]) -> List[int]:
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
        board: Board state as Board instance or 2D list of tile values.

    Returns:
        Flattened list of log2 values (row-major order).
    """
    raise NotImplementedError


def decode_board_log2(encoded: List[int], board_size: int) -> Board:
    """
    Decode log2-encoded board back to tile values.

    Args:
        encoded: Flattened list of log2 values.
        board_size: Size of the board (n×n).

    Returns:
        Board instance.
    """
    raise NotImplementedError


def encode_board_flat(board: Union[Board, List[List[int]]]) -> List[int]:
    """
    Encode board as flattened list of raw tile values.

    Args:
        board: Board state as Board instance or 2D list of tile values.

    Returns:
        Flattened list of tile values (row-major order).
    """
    raise NotImplementedError


def decode_board_flat(encoded: List[int], board_size: int) -> Board:
    """
    Decode flattened board back to 2D representation.

    Args:
        encoded: Flattened list of tile values.
        board_size: Size of the board (n×n).

    Returns:
        Board instance.
    """
    raise NotImplementedError

