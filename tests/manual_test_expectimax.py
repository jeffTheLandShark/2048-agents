
import numpy as np
from game import Board
from agents.expectimax import ExpectimaxAgent

def test_expectimax_run():
    # Create a board
    # 2 0 0 0
    # 0 0 0 0
    # 0 0 0 0
    # 0 0 0 0
    arr = np.zeros((4, 4), dtype=np.int32)
    arr[0, 0] = 2
    board = Board(arr)

    agent = ExpectimaxAgent(depth_limit=2, time_limit_ms=1000)

    print("Board:")
    print(arr)

    action = agent.choose_action(board, ["UP", "DOWN", "LEFT", "RIGHT"])
    print(f"Chosen action: {action}")

    # Test with a more complex board where specific move is obvious
    # 2 2 0 0  <- LEFT/RIGHT merges
    # 0 0 0 0
    # 0 0 0 0
    # 0 0 0 0
    arr[0, 1] = 2
    board = Board(arr)
    print("\nComplex Board:")
    print(arr)

    # LEFT should merge 2+2 -> 4. RIGHT should merge 2+2 -> 4.
    # Heuristics like empty tiles (merging frees a tile) and max tile (4 > 2) should favor merging.
    action = agent.choose_action(board, ["UP", "DOWN", "LEFT", "RIGHT"])
    print(f"Chosen action: {action}")

    # Check if it runs without error
    assert action in ["UP", "DOWN", "LEFT", "RIGHT"]

if __name__ == "__main__":
    test_expectimax_run()

