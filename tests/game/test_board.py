import numpy as np

from game.board import (
    Board,
    encode_board_flat,
    decode_board_flat,
    encode_board_log2,
    decode_board_log2,
)


def test_board_roundtrip_flat_and_log2():
    data = [[0, 2, 4, 0], [8, 0, 16, 0], [0, 32, 0, 64], [2, 0, 0, 0]]
    b = Board.from_list(data)
    flat = encode_board_flat(b)
    assert len(flat) == 16
    decoded = decode_board_flat(flat, 4)
    assert decoded == b

    log2 = encode_board_log2(b)
    # check log2 conversion for a couple of tiles
    assert log2[0] == 0
    assert log2[1] == 1  # 2 -> log2 1
    decoded2 = decode_board_log2(log2, 4)
    assert decoded2 == b


def test_board_get_set_and_copy():
    b = Board(np.zeros((3, 3), dtype=np.int32))
    b[0, 0] = 2
    assert b[0, 0] == 2
    c = b.copy()
    assert c == b
    c[0, 0] = 0
    assert c != b
