import numpy as np
from tictactoe import Board

test_board = list()
result = list()

test_board.append(np.array([['X', 'X', 'X'], ['O', 'X', 'O'], ['X', 'O', 'X']]))
result.append([])


def test_possible_moves():
    for t, r in zip(test_board, result):
        b = Board(t)
        assert b.possible_moves() == r
