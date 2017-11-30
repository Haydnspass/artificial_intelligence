import numpy as np
from tictactoe import Board
from tictactoe import minimax

test_board = list()
result = list()

test_board.append(np.array([['X', 'X', 'X'], ['O', 'X', 'O'], ['X', 'O', 'X']]))
result.append([])

test_board.append(np.array([['X', 'O', 'X'], ['O', '', 'O'], ['X', 'O', 'X']]))
result.append([[1, 1]])

test_alg = list()
result_alg = list()

"""
NOTE: The following board configurations must be valid in a sense that are reachable, 
since current and next players are determined by the board configuration.
"""

test_alg.append(np.array([
    ['X', '', 'X'],
    ['X', 'O', 'O'],
    ['O', 'O', 'X']]))

result_alg.append([0, 1])


test_alg.append(np.array([
    ['O', '', 'X'],
    ['O', '', 'X'],
    ['', '', '']]))

result_alg.append([2, 0])


test_alg.append(np.array([
    ['O', 'O', 'X'],
    ['O', 'X', 'X'],
    ['', '', '']]))

result_alg.append([2, 0])



def test_full_board():
    t = test_board[0]
    r = result[0]
    b = Board(t)
    assert b.possible_moves() == r


def test_semi_full_board():
    t = test_board[1]
    r = result[1]
    bl = Board(t)
    assert np.array_equal(bl.possible_moves(), r)


def test_minimax():
    for t, r in zip(test_alg, result_alg):
        b = Board(t)
        assert np.array_equal(minimax(b), r)


if __name__ == '__main__':
    test_semi_full_board()
    test_minimax()