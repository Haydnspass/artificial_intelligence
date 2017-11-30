import numpy as np
from copy import deepcopy

board_rows = 3
board_cols = 3


class Board(object):

    PLAYER_1 = 'O'
    PLAYER_2 = 'X'

    win_count = 3

    def __init__(self, board=np.empty((board_rows, board_cols), dtype=str)):
        self.board = board

    def possible_moves(self):
        if self.check_winner(self.PLAYER_1) or self.check_winner(self.PLAYER_2):
            return []

        is_empty = np.stack(np.where(self.board == ''), axis=1)
        if is_empty.size == 0:
            return []
        return is_empty

    def parse(self, move, player, deep=False):
        if deep:
            cloned_board = deepcopy(self)
            cloned_board.board[tuple(move)] = player
            return cloned_board
        else:
            self.board[tuple(move)] = player

    @property
    def num_coins(self):
        return np.sum(self.board == self.PLAYER_1) + np.sum(self.board == self.PLAYER_2)

    @property
    def current_player(self):
        return self.PLAYER_2 if self.num_coins % 2 == 0 else self.PLAYER_1

    @property
    def next_player(self):
        return self.PLAYER_1 if self.num_coins % 2 == 0 else self.PLAYER_2

    def utility(self, player, mode='soft'):
        u = 0
        if mode == 'soft':
            for i in range(board_rows):
                u = u + np.sum(self.board[i,:] == player)
            for i in range(board_cols):
                u = u + np.sum(self.board[:, i] == player)
            # maybe add nebendiagonalen later
            u = u + np.sum(np.diag(self.board) == player)
            u = u + np.sum(np.diag(np.fliplr(self.board)) == player)

            if self.check_winner(self, player):
                u = np.inf
        elif mode == 'hard':
            if self.check_winner(player):
                u = 1

        return u

    def check_winner(self, player):
        if self.num_coins <= 4:
            return False

        for i in range(board_rows):
            if np.sum(self.board[i,:] == player) >= self.win_count:
                return True
        for i in range(board_cols):
            if np.sum(self.board[:,i] == player) >= self.win_count:
                return True
        # maybe add nebendiagonalen later
        if np.sum(np.diag(self.board) == player) >= self.win_count:
            return True
        if np.sum(np.diag(np.fliplr(self.board)) == player) >= self.win_count:
            return True

        return False





def minimax(board, depth, maxi_player):
    if depth == 0 or board.possible_moves() == []:
        return (board.utility(board.current_player, mode='hard'), board.board)

    if maxi_player:
        best_v = -np.inf
        for c in board.possible_moves():
            clone = Board(board.board)
            clone.parse(c, clone.next_player)
            v, _ = minimax(clone, depth - 1, False)
            best_v = max(best_v, v)
        return (best_v, board.board)

    else:
        best_v = np.inf
        for c in board.possible_moves():
            clone = Board(board.board)
            clone.parse(c, clone.next_player)
            v, _ = minimax(clone, depth - 1, True)
            best_v = min(best_v, v)
        return (best_v, board.board)


if __name__ == '__main__':
    b = Board()
    v, b = minimax(b, 7, True)
    print(v, b)