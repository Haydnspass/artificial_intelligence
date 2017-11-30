import numpy as np

board_rows = 3
board_cols = 3


class Board(object):

    PLAYER_1 = 'O'
    PLAYER_2 = 'X'

    win_count = 3

    def __init__(self):
        self.board = np.empty((board_rows, board_cols), dtype=str)

    def possible_moves(self):
        is_empty = np.stack(np.where(self.board == ''), axis=1)
        return is_empty

    def parse(self, move, player):
        self.board[move] = player

    @property
    def num_coins(self):
        return np.sum(b.board == self.PLAYER_1) + np.sum(b.board == self.PLAYER_2)

    def utility(self, player):
        u = 0
        for i in range(board_rows):
            u = u + np.sum(self.board[i,:] == player)
        for i in range(board_cols):
            u = u + np.sum(self.board[:, i] == player)
        # maybe add nebendiagonalen later
        u = u + np.sum(np.diag(self.board) == player)
        u = u + np.sum(np.diag(np.fliplr(self.board)) == player)

        if self.check_winner(self, player):
            u = np.inf

        return u

    def check_winner(self, player):
        if self.num_coins <= 4:
            return False

        for i in range(board_rows):
            if np.sum(self.board[i,:] == player) >= win_count:
                return True
        for i in range(board_cols):
            if np.sum(self.board[:,i] == player) >= win_count:
                return True
        # maybe add nebendiagonalen later
        if np.sum(np.diag(self.board) == player) >= win_count:
            return True
        if np.sum(np.diag(np.fliplr(self.board)) == player) >= win_count:
            return True

        return False


if __name__ == '__main__':
    b = Board()
    print(b.board)