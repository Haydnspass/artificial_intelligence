import numpy as np
from copy import deepcopy

board_rows = 3
board_cols = 3


class Board(object):

    PLAYER_1 = 'O'
    PLAYER_2 = 'X'

    win_count = 3

    def __init__(self, board=np.empty((board_rows, board_cols), dtype=str), computer_player=[]):
        self.board = board
        if computer_player == []:
            self.computer_player = self.next_player
            print('Computer is player: ', self.computer_player)
        else:
            self.computer_player = computer_player

    def possible_moves(self):
        if self.check_winner(self.PLAYER_1) or self.check_winner(self.PLAYER_2):
            return []

        is_empty = np.stack(np.where(self.board == ''), axis=1)
        if is_empty.size == 0:
            return []
        return is_empty

    def parse(self, move, player, deep=True):
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
        if self.next_player == self.PLAYER_1:
            return self.PLAYER_2 # to make player 1 first player
        else:
            return self.PLAYER_1

    @property
    def next_player(self):
        if np.sum(self.board == self.PLAYER_1) > np.sum(self.board == self.PLAYER_2):
            return self.PLAYER_2 # to make player 1 first player
        else:
            return self.PLAYER_1

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

            if self.check_winner(player):
                u = np.inf
        elif mode == 'hard':
            if self.check_winner(player):
                u = 1

        return u

    def check_winner(self, player=[PLAYER_1, PLAYER_2]):
        if self.num_coins <= 4:
            return False

        for p in player:
            for i in range(board_rows):
                if np.sum(self.board[i, :] == p) >= self.win_count:
                    return True
            for i in range(board_cols):
                if np.sum(self.board[:, i] == p) >= self.win_count:
                    return True
            # maybe add nebendiagonalen later
            if np.sum(np.diag(self.board) == p) >= self.win_count:
                return True
            if np.sum(np.diag(np.fliplr(self.board)) == p) >= self.win_count:
                return True

        return False


def minimax(graph):

    def min_play(graph):
        if graph.check_winner():
            return graph.utility(graph.computer_player)

        best_score = float('inf')

        for move in graph.possible_moves():
            clone = graph.parse(move, graph.next_player, graph.computer_player)
            score = max_play(clone)

            if score < best_score:
                best_move = move
                best_score = score

        return best_score

    def max_play(graph):
        if graph.check_winner():
            return graph.utility(graph.computer_player)

        best_score = float('-inf')

        for move in graph.possible_moves():
            clone = graph.parse(move, graph.next_player, graph.computer_player)
            score = min_play(clone)

            if score > best_score:
                best_move = move
                best_score = score

        return best_score

    moves = graph.possible_moves()
    best_move = moves[0]
    best_score = float('-inf')
    for move in moves:
        clone = graph.parse(move, graph.next_player, graph.computer_player)
        score = min_play(clone)
        if score > best_score:
            best_move = move
            best_score = score
    return best_move


if __name__ == '__main__':
    b = Board(np.array([
    ['X', '', ''],
    ['', 'X', 'O'],
    ['X', 'O', '']]))
    v = minimax(b)
    print(v)
