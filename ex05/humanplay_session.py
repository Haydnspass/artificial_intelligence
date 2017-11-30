import numpy as np
from tictactoe import Board
from tictactoe import minimax
import click
import sys


def prompt_user(b):
    print(b.board)
    print('Enter ur move.')
    return tuple(map(int, input().split(',')))


def do_human_action(b):
    b.parse(prompt_user(b), b.next_player, deep=False)
    # print(b.board)
    return b

if __name__ == '__main__':
    b = Board(np.array([
    ['', '', ''],
    ['', 'X', 'O'],
    ['X', 'O', '']]))
    # first move of computer_player
    cm = minimax(b)
    print(cm)
    b.parse(cm, b.next_player, False)
    while True:
        b = do_human_action(b)
        if b.check_winner():
            break
        
        cm = minimax(b)
        b.parse(cm, b.next_player, False)
        print(b.board)
        if b.check_winner():
            break
