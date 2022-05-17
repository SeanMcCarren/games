# This should be an interface and hence a singleton class. However, for now,
# a module is a nice alternative to a singleton class! Downside: not lazily
# instantiated

import chess
from numpy.random import choice

def get_move(board):
    legal_moves = list(board.legal_moves)
    return choice(legal_moves)