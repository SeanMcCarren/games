import numpy as np

class GameState():
    def __init__(self, turn=0):
        self.turn = turn

    def move(self):
        pass

    def moves(self) -> np.ndarray:
        pass

    def score(self) -> float:
        pass

class Sudoku(GameState):
    def __init__(self, board=None, turn=0):
        if board is None:
            self.board = np.zeros(9, dtype=int)
        else:
            board = board
        super().__init__(turn=turn)
        self.executed_moves = []

    def moves(self):
        if self.eval() != 0: return []
        moves = [i for i, v in enumerate(self.board) if v==0]
        np.random.shuffle(moves)
        return moves
    
    def move_mask(self):
        return self.board == 0

    def move(self, move):
        self.board[move] = self.turn + 1
        self.turn = (self.turn + 1) % 2
        self.executed_moves.insert(0, move)
    
    def undo_move(self):
        move = self.executed_moves.pop(0)
        self.board[move] = 0
        self.turn = (self.turn + 1) % 2

    def eval(self):
        # if turn == 0 and there are three 1's in a row: +1
        # if turn == 1 and ...                         : -1
        # if turn == 0 and there are three 2's in a row: -1
        # if turn == 1 and ...                         : +1
        # so (turn+number)
        for i in range(3):
            if 0 != self.board[3*i] == self.board[3*i+1] == self.board[3*i+2] or \
                0 != self.board[i] == self.board[3+i] == self.board[6+i]:
                return 2 * ((self.board[4*i] + self.turn) % 2) - 1
        if 0 != self.board[0] == self.board[4] == self.board[8] or \
            0 != self.board[2] == self.board[4] == self.board[6]:
            return 2 * ((self.board[4] + self.turn) % 2) - 1
        return 0

    def __str__(self):
        chars = [' xo'[v] for v in self.board]
        return "-----\n%s %s %s\n%s %s %s\n%s %s %s" % tuple(chars)