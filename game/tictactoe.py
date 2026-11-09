EMPTY, BLACK, WHITE = 0, 1, 2   # same constants as the Go engine


class TicTacToe:
    LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

    def __init__(self):
        self.board = [EMPTY] * 9
        self.to_play = BLACK
        self.winner = None            # None = ongoing, 0 = draw

    def copy(self):
        g = TicTacToe.__new__(TicTacToe)
        g.board = self.board[:]
        g.to_play = self.to_play
        g.winner = self.winner
        return g

    def legal_moves(self):
        return [i for i, v in enumerate(self.board) if v == EMPTY]

    def play(self, move):
        if self.board[move] != EMPTY:
            raise ValueError("illegal move")
        self.board[move] = self.to_play
        for a, b, c in self.LINES:
            if self.board[a] == self.board[b] == self.board[c] != EMPTY:
                self.winner = self.board[a]
        if self.winner is None and EMPTY not in self.board:
            self.winner = 0
        self.to_play = 3 - self.to_play
