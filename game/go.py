EMPTY, BLACK, WHITE = 0, 1, 2
PASS = -1


def opponent(c):
    return 3 - c


class Go:
    def __init__(self, size=5, mode="capture", komi=None, superko=False):
        # komi: 9 on 7x7 (believed optimal under area scoring, not proven); 0.5 is a
        # test-only placeholder for other sizes. superko=False = no repetition check.
        self.n, self.mode = size, mode
        self.komi = komi if komi is not None else (9.0 if size == 7 else 0.5)
        self.superko = superko
        self.board = [EMPTY] * (size * size)
        self.to_play = BLACK
        self.captures = {BLACK: 0, WHITE: 0}   # stones captured BY each color
        self.passes = 0
        self.winner = None                     # None = ongoing, 0 = draw
        self.history = {tuple(self.board)}     # positional superko
        self.neighbors = [self._nbrs(i) for i in range(size * size)]

    def _nbrs(self, i):
        r, c = divmod(i, self.n)
        out = []
        if r > 0: out.append(i - self.n)
        if r < self.n - 1: out.append(i + self.n)
        if c > 0: out.append(i - 1)
        if c < self.n - 1: out.append(i + 1)
        return out

    def copy(self):
        g = Go.__new__(Go)
        g.n, g.mode, g.komi, g.superko = self.n, self.mode, self.komi, self.superko
        g.board = self.board[:]
        g.to_play = self.to_play
        g.captures = dict(self.captures)
        g.passes = self.passes
        g.winner = self.winner
        g.history = set(self.history)
        g.neighbors = self.neighbors      # read-only, safe to share
        return g

    def group(self, board, i):
        """Flood fill: return (stones, liberties) of the group at i."""
        color = board[i]
        stones, libs, stack = {i}, set(), [i]
        while stack:
            s = stack.pop()
            for nb in self.neighbors[s]:
                v = board[nb]
                if v == EMPTY:
                    libs.add(nb)
                elif v == color and nb not in stones:
                    stones.add(nb)
                    stack.append(nb)
        return stones, libs

    def _try(self, pos, color):
        """Return (new_board, n_captured), or None if the move is illegal."""
        if self.board[pos] != EMPTY:
            return None
        b = self.board[:]
        b[pos] = color
        captured = 0
        for nb in self.neighbors[pos]:
            if b[nb] == opponent(color):
                stones, libs = self.group(b, nb)
                if not libs:
                    for s in stones:
                        b[s] = EMPTY
                    captured += len(stones)
        _, libs = self.group(b, pos)
        if not libs:
            return None                        # suicide
        if self.superko and tuple(b) in self.history:
            return None                        # positional superko (off by default)
        return b, captured

    def legal_moves(self):
        moves = [p for p in range(self.n * self.n)
                 if self._try(p, self.to_play) is not None]
        return moves + [PASS]

    def play(self, pos):
        if pos == PASS:
            self.passes += 1
        else:
            res = self._try(pos, self.to_play)
            if res is None:
                raise ValueError("illegal move")
            self.board, cap = res
            self.captures[self.to_play] += cap
            self.passes = 0
            self.history.add(tuple(self.board))
            if self.mode == "capture" and cap > 0:
                self.winner = self.to_play     # first capture wins
        if self.passes >= 2 and self.winner is None:
            self.winner = self._score_winner()
        self.to_play = opponent(self.to_play)

    def area_score(self):
        score = {BLACK: 0, WHITE: 0}
        seen = set()
        for i, v in enumerate(self.board):
            if v != EMPTY:
                score[v] += 1
            elif i not in seen:
                region, borders, stack = {i}, set(), [i]
                while stack:
                    s = stack.pop()
                    for nb in self.neighbors[s]:
                        w = self.board[nb]
                        if w == EMPTY and nb not in region:
                            region.add(nb)
                            stack.append(nb)
                        elif w != EMPTY:
                            borders.add(w)
                seen |= region
                if len(borders) == 1:
                    score[borders.pop()] += len(region)
        return score

    def _score_winner(self):
        if self.mode == "capture":
            a, b = self.captures[BLACK], self.captures[WHITE]
        else:
            s = self.area_score()
            a, b = s[BLACK], s[WHITE] + self.komi
        return BLACK if a > b else WHITE if b > a else 0
