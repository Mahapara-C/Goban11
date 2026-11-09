"""Exact minimax solver for small games (tested on tic-tac-toe). Uses only the game interface
(copy, legal_moves, play, to_play, winner). Values are from the side to move: +1 win, 0 draw, -1 loss."""
import random


def default_key(g):
    return (tuple(g.board), g.to_play)     # tic-tac-toe / Go-style board games


class Solver:
    def __init__(self, key=default_key):
        self.key, self.cache = key, {}

    def value(self, g):
        if g.winner is not None:
            return 0 if g.winner == 0 else (1 if g.winner == g.to_play else -1)
        k = self.key(g)
        if k not in self.cache:
            best = -2
            for m in g.legal_moves():
                c = g.copy()
                c.play(m)
                best = max(best, -self.value(c))
            self.cache[k] = best
        return self.cache[k]

    def move_values(self, g):
        """{move: value of playing it, from the side to move}."""
        out = {}
        for m in g.legal_moves():
            c = g.copy()
            c.play(m)
            out[m] = -self.value(c)
        return out

    def optimal_moves(self, g):
        mv = self.move_values(g)
        best = max(mv.values())
        return [m for m, v in mv.items() if v == best]


class SolverAgent:
    """Plays uniformly at random among exactly-optimal moves."""
    def __init__(self, seed=None, solver=None):
        self.rng, self.solver = random.Random(seed), solver or Solver()

    def select_move(self, game):
        return self.rng.choice(self.solver.optimal_moves(game))
