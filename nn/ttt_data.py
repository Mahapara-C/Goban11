"""Tic-tac-toe position enumeration, symmetries, encoding and labels (solver and MCTS).
Labels are from the side to move: value in [-1, 1], policy = distribution over the 9 cells."""
import random
from game.tictactoe import TicTacToe
from agents.solver import Solver
from agents.mcts import uct_search

# The 8 symmetries of the 3x3 board as index permutations: new_board[i] = board[perm[i]].
_ROT = [6, 3, 0, 7, 4, 1, 8, 5, 2]          # rotate 90 degrees
_FLIP = [2, 1, 0, 5, 4, 3, 8, 7, 6]         # mirror left-right


def _compose(p, q):
    return [p[q[i]] for i in range(9)]


def _symmetries():
    out, cur = [], list(range(9))
    for _ in range(4):
        out.append(cur)
        out.append(_compose(cur, _FLIP))
        cur = _compose(cur, _ROT)
    return out


SYMS = _symmetries()


def canonical(board):
    """Smallest board over the 8 symmetries: identical for all symmetric copies of a position."""
    return min(tuple(board[p[i]] for i in range(9)) for p in SYMS)


def enumerate_positions():
    """All reachable positions as (board_tuple, to_play, is_terminal), found by search from the empty board."""
    seen, stack = {}, [TicTacToe()]
    while stack:
        g = stack.pop()
        k = (tuple(g.board), g.to_play)
        if k in seen:
            continue
        seen[k] = g.winner is not None
        if g.winner is None:
            for m in g.legal_moves():
                c = g.copy(); c.play(m)
                stack.append(c)
    return sorted((b, tp, term) for (b, tp), term in seen.items())


def to_game(board, to_play):
    g = TicTacToe()
    g.board, g.to_play = list(board), to_play
    return g


def encode(board, to_play):
    """18 floats: own stones (9), opponent stones (9), from the side to move's point of view."""
    return ([1.0 if v == to_play else 0.0 for v in board] +
            [1.0 if v == 3 - to_play else 0.0 for v in board])


def solver_labels(positions, solver=None):
    solver = solver or Solver()
    out = []
    for b, tp in positions:
        g = to_game(b, tp)
        opt = solver.optimal_moves(g)
        pi = [1 / len(opt) if i in opt else 0.0 for i in range(9)]
        out.append((pi, float(solver.value(g))))
    return out


def mcts_labels(positions, n_sims, seed=0):
    out = []
    for idx, (b, tp) in enumerate(positions):
        _, visits, v = uct_search(to_game(b, tp), n_sims, rng=random.Random(f"{seed}-{idx}"), return_stats=True)
        tot = sum(visits.values())
        out.append(([visits.get(i, 0) / tot for i in range(9)], v))
    return out


def split_by_class(positions, test_frac=0.2, seed=0):
    """Split positions into train/test so that no symmetric copy of a test position is in train."""
    classes = sorted({canonical(b) for b, _ in positions})
    random.Random(seed).shuffle(classes)
    n_test = round(len(classes) * test_frac)
    test_classes = set(classes[:n_test])
    train = [p for p in positions if canonical(p[0]) not in test_classes]
    test = [p for p in positions if canonical(p[0]) in test_classes]
    return train, test
