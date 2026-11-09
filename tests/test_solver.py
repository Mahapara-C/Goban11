import random
from game.tictactoe import TicTacToe
from agents.solver import Solver, SolverAgent
from agents.mcts import RandomAgent
from harness.tournament import run_match, summarize


def plain_minimax(g):
    """Independent reference: no memoization, no shared code with Solver."""
    if g.winner is not None:
        return 0 if g.winner == 0 else (1 if g.winner == g.to_play else -1)
    best = -2
    for m in g.legal_moves():
        c = g.copy(); c.play(m)
        best = max(best, -plain_minimax(c))
    return best


def reachable(g, seen):
    k = (tuple(g.board), g.to_play)
    if k in seen:
        return
    seen[k] = g
    if g.winner is None:
        for m in g.legal_moves():
            c = g.copy(); c.play(m)
            reachable(c, seen)


def test_empty_board_is_a_draw():
    assert Solver().value(TicTacToe()) == 0


def test_memoized_values_match_plain_minimax_on_sampled_positions():
    seen = {}
    reachable(TicTacToe(), seen)
    rng = random.Random(0)
    s = Solver()
    sample = rng.sample(sorted(seen), 300)          # plain minimax is slow near the root
    for k in sample:
        assert s.value(seen[k]) == plain_minimax(seen[k]), k


def test_win_and_block_positions():
    g = TicTacToe()
    for m in (0, 3, 1, 4):                          # black to move, wins at 2
        g.play(m)
    s = Solver()
    assert s.value(g) == 1 and s.optimal_moves(g) == [2]
    h = TicTacToe()
    for m in (0, 4, 1):                             # white to move must block at 2
        h.play(m)
    assert s.optimal_moves(h) == [2]


def test_terminal_values_use_side_to_move():
    g = TicTacToe()
    for m in (0, 3, 1, 4, 2):                       # black has won; white "to move"
        g.play(m)
    assert g.winner == 1 and Solver().value(g) == -1


def test_solver_agent_never_loses_and_solver_vs_solver_draws():
    s = summarize(run_match(TicTacToe, SolverAgent(seed=1), RandomAgent(seed=2), n_pairs=30))
    assert s["total"]["loss"] == 0 and s["total"]["win"] > 0
    s = summarize(run_match(TicTacToe, SolverAgent(seed=1), SolverAgent(seed=2), n_pairs=10))
    assert s["total"]["draw"] == 20
