import random
from game.tictactoe import TicTacToe
from game.go import Go, PASS
from agents.mcts import MCTSAgent, RandomAgent, uct_search


def play(black, white):
    g = TicTacToe()
    while g.winner is None:
        agent = black if g.to_play == 1 else white
        g.play(agent.select_move(g))
    return g.winner


def test_mcts_never_loses_to_random():
    for i in range(50):
        assert play(MCTSAgent(500, seed=i), RandomAgent(seed=i)) != 2   # MCTS as black
        assert play(RandomAgent(seed=i), MCTSAgent(500, seed=i)) != 1   # MCTS as white


def test_mcts_vs_mcts_mostly_draws():
    results = [play(MCTSAgent(1000, seed=i), MCTSAgent(1000, seed=100 + i))
               for i in range(20)]
    assert results.count(0) >= 18


def test_takes_immediate_win():
    g = TicTacToe()
    for m in (0, 3, 1, 4):          # black can win at 2
        g.play(m)
    assert MCTSAgent(300, seed=0).select_move(g) == 2


def test_mcts_smoke_on_go():
    g = Go(5, "capture")
    move = uct_search(g, 50)
    assert move in g.legal_moves() or move == PASS


def test_return_stats_consistent_with_plain_search():
    g = TicTacToe()
    for m in (0, 3, 1, 4):                       # black to move, wins at 2
        g.play(m)
    move, visits, value = uct_search(g, 300, rng=random.Random(5), return_stats=True)
    assert move == uct_search(g, 300, rng=random.Random(5))      # same seed, same move
    assert sum(visits.values()) == 300           # every simulation passes through one root child
    assert move == max(visits, key=visits.get)
    assert -1 <= value <= 1 and value > 0.5      # side to move is winning here
