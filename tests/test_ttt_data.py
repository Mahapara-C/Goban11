import random
from game.tictactoe import TicTacToe
from agents.solver import Solver
from nn.ttt_data import (SYMS, canonical, enumerate_positions, to_game, encode, solver_labels,
                         mcts_labels, split_by_class)


def test_symmetries_are_8_distinct_permutations_forming_a_group():
    assert len({tuple(p) for p in SYMS}) == 8 and all(sorted(p) == list(range(9)) for p in SYMS)
    S = {tuple(p) for p in SYMS}
    for p in SYMS:
        for q in SYMS:
            assert tuple(p[q[i]] for i in range(9)) in S       # closed under composition


def test_symmetries_map_winning_lines_to_winning_lines():
    lines = {frozenset(l) for l in TicTacToe.LINES}
    for p in SYMS:
        inv = {p[i]: i for i in range(9)}                        # old index -> new index
        assert {frozenset(inv[x] for x in l) for l in lines} == lines


def test_enumeration_counts_match_known_values():
    # Known values for tic-tac-toe (recalled from the literature, used here as an external check):
    # 5478 reachable positions in total, 958 of them terminal.
    pos = enumerate_positions()
    assert len(pos) == 5478 and sum(1 for p in pos if p[2]) == 958


def test_solver_value_is_symmetry_invariant_on_all_positions():
    s = Solver()
    for b, tp, term in enumerate_positions():
        if term:
            continue
        v = s.value(to_game(b, tp))
        for p in SYMS:
            assert s.value(to_game([b[p[i]] for i in range(9)], tp)) == v


def test_split_has_no_symmetry_leak_and_covers_everything():
    pos = [(b, tp) for b, tp, term in enumerate_positions() if not term]
    train, test = split_by_class(pos, 0.2, seed=0)
    assert len(train) + len(test) == len(pos) and test and train
    assert {canonical(b) for b, _ in train}.isdisjoint({canonical(b) for b, _ in test})


def test_encode_is_side_to_move_relative():
    g = TicTacToe(); g.play(0); g.play(4)                         # black at 0, white at 4, black to move
    x = encode(g.board, g.to_play)
    assert x[0] == 1 and x[9 + 4] == 1 and sum(x) == 2
    g.play(8)                                                     # now white to move: roles swap
    y = encode(g.board, g.to_play)
    assert y[4] == 1 and y[9 + 0] == 1 and y[9 + 8] == 1


def test_label_shapes_and_win_position():
    g = TicTacToe()
    for m in (0, 3, 1, 4):                                       # black wins at 2
        g.play(m)
    pos = [(tuple(g.board), g.to_play)]
    (pi, v), = solver_labels(pos)
    assert pi[2] == 1.0 and abs(sum(pi) - 1) < 1e-9 and v == 1.0
    (pi, v), = mcts_labels(pos, 300)
    assert abs(sum(pi) - 1) < 1e-9 and pi.index(max(pi)) == 2 and v > 0.5
    assert all(pi[i] == 0 for i in (0, 1, 3, 4))                 # occupied cells get no mass


def test_canonical_is_invariant_and_class_count_matches_known_value():
    pos = enumerate_positions()
    for b, tp, _ in pos:
        c = canonical(b)
        assert all(canonical(tuple(b[p[i]] for i in range(9))) == c for p in SYMS)
    # Known value (recalled from the literature, external check): 765 essentially different positions.
    assert len({canonical(b) for b, _, _ in pos}) == 765


def test_every_symmetric_copy_of_a_test_position_is_in_test():
    pos = [(b, tp) for b, tp, term in enumerate_positions() if not term]
    train, test = split_by_class(pos, 0.2, seed=0)
    train_set, test_set = set(train), set(test)
    for b, tp in test:
        for p in SYMS:
            copy = (tuple(b[p[i]] for i in range(9)), tp)
            assert copy in test_set and copy not in train_set
