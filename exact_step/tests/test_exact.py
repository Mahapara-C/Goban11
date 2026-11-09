import math
from game.tictactoe import TicTacToe
from agents.solver import Solver
from agents.mcts import RandomAgent
from harness.tournament import run_match, summarize
from harness.exact import (uniform, solver_policy, deterministic, outcome_distribution, as_subject,
                           best_response_policy, first_blunders, BLACK, WHITE)

S = Solver()
PERFECT = solver_policy(S)


def first(g):
    return {g.legal_moves()[0]: 1.0}


def last(g):
    return {g.legal_moves()[-1]: 1.0}


def test_distributions_sum_to_one_and_perfect_vs_perfect_is_a_draw():
    for pb, pw in ((uniform, uniform), (PERFECT, uniform), (uniform, PERFECT), (first, last)):
        assert abs(sum(outcome_distribution(TicTacToe, pb, pw)) - 1) < 1e-12
    bw, d, ww = outcome_distribution(TicTacToe, PERFECT, PERFECT)
    assert bw == 0.0 and ww == 0.0 and abs(d - 1) < 1e-12


def test_deterministic_policies_give_a_point_mass_equal_to_the_played_game():
    d = outcome_distribution(TicTacToe, first, last)
    g = TicTacToe()
    while g.winner is None:
        g.play((first if g.to_play == BLACK else last)(g).popitem()[0])
    assert d == ((1.0, 0.0, 0.0) if g.winner == BLACK else (0.0, 0.0, 1.0) if g.winner == WHITE else (0.0, 1.0, 0.0))


def test_exact_matches_sampled_play_through_the_harness():
    # Independent check: the exact recursion vs actually playing 3000 games with the harness.
    exact = as_subject(outcome_distribution(TicTacToe, first, uniform), BLACK)       # subject = black, plays first legal move
    class FirstAgent:
        def select_move(self, g): return g.legal_moves()[0]
    s = summarize(run_match(TicTacToe, FirstAgent(), RandomAgent(seed=3), n_pairs=1500))["as_black"]
    n = sum(s[o] for o in ("win", "draw", "loss"))
    for p, k in zip(exact, (s["win"], s["draw"], s["loss"])):
        assert abs(k / n - p) < 4 * math.sqrt(max(p * (1 - p), 1e-9) / n) + 1e-9, (exact, s)


def test_first_blunder_mass_equals_loss_probability_against_a_perfect_opponent():
    # Against a perfect opponent every blunder is decisive, so these two computations must agree.
    for color, pol in ((BLACK, first), (WHITE, first), (BLACK, last), (WHITE, last)):
        if color == BLACK:
            dist = outcome_distribution(TicTacToe, pol, PERFECT)
        else:
            dist = outcome_distribution(TicTacToe, PERFECT, pol)
        loss = as_subject(dist, color)[2]
        mass = sum(b["prob"] for b in first_blunders(TicTacToe, pol, color, PERFECT, S))
        assert loss > 0 and abs(loss - mass) < 1e-12, (color, loss, mass)


def test_blunder_records_are_genuine_blunders():
    for b in first_blunders(TicTacToe, first, WHITE, PERFECT, S):
        assert b["move"] not in b["optimal"] and b["value_after"] < b["value_before"]
        assert b["to_play"] == WHITE and 0 < b["prob"] <= 1


def test_perfect_player_has_no_blunders():
    assert first_blunders(TicTacToe, PERFECT, BLACK, PERFECT, S) == []
    assert first_blunders(TicTacToe, PERFECT, WHITE, uniform, S) == []


def test_best_response_is_an_upper_bound_and_consistent():
    score = lambda dist, c: as_subject(dist, c)[0] + 0.5 * as_subject(dist, c)[1]
    for color in (BLACK, WHITE):
        br = best_response_policy(color, uniform)
        def dist(pol):
            return outcome_distribution(TicTacToe, pol, uniform) if color == BLACK else outcome_distribution(TicTacToe, uniform, pol)
        best = score(dist(br), color)
        for other in (PERFECT, first, last, uniform):
            assert best >= score(dist(other), color) - 1e-12
        assert best > score(dist(PERFECT), color)          # exploiting Random beats merely perfect play


def _after(*moves):
    def make():
        g = TicTacToe()
        for m in moves:
            g.play(m)
        return g
    return make


def _plain_expectimax(g, color):
    """Independent reference: no memoization, no shared code with best_response_policy."""
    if g.winner is not None:
        return 0.5 if g.winner == 0 else (1.0 if g.winner == color else 0.0)
    vals = []
    for m in g.legal_moves():
        c = g.copy(); c.play(m)
        vals.append(_plain_expectimax(c, color))
    return max(vals) if g.to_play == color else sum(vals) / len(vals)


def test_best_response_achieves_the_independently_computed_optimum():
    for start in ((4,), (0, 4), (4, 0, 8)):
        make = _after(*start)
        for color in (BLACK, WHITE):
            br = best_response_policy(color, uniform)
            dist = (outcome_distribution(make, br, uniform) if color == BLACK
                    else outcome_distribution(make, uniform, br))
            w, d, _ = as_subject(dist, color)
            assert abs((w + 0.5 * d) - _plain_expectimax(make(), color)) < 1e-12, (start, color)


def test_report_runs_on_stand_in_players_and_self_checks():
    from harness.exact_report import report
    lines = []
    res = report("first-move", first, S, out=lines.append)
    assert abs(sum(res["vs_random"]["avg"]) - 1) < 1e-9 and res["blunders"][WHITE]
    assert any("mass check OK" in l for l in lines)
    res = report("perfect", PERFECT, S, out=lines.append)
    assert res["vs_perfect"][BLACK][2] == 0 and res["blunders"][BLACK] == [] and res["blunders"][WHITE] == []
