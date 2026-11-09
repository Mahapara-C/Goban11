import pytest
from game.tictactoe import TicTacToe
from game.go import Go
from agents.mcts import MCTSAgent, RandomAgent
from functools import partial
from harness.tournament import (wilson, play_game, random_opening, run_match, run_match_parallel,
                                summarize, IllegalMoveError, BLACK, WHITE)


class FirstMoveAgent:
    def __init__(self, seed=None):
        pass

    def select_move(self, game):
        return game.legal_moves()[0]


class LastMoveAgent(FirstMoveAgent):
    def select_move(self, game):
        return game.legal_moves()[-1]


class IllegalAgent:
    def select_move(self, game):
        return 99


def test_wilson_matches_statsmodels_reference_values():
    # Reference values computed independently with statsmodels 0.15.0 proportion_confint(method="wilson")
    ref = {(0, 10): (0.0, 0.27753279986288926),
           (5, 10): (0.23659309051256394, 0.7634069094874361),
           (10, 10): (0.7224672001371106, 1.0),
           (7, 20): (0.18119182410108203, 0.5671457233147638),
           (1, 3): (0.06149194472039626, 0.7923403991979523)}
    for (k, n), (lo, hi) in ref.items():
        got = wilson(k, n)
        assert abs(got[0] - lo) < 1e-9 and abs(got[1] - hi) < 1e-9, (k, n, got)


def test_wilson_bounds_and_containment():
    for n in range(1, 31):
        for k in range(n + 1):
            lo, hi = wilson(k, n)
            eps = 1e-12   # k=0 / k=n endpoints are exact in theory, off by ~1e-17 in floats
            assert 0.0 <= lo <= k / n + eps and k / n - eps <= hi <= 1.0


def test_color_bookkeeping_mcts_vs_random():
    # A = MCTS, B = random. If colors/outcomes were mislabeled, A would show losses.
    recs = run_match(TicTacToe, MCTSAgent(300, seed=1), RandomAgent(seed=2), n_pairs=30)
    s = summarize(recs)
    assert len(recs) == 60
    assert s["total"]["loss"] == 0 and s["total"]["win"] > 0
    assert s["as_black"]["win"] + s["as_black"]["draw"] == 30
    assert s["as_white"]["win"] + s["as_white"]["draw"] == 30


def test_identical_deterministic_agents_split_evenly():
    # Same deterministic agent on both sides: A's wins must equal A's losses across a pair.
    s = summarize(run_match(TicTacToe, FirstMoveAgent(), FirstMoveAgent(), n_pairs=5))
    assert s["total"]["win"] == s["total"]["loss"]
    assert s["total"]["win"] + s["total"]["draw"] + s["total"]["loss"] == 10


def test_illegal_move_raises():
    with pytest.raises(IllegalMoveError):
        play_game(TicTacToe, IllegalAgent(), FirstMoveAgent())


def test_move_cap_gives_unfinished_not_draw():
    recs = run_match(TicTacToe, FirstMoveAgent(), FirstMoveAgent(), n_pairs=2, max_plies=3)
    s = summarize(recs)
    assert s["total"]["unfinished"] == 4 and s["n_finished"] == 0 and "win_rate" not in s


def test_pair_shares_opening_and_it_is_reproducible():
    a = run_match(TicTacToe, RandomAgent(seed=0), RandomAgent(seed=1), n_pairs=6, opening_plies=2, seed=7)
    b = run_match(TicTacToe, RandomAgent(seed=5), RandomAgent(seed=6), n_pairs=6, opening_plies=2, seed=7)
    for i in range(0, 12, 2):
        assert a[i]["opening"] == a[i + 1]["opening"] and len(a[i]["opening"]) == 2
        assert a[i]["moves"][:2] == a[i]["opening"] == a[i + 1]["moves"][:2]
        assert a[i]["a_color"] == BLACK and a[i + 1]["a_color"] == WHITE
    assert [r["opening"] for r in a] == [r["opening"] for r in b]   # depends on seed only, not agents


def test_random_opening_has_no_passes_and_leaves_game_ongoing():
    import random
    for i in range(30):
        mv = random_opening(lambda: Go(5, "capture"), 4, random.Random(i))
        assert len(mv) == 4 and all(m >= 0 for m in mv)


def test_go_smoke_run_completes():
    # Smoke only: checks the harness runs on Go, says nothing about strength.
    recs = run_match(lambda: Go(5, "capture"), MCTSAgent(20, seed=1), RandomAgent(seed=2),
                     n_pairs=2, max_plies=200)
    s = summarize(recs)
    assert sum(s["total"].values()) == 4


def test_parallel_results_independent_of_worker_count():
    kw = dict(new_game=TicTacToe, make_a=partial(MCTSAgent, 40), make_b=RandomAgent,
              n_pairs=8, opening_plies=1, seed=3)
    r1 = run_match_parallel(workers=1, **kw)
    r2 = run_match_parallel(workers=2, **kw)
    r3 = run_match_parallel(workers=3, **kw)
    assert r1 == r2 == r3 and len(r1) == 16
    assert [r["pair"] for r in r1] == [i for i in range(8) for _ in (0, 1)]   # order preserved


def test_parallel_seed_matters_and_pairs_differ():
    kw = dict(new_game=TicTacToe, make_a=RandomAgent, make_b=RandomAgent, n_pairs=10)
    r0 = run_match_parallel(seed=0, **kw)
    r1 = run_match_parallel(seed=1, **kw)
    assert [r["moves"] for r in r0] != [r["moves"] for r in r1]
    assert len({tuple(r["moves"]) for r in r0}) > 3      # pairs are not clones of one game


def test_parallel_seating_and_outcomes_independent_of_agent_strength():
    # Deterministic agents: A always plays the first legal move, B the last. Seating is visible in
    # the first move (0 vs 8); outcomes are recomputed by replaying the moves on a fresh game.
    recs = run_match_parallel(TicTacToe, FirstMoveAgent, LastMoveAgent, n_pairs=3, workers=2)
    assert len(recs) == 6
    for r in recs:
        assert r["moves"][0] == (0 if r["a_color"] == BLACK else 8)
        g = TicTacToe()
        for m in r["moves"]:
            g.play(m)
        assert g.winner == r["winner"]
        expected = "draw" if g.winner == 0 else ("win" if g.winner == r["a_color"] else "loss")
        assert r["outcome"] == expected


SEEDS_SEEN = []


def _recording_factory(seed):
    SEEDS_SEEN.append(seed)
    return RandomAgent(seed=seed)


def test_parallel_agent_seeds_distinct_per_pair_color_and_role():
    SEEDS_SEEN.clear()
    run_match_parallel(TicTacToe, _recording_factory, _recording_factory, n_pairs=5, workers=1)
    assert len(SEEDS_SEEN) == 20 and len(set(SEEDS_SEEN)) == 20   # 5 pairs x 2 colors x (A, B)


def test_parallel_go_smoke():
    recs = run_match_parallel(partial(Go, 5, "capture"), partial(MCTSAgent, 20), RandomAgent,
                              n_pairs=2, max_plies=200, workers=2)
    assert sum(summarize(recs)["total"].values()) == 4
