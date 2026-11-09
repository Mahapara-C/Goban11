import statistics
from nn.seed_stats import per_ply, bootstrap_diff_ci, summarize_runs, _stats


def test_per_ply_groups_by_stone_count():
    pos = [((0,) * 9, 1), ((1, 0, 0, 0, 0, 0, 0, 0, 0), 2), ((0, 0, 0, 0, 2, 0, 0, 0, 0), 1), ((1, 2, 0, 0, 0, 0, 0, 0, 0), 1)]
    assert per_ply(pos, [True, True, False, False]) == {0: (1, 1.0), 1: (2, 0.5), 2: (1, 0.0)}


def test_bootstrap_ci_known_cases():
    d, lo, hi = bootstrap_diff_ci([0.0] * 10, [1.0] * 10, n_boot=500)
    assert d == 1.0 and lo == 1.0 and hi == 1.0                       # no overlap: CI is the point 1
    d, lo, hi = bootstrap_diff_ci([0, 1] * 5, [0, 1] * 5, n_boot=2000)
    assert d == 0.0 and lo < 0 < hi                                   # identical samples: CI straddles 0
    d, lo, hi = bootstrap_diff_ci([5.0] * 6, [3.0] * 6, n_boot=100)
    assert d == -2.0                                                  # sign: mean(b) - mean(a)
    assert bootstrap_diff_ci([1, 2, 3], [2, 3, 4], 300, seed=1) == bootstrap_diff_ci([1, 2, 3], [2, 3, 4], 300, seed=1)


def test_stats_match_statistics_module():
    xs = [0.1, 0.4, 0.2, 0.9]
    s = _stats(xs)
    assert s["mean"] == statistics.fmean(xs) and abs(s["sd"] - statistics.stdev(xs)) < 1e-12 and s["max"] == 0.9


def _run(source, seed, pw, rate):
    return {"source": source, "seed": seed, "p_loss_white": pw, "p_loss_black": 0.0, "blunder_positions": int(pw > 0),
            "rnd_draw": 0.05, "rnd_loss": 0.0, "test_opt": 0.99, "test_vclass": 0.88,
            "ply_train": {1: (9, rate), 2: (72, 1.0)}, "ply_test": {2: (10, 0.9)}}


def test_summarize_runs_on_synthetic_runs():
    runs = [_run("solver", s, 0.0, 1.0) for s in range(4)] + [_run("mcts", s, 0.25, 0.5) for s in range(4)]
    lines = []
    res = summarize_runs(runs, out=lines.append, n_boot=200)
    text = "\n".join(lines)
    assert res["stats"]["solver"]["p_loss_white"]["mean"] == 0.0 and res["stats"]["mcts"]["p_loss_white"]["mean"] == 0.25
    assert res["diff_ci"]["p_loss_white"][0] == 0.25
    assert "4 of 4 never lose to a perfect player" in text and "0 of 4 never lose" in text
    assert "   1 |   9 |  1.000 |  0.500" in text                    # per-ply row: solver 1.0, mcts 0.5 at 1 stone
