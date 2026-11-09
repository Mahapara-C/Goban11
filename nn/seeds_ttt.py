"""Multi-seed tic-tac-toe experiment: for each seed, train one net on solver labels and one on MCTS labels
(same split, architecture and hyperparameters as nn/run_ttt.py; the seed changes the initial weights, the
batch order and the MCTS label noise), then measure each net EXACTLY (no sampling) against a perfect opponent
and against Random, and its optimal-move rate by number of stones on the board.
Run from the project root:  python -m nn.seeds_ttt [n_seeds]      (default 10)
Seed 0 should reproduce the nets from nn/exact_ttt.py; this is checked and printed at the end."""
import json
import sys
import time

import torch

from agents.net_agent import NetAgent
from agents.solver import Solver
from harness.exact import deterministic, BLACK, WHITE
from harness.exact_report import report
from nn.net import PolicyValueNet, to_tensors, train, predict
from nn.run_ttt import MCTS_SIMS, HIDDEN, EPOCHS, LR, BATCH, SEED, metrics
from nn.seed_stats import per_ply, summarize_runs
from nn.ttt_data import enumerate_positions, split_by_class, solver_labels, mcts_labels

EARLIER_P_LOSS_WHITE = {"solver": 0.1111, "mcts": 0.3492}      # seed 0, from the earlier exact_ttt output


def cached(policy):
    memo = {}

    def pol(g):
        k = (tuple(g.board), g.to_play)
        if k not in memo:
            memo[k] = policy(g)
        return memo[k]
    return pol


def hits(probs, truth_pi):
    return (truth_pi[torch.arange(len(truth_pi)), probs.argmax(1)] > 0).tolist()


def main(n_seeds=10):
    pos = [(b, tp) for b, tp, term in enumerate_positions() if not term]
    train_pos, test_pos = split_by_class(pos, 0.2, seed=SEED)          # split fixed; only the nets vary
    solver = Solver()
    Xtr, mtr, pi_tr, _ = to_tensors(train_pos, solver_labels(train_pos, solver))
    Xte, mte, pi_te, v_te = to_tensors(test_pos, solver_labels(test_pos, solver))
    runs = []
    for seed in range(n_seeds):
        for src in ("solver", "mcts"):
            t = time.time()
            labels = solver_labels(train_pos, solver) if src == "solver" else mcts_labels(train_pos, MCTS_SIMS, seed)
            model = PolicyValueNet(HIDDEN, seed)
            train(model, *to_tensors(train_pos, labels), epochs=EPOCHS, lr=LR, batch=BATCH, seed=seed)
            ptr, _ = predict(model, Xtr, mtr)
            pte, vte = predict(model, Xte, mte)
            m = metrics(pte, vte, pi_te, v_te)
            res = report(f"{src}-{seed}", cached(deterministic(NetAgent(model))), solver, out=lambda s: None)
            runs.append({"source": src, "seed": seed,
                         "p_loss_white": res["vs_perfect"][WHITE][2], "p_loss_black": res["vs_perfect"][BLACK][2],
                         "blunder_positions": len(res["blunders"][BLACK]) + len(res["blunders"][WHITE]),
                         "rnd_draw": res["vs_random"]["avg"][1], "rnd_loss": res["vs_random"]["avg"][2],
                         "test_opt": m["optimal_move_rate"], "test_vclass": m["value_class_acc"],
                         "ply_train": per_ply(train_pos, hits(ptr, pi_tr)), "ply_test": per_ply(test_pos, hits(pte, pi_te))})
            print(f"seed {seed} [{src}] done in {time.time() - t:.1f}s: P(loss as white) {runs[-1]['p_loss_white']:.4f}", flush=True)
    summarize_runs(runs)
    for src in ("solver", "mcts"):
        r0 = next((r for r in runs if r["source"] == src and r["seed"] == 0), None)
        if r0 is not None:
            ok = abs(r0["p_loss_white"] - EARLIER_P_LOSS_WHITE[src]) < 5e-4
            print(f"seed-0 check [{src}]: P(loss as white) {r0['p_loss_white']:.4f} vs earlier {EARLIER_P_LOSS_WHITE[src]:.4f}: "
                  f"{'matches' if ok else 'MISMATCH (nets differ from the earlier run)'}")
    with open("seeds_ttt_results.json", "w") as f:
        json.dump(runs, f)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
