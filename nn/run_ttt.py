"""Tic-tac-toe experiment: one net trained on exact-solver labels, one on MCTS labels (same
architecture, positions, hyperparameters, seed). Evaluated on held-out symmetry classes against the
exact solver, then played against Random and against the perfect SolverAgent via the tournament harness.
Run from the project root:  python -m nn.run_ttt
Hyperparameters are fixed up front and NOT tuned on the test set."""
import time
from functools import partial

import torch

from agents.mcts import RandomAgent
from agents.net_agent import NetAgent
from agents.solver import SolverAgent
from harness.tournament import run_match_parallel, summarize, format_summary
from nn.net import PolicyValueNet, to_tensors, train, predict
from nn.ttt_data import enumerate_positions, split_by_class, solver_labels, mcts_labels
from game.tictactoe import TicTacToe

MCTS_SIMS, HIDDEN, EPOCHS, LR, BATCH, SEED = 500, 128, 100, 1e-3, 64, 0
N_PAIRS = 100                                   # 200 games per match


def metrics(policy_probs, value, truth_pi, truth_v):
    """vs the exact solver: how often the top move is solver-optimal; value error; value-class accuracy."""
    n = len(truth_v)
    top = policy_probs.argmax(1)
    return {"optimal_move_rate": float((truth_pi[torch.arange(n), top] > 0).float().mean()),
            "value_mse": float(((value - truth_v) ** 2).mean()),
            "value_class_acc": float((value.round().clamp(-1, 1) == truth_v).float().mean())}


def fmt(m):
    return f"optimal-move {m['optimal_move_rate']:.3f}  value MSE {m['value_mse']:.3f}  value-class {m['value_class_acc']:.3f}"


def main():
    pos = [(b, tp) for b, tp, term in enumerate_positions() if not term]
    train_pos, test_pos = split_by_class(pos, 0.2, seed=SEED)
    print(f"positions: {len(pos)} non-terminal; train {len(train_pos)}, test {len(test_pos)} (split by symmetry class)")

    t = time.time()
    lab = {"solver": (solver_labels(train_pos), solver_labels(test_pos)),
           "mcts": (mcts_labels(train_pos, MCTS_SIMS, SEED), mcts_labels(test_pos, MCTS_SIMS, SEED))}
    print(f"labels built in {time.time() - t:.1f}s (MCTS: {MCTS_SIMS} sims/position)")
    Xtr, mtr, _, _ = to_tensors(train_pos, lab["solver"][0])
    Xte, mte, pi_te, v_te = to_tensors(test_pos, lab["solver"][1])        # ground truth = solver, always

    # label quality: how good are the MCTS labels themselves, judged by the solver?
    _, _, pim, vm = to_tensors(test_pos, lab["mcts"][1])
    print("MCTS labels on test positions (not a net):", fmt(metrics(pim, vm, pi_te, v_te)))

    models = {"untrained": PolicyValueNet(HIDDEN, SEED)}
    for name in ("solver", "mcts"):
        model = PolicyValueNet(HIDDEN, SEED)
        X, mask, pi, v = to_tensors(train_pos, lab[name][0])
        t = time.time()
        hist = train(model, X, mask, pi, v, epochs=EPOCHS, lr=LR, batch=BATCH, seed=SEED)
        print(f"trained on {name} labels: loss {hist[0]:.3f} -> {hist[-1]:.3f} in {time.time() - t:.1f}s")
        models[name] = model

    _, _, pi_tr, v_tr = to_tensors(train_pos, lab["solver"][0])
    for name, model in models.items():
        print(f"[{name:9s}] train: {fmt(metrics(*predict(model, Xtr, mtr), pi_tr, v_tr))}")
        print(f"[{name:9s}] test : {fmt(metrics(*predict(model, Xte, mte), pi_te, v_te))}")

    for name in ("solver", "mcts"):
        for opp_name, opp in (("Random", RandomAgent), ("SolverAgent(perfect)", SolverAgent)):
            recs = run_match_parallel(TicTacToe, partial(NetAgent, models[name]), opp, n_pairs=N_PAIRS, seed=SEED)
            print(format_summary(summarize(recs), f"net[{name}]", opp_name))


if __name__ == "__main__":
    main()
