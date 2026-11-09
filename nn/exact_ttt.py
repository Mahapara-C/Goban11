"""Exact analysis of the two tic-tac-toe nets (solver-label and MCTS-label), no sampling.
Retrains them with the same data, hyperparameters and seed as nn/run_ttt.py, prints their test metrics so
you can confirm they match the earlier run, then calls harness.exact_report on each.
Run from the project root:  python -m nn.exact_ttt"""
from agents.net_agent import NetAgent
from agents.solver import Solver
from harness.exact import deterministic
from harness.exact_report import report
from nn.net import PolicyValueNet, to_tensors, train, predict
from nn.run_ttt import MCTS_SIMS, HIDDEN, EPOCHS, LR, BATCH, SEED, metrics, fmt
from nn.ttt_data import enumerate_positions, split_by_class, solver_labels, mcts_labels


def main():
    pos = [(b, tp) for b, tp, term in enumerate_positions() if not term]
    train_pos, test_pos = split_by_class(pos, 0.2, seed=SEED)
    Xte, mte, pi_te, v_te = to_tensors(test_pos, solver_labels(test_pos))
    solver = Solver()
    for name in ("solver", "mcts"):
        labels = solver_labels(train_pos) if name == "solver" else mcts_labels(train_pos, MCTS_SIMS, SEED)
        model = PolicyValueNet(HIDDEN, SEED)
        train(model, *to_tensors(train_pos, labels), epochs=EPOCHS, lr=LR, batch=BATCH, seed=SEED)
        print(f"\n[{name}] test metrics (compare with the earlier run_ttt output): "
              f"{fmt(metrics(*predict(model, Xte, mte), pi_te, v_te))}")
        report(f"net trained on {name} labels", deterministic(NetAgent(model)), solver)


if __name__ == "__main__":
    main()
