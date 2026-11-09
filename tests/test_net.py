import random
import pytest
torch = pytest.importorskip("torch")          # whole file is skipped if PyTorch is not installed

from agents.solver import Solver
from agents.net_agent import NetAgent
from nn.ttt_data import enumerate_positions, solver_labels, to_game
from nn.net import PolicyValueNet, to_tensors, train, predict

NONTERMINAL = [(b, tp) for b, tp, term in enumerate_positions() if not term]


def test_forward_shapes_and_value_range():
    X, mask, pi, v = to_tensors(NONTERMINAL[:10], solver_labels(NONTERMINAL[:10]))
    logits, val = PolicyValueNet(seed=0)(X)
    assert logits.shape == (10, 9) and val.shape == (10,)
    assert bool((val >= -1).all() and (val <= 1).all())


def test_policy_has_no_mass_on_illegal_cells_and_agent_plays_legal_moves():
    model = PolicyValueNet(seed=0)                      # untrained
    X, mask, _, _ = to_tensors(NONTERMINAL, solver_labels(NONTERMINAL))
    probs, _ = predict(model, X, mask)
    assert float(probs[~mask].max()) == 0.0
    assert torch.allclose(probs.sum(1), torch.ones(len(NONTERMINAL)), atol=1e-5)
    agent = NetAgent(model)
    for b, tp in random.Random(0).sample(NONTERMINAL, 200):
        g = to_game(b, tp)
        assert agent.select_move(g) in g.legal_moves()


def test_training_is_reproducible():
    pos = NONTERMINAL[:64]
    X, mask, pi, v = to_tensors(pos, solver_labels(pos))
    h1 = train(PolicyValueNet(seed=1), X, mask, pi, v, epochs=3, seed=2)
    h2 = train(PolicyValueNet(seed=1), X, mask, pi, v, epochs=3, seed=2)
    assert h1 == h2


def test_single_position_is_fit_so_index_and_mask_alignment_is_right():
    g = to_game((1, 1, 0, 2, 2, 0, 0, 0, 0), 1)         # black to move; cell 2 completes the top row
    pos = [((1, 1, 0, 2, 2, 0, 0, 0, 0), 1)]
    (pi, v), = solver_labels(pos)
    assert pi[2] == 1.0 and v == 1.0                     # solver: cell 2 is the ONLY optimal move, value +1
    X, mask, pit, vt = to_tensors(pos, [(pi, v)])
    model = PolicyValueNet(seed=0)
    hist = train(model, X, mask, pit, vt, epochs=300, batch=1)
    assert hist[-1] < hist[0]
    assert NetAgent(model).select_move(g) == 2


def test_training_on_solver_labels_improves_optimal_move_rate():
    # Weak but safe direction-only check; tighten the thresholds after the first real run.
    pos = random.Random(0).sample(NONTERMINAL, 300)
    labels = solver_labels(pos)
    X, mask, pi, v = to_tensors(pos, labels)
    model = PolicyValueNet(seed=0)
    rate = lambda: float((pi[torch.arange(len(pos)), predict(model, X, mask)[0].argmax(1)] > 0).float().mean())
    before = rate()
    hist = train(model, X, mask, pi, v, epochs=100)
    assert hist[-1] < hist[0] and rate() > before
