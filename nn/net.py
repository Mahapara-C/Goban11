"""Small policy/value MLP for tic-tac-toe (PyTorch). Input: 18 floats (own/opponent stones from the
side to move's view). Output: 9 policy logits (illegal cells are masked out) and a value in [-1, 1]."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.ttt_data import encode


class PolicyValueNet(nn.Module):
    def __init__(self, hidden=128, seed=0):
        super().__init__()
        torch.manual_seed(seed)                # makes the initial weights reproducible
        self.body = nn.Sequential(nn.Linear(18, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.policy = nn.Linear(hidden, 9)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return self.policy(h), torch.tanh(self.value(h)).squeeze(-1)


def to_tensors(positions, labels):
    """positions: [(board, to_play)], labels: [(policy_9, value)] -> X, legal-mask, policy target, value target."""
    X = torch.tensor([encode(b, tp) for b, tp in positions], dtype=torch.float32)
    mask = torch.tensor([[v == 0 for v in b] for b, _ in positions])
    pi = torch.tensor([l[0] for l in labels], dtype=torch.float32)
    v = torch.tensor([l[1] for l in labels], dtype=torch.float32)
    return X, mask, pi, v


def train(model, X, mask, pi, v, epochs=100, lr=1e-3, batch=64, seed=0):
    """Loss = soft-target cross-entropy over legal moves + MSE on value. Returns mean loss per epoch."""
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    n, history = X.shape[0], []
    for _ in range(epochs):
        order = torch.randperm(n, generator=gen)
        total = 0.0
        for i in range(0, n, batch):
            idx = order[i:i + batch]
            logits, val = model(X[idx])
            logp = F.log_softmax(logits.masked_fill(~mask[idx], -1e9), dim=1)
            loss = -(pi[idx] * logp).sum(1).mean() + F.mse_loss(val, v[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        history.append(total / n)
    return history


@torch.no_grad()
def predict(model, X, mask):
    """Returns (policy probabilities with illegal cells at 0, value)."""
    model.eval()
    logits, val = model(X)
    return F.softmax(logits.masked_fill(~mask, -1e9), dim=1), val
