import torch
from nn.ttt_data import encode
from nn.net import predict


class NetAgent:
    """Greedy: plays the legal move with the highest policy probability. Deterministic, so `seed`
    is accepted only to fit the tournament factory interface make(seed=int)."""
    def __init__(self, model, seed=None):
        self.model = model

    def select_move(self, game):
        x = torch.tensor([encode(game.board, game.to_play)], dtype=torch.float32)
        mask = torch.tensor([[v == 0 for v in game.board]])
        probs, _ = predict(self.model, x, mask)
        return int(probs[0].argmax())
