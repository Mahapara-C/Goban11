import math, random


class Node:
    def __init__(self, game, parent=None, move=None):
        self.game, self.parent, self.move = game, parent, move
        self.player_just_moved = 3 - game.to_play
        self.children = []
        self.untried = game.legal_moves() if game.winner is None else []
        self.visits, self.wins = 0, 0.0

    def best_child(self, c):
        lnN = math.log(self.visits)
        return max(self.children,
                   key=lambda ch: ch.wins / ch.visits + c * math.sqrt(lnN / ch.visits))


def uct_search(game, n_sims, c=1.41, rng=random, max_rollout=500, return_stats=False):
    root = Node(game.copy())
    if not root.untried:
        raise ValueError("no moves available (game is over)")
    for _ in range(n_sims):
        node = root
        # 1. selection: descend while fully expanded
        while not node.untried and node.children:
            node = node.best_child(c)
        # 2. expansion: add one new child
        if node.untried:
            move = node.untried.pop(rng.randrange(len(node.untried)))
            g = node.game.copy()
            g.play(move)
            child = Node(g, node, move)
            node.children.append(child)
            node = child
        # 3. rollout: random playout (capped, so odd games can't loop forever)
        g = node.game.copy()
        steps = 0
        while g.winner is None and steps < max_rollout:
            g.play(rng.choice(g.legal_moves()))
            steps += 1
        result = g.winner
        # 4. backpropagation: reward from each node's "player who just moved"
        while node is not None:
            node.visits += 1
            if result is None or result == 0:
                node.wins += 0.5
            elif result == node.player_just_moved:
                node.wins += 1.0
            node = node.parent
    best = max(root.children, key=lambda ch: ch.visits).move
    if return_stats:
        # visits per root move, and the root value for the side to move in [-1, 1]
        # (root.wins is credited to the player who just moved, so flip it)
        return best, {ch.move: ch.visits for ch in root.children}, 1 - 2 * root.wins / root.visits
    return best


class MCTSAgent:
    def __init__(self, n_sims=500, c=1.41, seed=None):
        self.n_sims, self.c = n_sims, c
        self.rng = random.Random(seed)

    def select_move(self, game):
        return uct_search(game, self.n_sims, self.c, self.rng)


class RandomAgent:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def select_move(self, game):
        return self.rng.choice(game.legal_moves())
