"""Exact (no sampling) outcome probabilities and blunder analysis for small games.
A policy is a function game -> {move: probability}; it must depend only on the position
(true for random, solver and greedy-net players), so results can be memoized per position."""
from agents.solver import default_key

BLACK, WHITE = 1, 2


def uniform(g):
    ms = g.legal_moves()
    return {m: 1 / len(ms) for m in ms}


def solver_policy(solver):
    """Uniform over exactly-optimal moves: the distribution SolverAgent samples from."""
    def pol(g):
        opt = solver.optimal_moves(g)
        return {m: 1 / len(opt) for m in opt}
    return pol


def deterministic(agent):
    return lambda g: {agent.select_move(g): 1.0}


def outcome_distribution(new_game, policy_black, policy_white, key=default_key):
    """Exact (P black wins, P draw, P white wins) when both sides sample moves from their policies."""
    memo = {}

    def rec(g):
        if g.winner is not None:
            return (1.0, 0.0, 0.0) if g.winner == BLACK else (0.0, 0.0, 1.0) if g.winner == WHITE else (0.0, 1.0, 0.0)
        k = key(g)
        if k not in memo:
            pol = policy_black if g.to_play == BLACK else policy_white
            tot = [0.0, 0.0, 0.0]
            for m, p in pol(g).items():
                assert m in g.legal_moves(), f"policy chose illegal move {m}"
                c = g.copy()
                c.play(m)
                r = rec(c)
                for i in range(3):
                    tot[i] += p * r[i]
            memo[k] = tuple(tot)
        return memo[k]

    return rec(new_game())


def as_subject(dist, subject_color):
    """(win, draw, loss) from the subject's point of view."""
    bw, d, ww = dist
    return (bw, d, ww) if subject_color == BLACK else (ww, d, bw)


def best_response_policy(subject_color, opponent_policy, key=default_key):
    """Expectimax: maximizes the subject's expected score (win 1, draw 0.5, loss 0) against a fixed
    opponent policy. Ties are split uniformly. Returns a policy for the subject's turns."""
    memo = {}

    def val(g):
        if g.winner is not None:
            return 0.5 if g.winner == 0 else (1.0 if g.winner == subject_color else 0.0)
        k = key(g)
        if k not in memo:
            if g.to_play == subject_color:
                memo[k] = max(q(g).values())
            else:
                tot = 0.0
                for m, p in opponent_policy(g).items():
                    c = g.copy()
                    c.play(m)
                    tot += p * val(c)
                memo[k] = tot
        return memo[k]

    def q(g):
        out = {}
        for m in g.legal_moves():
            c = g.copy()
            c.play(m)
            out[m] = val(c)
        return out

    def policy(g):
        vals = q(g)
        best = max(vals.values())
        top = [m for m, v in vals.items() if v >= best - 1e-12]
        return {m: 1 / len(top) for m in top}

    return policy


def first_blunders(new_game, subject_policy, subject_color, opponent_policy, solver, key=default_key):
    """Every position (with its probability of being reached) where the subject first plays a move
    that lowers the game-theoretic value of the position. Lines are merged by position, layer by
    layer; play stops at a first blunder."""
    g0 = new_game()
    frontier, out = {key(g0): (g0, 1.0)}, []
    while frontier:
        nxt = {}
        for g, prob in frontier.values():
            if g.winner is not None:
                continue
            mine = g.to_play == subject_color
            pol = subject_policy if mine else opponent_policy
            here = solver.value(g)
            for m, p in pol(g).items():
                c = g.copy()
                c.play(m)
                if mine and -solver.value(c) < here:
                    out.append({"board": tuple(g.board), "to_play": g.to_play, "move": m,
                                "optimal": solver.optimal_moves(g), "value_before": here,
                                "value_after": -solver.value(c), "prob": prob * p,
                                "ply": sum(1 for v in g.board if v)})
                    continue
                k = key(c)
                nxt[k] = (c, nxt[k][1] + prob * p) if k in nxt else (c, prob * p)
        frontier = nxt
    return out
