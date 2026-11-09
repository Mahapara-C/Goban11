"""Tournament harness: paired, color-alternating matches between two agents.

Uses only the game interface (copy, legal_moves, play, to_play, winner) and the agent
interface select_move(game). Winner convention: None = ongoing, 0 = draw, 1 = BLACK, 2 = WHITE.
"""
import random
from statistics import NormalDist

BLACK, WHITE = 1, 2
Z95 = NormalDist().inv_cdf(0.975)


class IllegalMoveError(Exception):
    pass


def wilson(k, n, z=Z95):
    """Wilson score interval for k successes in n trials. k may be fractional (see summarize)."""
    if n <= 0:
        raise ValueError("n must be positive")
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def play_game(new_game, black, white, opening=(), max_plies=None, check_legal=True):
    """Play one game. Returns (winner, moves). winner is None if max_plies was hit first."""
    g = new_game()
    moves = []
    for m in opening:
        g.play(m)
        moves.append(m)
    while g.winner is None:
        if max_plies is not None and len(moves) >= max_plies:
            break
        agent = black if g.to_play == BLACK else white
        m = agent.select_move(g)
        if check_legal and m not in g.legal_moves():
            raise IllegalMoveError(f"{type(agent).__name__} chose illegal move {m} at ply {len(moves)}")
        g.play(m)
        moves.append(m)
    return g.winner, moves


def random_opening(new_game, k, rng, max_tries=100):
    """k random non-pass plies that leave the game ongoing (negative moves are passes: PASS = -1)."""
    for _ in range(max_tries):
        g = new_game()
        moves = []
        for _ in range(k):
            if g.winner is not None:
                break
            choices = [m for m in g.legal_moves() if m >= 0]
            if not choices:
                break
            m = rng.choice(choices)
            g.play(m)
            moves.append(m)
        if g.winner is None and len(moves) == k:
            return moves
    raise RuntimeError(f"could not generate a {k}-ply opening that leaves the game ongoing")


def _outcome(winner, a_color):
    if winner is None:
        return "unfinished"
    if winner == 0:
        return "draw"
    return "win" if winner == a_color else "loss"


def run_match(new_game, agent_a, agent_b, n_pairs, opening_plies=0, max_plies=None, seed=0):
    """n_pairs pairs; in each pair both games start from the same opening, A is black in one
    and white in the other. Returns a list of per-game records (from A's point of view)."""
    records = []
    for i in range(n_pairs):
        opening = random_opening(new_game, opening_plies, random.Random(f"{seed}-{i}"))
        for a_color in (BLACK, WHITE):
            black, white = (agent_a, agent_b) if a_color == BLACK else (agent_b, agent_a)
            winner, moves = play_game(new_game, black, white, opening, max_plies)
            records.append({"pair": i, "a_color": a_color, "opening": list(opening),
                            "winner": winner, "outcome": _outcome(winner, a_color), "moves": moves})
    return records


def _sub_seed(*parts):
    """Deterministic int seed from labels (str seeding is stable across processes; hash() is not)."""
    return random.Random("-".join(map(str, parts))).randrange(2 ** 31)


def _play_pair(task):
    new_game, make_a, make_b, i, opening_plies, max_plies, seed = task
    opening = random_opening(new_game, opening_plies, random.Random(f"{seed}-{i}"))
    out = []
    for a_color in (BLACK, WHITE):
        a = make_a(seed=_sub_seed(seed, i, a_color, "a"))     # fresh agents per game, seeded
        b = make_b(seed=_sub_seed(seed, i, a_color, "b"))     # by (seed, pair, color) only
        black, white = (a, b) if a_color == BLACK else (b, a)
        winner, moves = play_game(new_game, black, white, opening, max_plies)
        out.append({"pair": i, "a_color": a_color, "opening": list(opening),
                    "winner": winner, "outcome": _outcome(winner, a_color), "moves": moves})
    return out


def run_match_parallel(new_game, make_a, make_b, n_pairs, opening_plies=0, max_plies=None,
                       seed=0, workers=1):
    """Like run_match, but builds fresh agents per game via make_a(seed=int) / make_b(seed=int),
    so results depend only on (seed, pair, color), never on worker scheduling. workers=1 runs
    in-process. With workers>1 all arguments must be picklable: use functools.partial or classes,
    not lambdas (e.g. partial(Go, 5, "capture"), partial(MCTSAgent, 100))."""
    tasks = [(new_game, make_a, make_b, i, opening_plies, max_plies, seed) for i in range(n_pairs)]
    if workers == 1:
        chunks = [_play_pair(t) for t in tasks]
    else:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            chunks = pool.map(_play_pair, tasks, chunksize=1)    # map keeps task order
    return [r for chunk in chunks for r in chunk]


def summarize(records):
    """Counts and intervals for agent A.
    Win rate: Wilson on W / (W+D+L), draws count as non-wins.
    Score:    Wilson formula applied to (W + 0.5*D) / (W+D+L). This is an APPROXIMATION:
              the outcome is not binomial, so the interval is a heuristic.
    Unfinished games are reported separately and excluded from N.
    Both intervals assume independent games; the two games of a pair share an opening."""
    def tally(rs):
        return {o: sum(1 for r in rs if r["outcome"] == o) for o in ("win", "draw", "loss", "unfinished")}
    out = {"total": tally(records),
           "as_black": tally([r for r in records if r["a_color"] == BLACK]),
           "as_white": tally([r for r in records if r["a_color"] == WHITE])}
    t = out["total"]
    n = t["win"] + t["draw"] + t["loss"]
    out["n_finished"] = n
    if n:
        out["win_rate"] = t["win"] / n
        out["win_rate_ci"] = wilson(t["win"], n)
        out["score"] = (t["win"] + 0.5 * t["draw"]) / n
        out["score_ci_approx"] = wilson(t["win"] + 0.5 * t["draw"], n)
    return out


def format_summary(s, name_a="A", name_b="B"):
    t = s["total"]
    lines = [f"{name_a} vs {name_b}: W{t['win']} D{t['draw']} L{t['loss']} unfinished {t['unfinished']} "
             f"(N finished = {s['n_finished']})"]
    for key in ("as_black", "as_white"):
        c = s[key]
        lines.append(f"  {name_a} {key.replace('_', ' ')}: W{c['win']} D{c['draw']} L{c['loss']} U{c['unfinished']}")
    if s["n_finished"]:
        lo, hi = s["win_rate_ci"]
        lines.append(f"  win rate {s['win_rate']:.3f}  Wilson95 [{lo:.3f}, {hi:.3f}]  (draws = non-wins)")
        lo, hi = s["score_ci_approx"]
        lines.append(f"  score    {s['score']:.3f}  approx Wilson95 [{lo:.3f}, {hi:.3f}]  (W+0.5D; heuristic)")
    if t["unfinished"]:
        lines.append("  WARNING: unfinished games present; they are excluded from N.")
    return "\n".join(lines)
