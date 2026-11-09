"""Exact tic-tac-toe report for one player (a policy): outcomes vs Random and vs a perfect player,
and every position where it first blunders against perfect play. Torch-free; the player is just a policy."""
from collections import defaultdict
from game.tictactoe import TicTacToe
from harness.exact import (uniform, solver_policy, outcome_distribution, as_subject, first_blunders,
                           BLACK, WHITE)
from nn.ttt_data import canonical

NAME = {BLACK: "black", WHITE: "white"}


def render(board):
    sym = {0: ".", 1: "X", 2: "O"}
    return ["".join(sym[v] for v in board[r * 3:(r + 1) * 3]) for r in range(3)]


def dist_for(policy, opponent, color):
    return (outcome_distribution(TicTacToe, policy, opponent) if color == BLACK
            else outcome_distribution(TicTacToe, opponent, policy))


def report(name, policy, solver, top=5, out=print):
    perfect = solver_policy(solver)
    res = {"vs_random": {}, "vs_perfect": {}, "blunders": {}}
    out(f"== {name} ==")
    for label, opp, key in (("Random", uniform, "vs_random"), ("perfect SolverAgent", perfect, "vs_perfect")):
        avg = [0.0, 0.0, 0.0]
        for color in (BLACK, WHITE):
            w, d, l = as_subject(dist_for(policy, opp, color), color)
            res[key][color] = (w, d, l)
            avg = [a + x / 2 for a, x in zip(avg, (w, d, l))]
            out(f"  vs {label:20s} as {NAME[color]}: W {w:.4f}  D {d:.4f}  L {l:.4f}")
        res[key]["avg"] = tuple(avg)
        out(f"  vs {label:20s} average   : W {avg[0]:.4f}  D {avg[1]:.4f}  L {avg[2]:.4f}")
    for color in (BLACK, WHITE):
        bl = first_blunders(TicTacToe, policy, color, perfect, solver)
        mass = sum(b["prob"] for b in bl)
        loss = res["vs_perfect"][color][2]
        assert abs(mass - loss) < 1e-9, f"blunder mass {mass} != loss probability {loss}"   # exact self-check
        res["blunders"][color] = bl
        classes = {canonical(b["board"]) for b in bl}
        out(f"  as {NAME[color]}: P(loss to perfect) = {loss:.4f}  (expected losses per 100 games: {100 * loss:.1f});"
            f" {len(bl)} blunder positions in {len(classes)} symmetry classes; mass check OK")
        if bl:
            by_ply = defaultdict(float)
            for b in bl:
                by_ply[b["ply"]] += b["prob"]
            out("    probability of first blunder by ply (stones on board before the move): " +
                ", ".join(f"{p}: {x:.4f}" for p, x in sorted(by_ply.items())))
            for b in sorted(bl, key=lambda b: -b["prob"])[:top]:
                rows = render(b["board"])
                out(f"    reach prob {b['prob']:.4f}: played cell {b['move']}, optimal {b['optimal']} "
                    f"(value {b['value_before']:+d} -> {b['value_after']:+d})")
                out("        " + "\n        ".join(rows))
    return res
