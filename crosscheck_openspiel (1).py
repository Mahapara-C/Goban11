"""Optional cross-check of game/go.py against OpenSpiel's Go (pip install open_spiel).

Run from the project root:  python tests/crosscheck_openspiel.py
Uses max_game_length=2000 because OpenSpiel defaults to 2*N*N moves and ends the game there.
Plays random games in both engines in lock-step (5x5, komi 0.5, area mode) and
compares (a) the legal-move sets at every position and (b) the winner at the end.
OpenSpiel numbers rows from the bottom, this engine from the top, so moves are mapped.
Known, explained difference: see project_brief.md ("Known differences vs OpenSpiel").
"""
import random
import sys
sys.path.insert(0, ".")
import pyspiel
from game.go import Go, PASS, BLACK, WHITE

N = 5


def to_os(i):
    if i == PASS:
        return N * N
    r, c = divmod(i, N)
    return (N - 1 - r) * N + c


def from_os(a):
    if a == N * N:
        return PASS
    r, c = divmod(a, N)
    return (N - 1 - r) * N + c


def main(n_games=300, seed=1):
    game = pyspiel.load_game("go", {"board_size": N, "komi": 0.5, "max_game_length": 2000})
    rng = random.Random(seed)
    positions = move_mismatch_games = 0
    result_checked = result_mismatch = 0
    for _ in range(n_games):
        s = game.new_initial_state()
        g = Go(N, "area", komi=0.5, superko=True)
        diverged = False
        while not s.is_terminal() and g.winner is None:
            mine = {to_os(m) for m in g.legal_moves()}
            theirs = set(s.legal_actions())
            positions += 1
            if mine != theirs:
                move_mismatch_games += 1
                diverged = True
                break
            a = rng.choice(sorted(theirs))
            s.apply_action(a)
            g.play(from_os(a))
        if not diverged:
            result_checked += 1
            os_winner = BLACK if s.returns()[0] > 0 else WHITE if s.returns()[0] < 0 else 0
            if os_winner != g.winner:
                result_mismatch += 1
    print(f"games: {n_games}; positions compared: {positions}")
    print(f"games where legal-move sets diverged: {move_mismatch_games}")
    print(f"games compared to the end: {result_checked}; winner mismatches: {result_mismatch}")


if __name__ == "__main__":
    main()
