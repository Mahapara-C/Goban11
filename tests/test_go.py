import random
from game.go import Go, BLACK, WHITE, EMPTY, PASS


def setup(g, black=(), white=(), to_play=BLACK):
    """Place stones directly (test helper) and record the position for superko."""
    for p in black:
        g.board[p] = BLACK
    for p in white:
        g.board[p] = WHITE
    g.to_play = to_play
    g.history.add(tuple(g.board))
    return g


def test_corner_capture():
    g = setup(Go(5, "area"), black=[5], white=[0])
    g.play(1)                                  # black takes white's last liberty
    assert g.board[0] == EMPTY and g.captures[BLACK] == 1


def test_suicide_illegal():
    g = setup(Go(5, "area"), white=[1, 5])
    assert 0 not in g.legal_moves()            # black at 0 would have no liberties


def test_capture_makes_zero_liberty_point_legal():
    # Black at 0 has no empty neighbours, but captures white 1 and 5 first.
    g = setup(Go(5, "area"), black=[2, 6, 10], white=[1, 5])
    assert 0 in g.legal_moves()
    g.play(0)
    assert g.board[1] == EMPTY and g.board[5] == EMPTY
    assert g.captures[BLACK] == 2


def test_multi_stone_group_capture():
    g = setup(Go(5, "area"), black=[5, 6], white=[0, 1])
    g.play(2)                                  # white group {0,1} loses last liberty
    assert g.board[0] == EMPTY and g.board[1] == EMPTY
    assert g.captures[BLACK] == 2


def test_cannot_play_on_occupied_point():
    g = setup(Go(5, "area"), black=[12])
    assert 12 not in g.legal_moves()


def test_ko_recapture_illegal():
    # Standard ko shape (row, col -> index r*5+c). Black plays 7, capturing white 6.
    # White recapturing at 6 would recreate the earlier position -> forbidden.
    g = setup(Go(5, "area", superko=True),
              black=[1, 5, 11],               # (0,1) (1,0) (2,1)
              white=[2, 6, 8, 12])            # (0,2) (1,1) (1,3) (2,2)
    g.play(7)                                  # black captures the white stone at 6
    assert g.board[6] == EMPTY and g.captures[BLACK] == 1
    assert g.to_play == WHITE
    assert 6 not in g.legal_moves()            # immediate retake is a ko violation
    try:
        g.play(6)
        assert False, "ko retake should raise"
    except ValueError:
        pass


def test_area_score_hand_built():
    # col 2 black wall, col 3 white wall; cols 0-1 are black territory, col 4 white's
    black = [r * 5 + 2 for r in range(5)]
    white = [r * 5 + 3 for r in range(5)]
    g = setup(Go(5, "area"), black=black, white=white)
    s = g.area_score()
    assert s[BLACK] == 5 + 10 and s[WHITE] == 5 + 5


def test_area_score_empty_board_is_neutral():
    s = Go(5, "area").area_score()
    assert s[BLACK] == 0 and s[WHITE] == 0


def test_two_passes_end_area_game_and_komi_decides():
    g = Go(5, "area", komi=0.5)
    g.play(PASS); g.play(PASS)
    assert g.winner == WHITE                   # 0-0 on board, komi 0.5 goes to white


def test_capture_mode_first_capture_wins():
    g = setup(Go(5, "capture"), black=[5], white=[0])
    g.play(1)
    assert g.winner == BLACK


def test_legal_moves_does_not_mutate_board():
    g = setup(Go(5, "area"), black=[6, 7], white=[11, 12])
    before = (g.board[:], set(g.history), dict(g.captures), g.to_play, g.passes)
    g.legal_moves()
    assert (g.board, g.history, g.captures, g.to_play, g.passes) == before


def test_copy_is_independent():
    g = Go(5, "area")
    h = g.copy()
    h.play(12)
    assert g.board[12] == EMPTY and len(g.history) == 1
    assert h.board[12] == BLACK and len(h.history) == 2
    assert g.to_play == BLACK and h.to_play == WHITE


def test_random_games_terminate_and_stay_consistent():
    rng = random.Random(0)
    for mode in ("capture", "area"):
        for _ in range(100):
            g = Go(5, mode)
            moves = 0
            while g.winner is None and moves < 2000:
                m = rng.choice(g.legal_moves())
                g.play(m)
                moves += 1
                # no stone group on the board may have zero liberties
                for i, v in enumerate(g.board):
                    if v != EMPTY:
                        assert g.group(g.board, i)[1], "group with no liberties"
            assert g.winner is not None, "game failed to terminate"


def _ko_position(**kw):
    g = setup(Go(5, "area", **kw), black=[1, 5, 11], white=[2, 6, 8, 12])
    g.play(7)                                  # black captures the white stone at 6
    return g


def test_ko_retake_legal_when_repetition_check_off():
    # Default is no repetition check; this pins that the flag really toggles the rule.
    assert 6 in _ko_position().legal_moves()
    assert 6 not in _ko_position(superko=True).legal_moves()


def test_default_komi_and_flags_survive_copy():
    assert Go(7, "area").komi == 9.0
    assert Go(5, "area").komi == 0.5 and Go(7, "area", komi=3).komi == 3
    g = Go(5, "area", komi=2.5, superko=True).copy()
    assert g.komi == 2.5 and g.superko is True and Go(5).copy().superko is False


def test_capture_mode_unaffected_by_repetition_flag():
    # Reasoned: first capture ends capture-mode games, and no board repeats without a capture.
    # Checked empirically on random play: legal-move sets identical with the flag on and off.
    rng = random.Random(1)
    for _ in range(100):
        a, b = Go(5, "capture"), Go(5, "capture", superko=True)
        while a.winner is None:
            assert a.legal_moves() == b.legal_moves()
            m = rng.choice(a.legal_moves())
            a.play(m); b.play(m)
        assert b.winner == a.winner
