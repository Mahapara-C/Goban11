# Project brief: Go (baduk) agent comparison

Last updated: 2026-11-09. Status: engine, tic-tac-toe, generic UCT-MCTS, tournament harness and tests built and passing.
Nothing below is marked "verified" unless it was actually run (see "Verification status").

## Goal
AI lab project, repo/resume ready, about two weeks, on free Kaggle/Colab GPUs.
Compare on small-board Go: (1) classical MCTS (random rollouts), (2) learned policy/value
network, (3) hybrid (network-guided MCTS / PUCT). Equal-compute comparisons, scaling curves,
tactical test positions, alternating colors, Wilson confidence intervals.

## Project constraints (apply to EVERY chat, both roles)
1. Rigorous rule checking
   - The rules spec is Tromp-Taylor (area mode) plus first-capture-wins (capture mode).
     Rule questions are answered from the spec, not from memory or intuition.
   - Every rule behavior (capture, suicide, ko/superko, scoring, passing, game end) has a
     unit test before any agent code depends on it.
   - Engine behavior is cross-checked against an independent implementation (OpenSpiel)
     on random positions before results are trusted.
2. Sound logic build-up
   - Build order: rules -> tests -> baselines -> tournament harness -> network -> hybrid ->
     evaluation. Do not start a layer until the one below it is tested.
   - Every new component is first validated on a simpler game (tic-tac-toe / Connect-4)
     where correctness is easy to check.
   - Each step states what it assumes, what it checks, and how a failure would show up.
3. No hallucination
   - Do not invent facts, citations, APIs, function signatures, library behavior, paper
     results, or numbers. Separate clearly: verified / reasoned but unverified / unknown.
   - Code that has not been run is described as untested.
   - Experimental claims are made only from measured results, with sample size and
     confidence interval.
4. Ask first when unsure
   - If there is even slight uncertainty about a rule, requirement, library behavior, or the
     user's intent, stop and ask before proceeding. Do not guess or silently pick an assumption.
   - When asking, name exactly which part is uncertain so it can be answered quickly.

## Chat roles
- BUILD: write and debug code. Minimal, targeted edits to existing code instead of rewrites;
  show the changed section inline in the reply.
- QUESTIONS: user is learning Go rules, MCTS, NN basics. Explain from first principles with
  small concrete examples; no large code unless asked.

## Game interface (every game must follow)
    g.copy()  g.legal_moves()  g.play(move)  g.to_play  g.winner
    winner: None = ongoing, 0 = draw, else BLACK=1 / WHITE=2.   Go passing move: PASS = -1.

## Files (repo layout: game/, agents/, tests/)
- game/go.py          Go engine. Modes: "capture" (first capture wins) and "area"
                      (Tromp-Taylor area scoring + komi). No-suicide. Positional superko is a
                      flag, OFF by default (superko=False = no repetition check).
                      komi default: 9.0 on 7x7, 0.5 placeholder otherwise (test-only).
- game/tictactoe.py   Tic-tac-toe, same interface.
- agents/mcts.py      Node, uct_search (UCT + random rollouts), MCTSAgent, RandomAgent.
- harness/tournament.py   Paired color-alternating matches, W/D/L/unfinished, Wilson CIs.
                      run_match (persistent agents, serial) and run_match_parallel (fresh agents per game
                      via factory(seed=int), results independent of worker count; factories must be
                      picklable: functools.partial or classes, not lambdas).
- agents/solver.py    Exact minimax Solver (memoized) + SolverAgent (random among optimal moves).
- agents/net_agent.py NetAgent (greedy over masked policy). UNTESTED (needs torch).
- nn/ttt_data.py      Tic-tac-toe positions, 8 symmetries, encoding, solver/MCTS labels, split by symmetry class.
- nn/net.py           PolicyValueNet (MLP 18->128->128, 9 policy logits + tanh value), train, predict. UNTESTED.
- nn/run_ttt.py       Experiment: net on solver labels vs net on MCTS labels (RUN by user 2026-11-09).
- harness/exact.py, harness/exact_report.py   Exact (no sampling) outcome probabilities, best response,
                      first-blunder enumeration vs a perfect opponent. Torch-free, tested.
- nn/seeds_ttt.py     Multi-seed experiment (both label sources, exact metrics, per-ply accuracy); nn/seed_stats.py is its
                      torch-free, tested bookkeeping. NOT RUN yet (needs torch).
- nn/exact_ttt.py     Retrains both nets and runs the exact report (RUN by user 2026-11-09; retrained nets matched the
                      run_ttt test metrics exactly).
- tests/test_go.py    Engine tests (16).   tests/test_mcts.py  Search tests (4).
- tests/test_tournament.py   Harness tests (14).
- tests/test_solver.py (5), tests/test_ttt_data.py (9), tests/test_net.py (5, skipped without torch, NEVER RUN).
- tests/crosscheck_openspiel.py   Optional lock-step comparison against OpenSpiel.
- pytest.ini, requirements.txt. Run tests: `python -m pytest -q` from the project root.

## Verification status (what was actually run)
- User ran full suite locally: 70 passed (2026-11-09, after replacing stale test_go.py).
- Exact tools (sandbox): 9 tests; 5 mutations caught after strengthening one test (best-response check
  now compares with an independent unmemoized expectimax). Exact draw baseline vs uniform Random:
  perfect/SolverAgent W .9678 D .0322 as black, W .7775 D .2225 as white (avg D .1274); best response
  avg W .9634 D .0330 L .0036.
- run_ttt output (user, 1 seed, 1 run): both nets ~0.99 optimal-move on held-out; solver net lost 5/200 and
  MCTS net 27/200 to SolverAgent (all as white); caveat: games not independent. Nets drew 8% / 5% vs Random.
- exact_ttt output (user, same seed-0 nets). Exact P(loss to a perfect opponent): both nets 0 as black.
  As white: solver-label net 0.1111 (1 blunder position: reply to a centre opening is an edge instead of a
  corner); MCTS-label net 0.3492 (3 opening replies wrong: centre opening -> edge; corner openings at cells 6
  and 8 -> not the centre; plus 1 ply-3 trap, X0 O4 X8 -> O plays a corner). Cells 0 and 2 openings were
  answered correctly, so the learned policy is not symmetric. All blunder positions were in the TRAINING
  set (all 9 ply-1 positions are in train): fitting failures, not held-out generalization. MCTS label top move
  was correct in the blunder positions but diffuse (mass 0.20-0.23); MCTS value labels at ply 1 were -0.28..-0.49
  vs true 0 (biased). Exact vs Random (avg over colors): solver net W .924 D .071 L .005; MCTS net W .935
  D .055 L .011 (perfect player never loses to Random; best response L .004).
- The sampled run_ttt losses (5/100 and 27/100 as white) are reproduced EXACTLY from the realized black openings
  (5 centre openings; 26 openings in {4,6,8} + 1 ply-3 loss). The 100 sampled openings under-drew the centre
  (5 vs 11.1 expected), and both nets shared the same opening sequence (same seed): sampled counts there say
  little about strength.
- seeds_ttt output (user, 10 seeds per label source, fixed split, untuned hyperparameters; seed 0 reproduced the
  earlier nets exactly). Exact P(loss to perfect) as white: solver-label nets mean .026 (sd .046, range 0-.111;
  6/10 never lose); MCTS-label nets mean .337 (sd .128, range .155-.574; 0/10 never lose). Ranges do not overlap;
  diff of means +.311, percentile-bootstrap 95% CI [+.234, +.392] (approximate). As black: all 20 nets 0.
  => the label-source effect on play strength is real in this setup (not single-seed noise). The mechanism
  (diffuse top-mass, label noise, value bias) is UNTESTED.
- Held-out optimal-move rate ranks the other way (MCTS .996 vs solver .989; value-class equal .886/.888):
  held-out accuracy (all positions weighted equally) is a misleading proxy for play strength here.
- Per-ply TRAIN optimal-move rate (mean over seeds), solver vs MCTS: 1 stone .978 vs .678 (9 positions); 2: 1.000
  vs .904; 3: .994 vs .944; >=4: ~1.000 both. Errors concentrate in shallow positions even on training data.
  Both sources had identical positions/weights, so the gap is due to the labels; "shallow positions are
  under-weighted in the loss" is still an untested hypothesis and cannot by itself explain the source gap.
- Vs Random (exact, avg colors): draw rate solver .0651 vs MCTS .0636 (diff -.0015, CI [-.0061,+.0034]): NO
  difference; the earlier seed-0 gap (and the idea that MCTS labels teach exploiting random play) is retracted.
  Loss rate solver .0014 vs MCTS .0069 (diff +.0056, CI [+.0029,+.0083]). Perfect player: draw .1274, loss 0.
- Go probe (sandbox, existing engine + Solver keyed on (board, to_play, passes)): 3x3 capture mode solved in 1.1 s,
  48,658 positions, value 0 (draw). 4x4 did NOT finish in 240 s (size unknown). 5x5 exact solving is not
  available with this engine (reasoned, not measured). NOTE: Solver's default key omits the pass count, which is
  wrong for Go; always pass key=(board, to_play, passes).
- Earlier sandbox line: 49 passed, 1 file skipped (test_net.py: torch could not be installed in the sandbox, disk full) on
  2026-11-09. test_mcts has 5 tests now (return_stats test added).
- Solver: matches unmemoized minimax on 300 sampled positions; empty board = draw; mutation-checked.
- Data layer: enumeration gives 5478 positions / 958 terminal / 765 symmetry classes (match recalled
  literature values, used as external checks); solver value symmetry-invariant on all positions;
  split by symmetry class has no leak (an earlier vacuous leak test was found by mutation and fixed).
- Earlier line: 34/34 on 2026-11-09 before the solver/data/net work. Earlier 17/17 on 2026-11-02 (pytest 9.2.0).
- Harness: Wilson matches statsmodels 0.15.0 on 5 cases; mutation-checked (wrong color compare,
  wrong seating, unfinished-as-draw: each caught). Repetition flag mutation-checked (3 mutations caught).
- OpenSpiel cross-check re-run 2026-11-09 with superko=True: identical to the numbers above.
  With superko=False: 41/300 games diverge on legal moves (10,226 positions compared), 259 compared
  to the end, 0 winner mismatches. Expected: OpenSpiel enforces a repetition rule, this config does not.
- Parallel harness: identical records for workers=1/2/3 (tic-tac-toe) and 1/2 (Go 5x5 capture, 16 games);
  5 mutations caught. Speedup NOT measured (sandbox had 1 CPU). Python 3.12 warns that fork() from a
  multi-threaded process may deadlock; not observed, not investigated.
- Capture mode: legal moves identical with the flag on/off over 100 random games (empirical, not a proof).
- Go smoke (harness, capture mode, MCTS 100 sims vs Random, 10 pairs, cap 2*N^2 plies, one run each):
  5x5 W20 D0 L0 (5.3 s); 7x7 W19 D0 L1 (32.8 s). Provisional; says nothing about strength.
  7x7 ~6x slower per game here.
- Ko test mutation-checked: with the superko history removed, the retake becomes legal,
  so the test really exercises the ko rule.
- Random-game test: 100 games per mode on 5x5, all terminate, no group ever has 0 liberties.
- OpenSpiel cross-check (open_spiel pip package, 5x5, komi 0.5, 300 random games, 11,112
  positions, `max_game_length` raised to 2000):
  * Winner: 296 games compared to the end, 0 mismatches. (4 games had already diverged on
    legal moves and were excluded.)
  * Legal moves: identical except 4 positions. In all 4, OpenSpiel allows a move that this
    engine forbids; the move is not suicide, recreates an earlier board, and does NOT
    repeat a board with the same side to move. This is CONSISTENT WITH OpenSpiel using
    situational superko, but that was inferred from behavior, not confirmed from its source.
  * This engine never allowed a move OpenSpiel forbade, in these samples.
  * OpenSpiel's default `max_game_length` is 50 on a 5x5 board and it ends games there;
    that caused an earlier false alarm of 52 winner mismatches. Always raise it.
- NOT verified: speed/performance; seki, dead-stone disputes, and long cycles beyond random
  play; Go engine on 7x7+; MCTS strength on Go; the stochastic MCTS tests rely on fixed seeds
  (the 18/20 draws threshold was set from experience, not derived).

## Decisions made
- Atari Go (capture mode) pipeline first, then area mode on 5x5/7x7 as stretch.
- NO repetition check by default (user decision, 2026-11-09). The positional-superko code (Tromp-Taylor
  rule 6) stays behind superko=True. DEVIATION from the Tromp-Taylor spec: area-mode results are not
  T-T results. Capture mode is unaffected (reasoned + empirically checked above). No-suicide variant
  (T-T itself allows suicide); also a deviation.
- Passing is always legal, in both modes (user decision).
- Komi for 7x7 area = 9. Source: KataGo opening-book authors' belief for area scoring, not proven;
  it was derived under Tromp-Taylor rules, not this engine's no-suicide/no-repetition variant
  (UNVERIFIED that 9 stays balanced here). 5x5 area: balanced komi is ~the whole board (sources say
  24 or 25, ambiguous), so 5x5 is run in capture mode only.
- Board size: 5x5 primary, 7x7 added later (user decision).
- Training on solver/MCTS-generated data before pure self-play.
- Framework PyTorch (user decision). Train BOTH label sources (my reading: two separate nets, same
  architecture/positions/hyperparameters, compared; not one mixed net). MCTS labels: 500 sims/position,
  policy = root visit fractions, value = 1 - 2*root.wins/root.visits. Hyperparameters (hidden 128,
  lr 1e-3, batch 64, 100 epochs) are untuned guesses, fixed up front; not to be tuned on the test set.

## OPEN decisions
None currently. (Superko, passing, komi, board size resolved 2026-11-09; see "Decisions made".)

## Claims made in earlier chat that are NOT yet verified
- That 5x5 Go (and possibly Atari Go on small boards) is solved / exact ground truth is available. (Probe above:
  not available for 5x5 with this engine; 3x3 solved; 4x4 unknown.)
- That small boards strongly favor Black (hence komi tuning).
- Exact Kaggle/Colab quotas and session limits (sources disagreed; check the live account).
- That public 9x9 game-record datasets and a fast Python MCTS (with numba) are adequate.

## Next steps
0. NEXT GAME DECIDED: Go (user, 2026-11-09). Tic-tac-toe multi-seed experiment DONE (see Verification status).
   Go plan and two open questions are in the chat (label source without an exact solver; whether to run a
   tic-tac-toe ablation first).
1. Tighten test_net.py thresholds from real numbers (user's runs).
2. Then Connect-4 or Go (network on a harder game), then hybrid PUCT.
3. Speed work: avoid copying the full superko history per node (e.g. Zobrist hashes).
