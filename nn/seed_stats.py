"""Torch-free bookkeeping for the multi-seed tic-tac-toe experiment: per-ply accuracy, a bootstrap CI for
a difference of means, and the printed summary. A "run" is one trained net, as a dict (see seeds_ttt.py)."""
import random
import statistics


def per_ply(positions, hits):
    """positions: [(board, to_play)], hits: [bool]. -> {stones_on_board: (n, hit_rate)}."""
    acc = {}
    for (b, _), h in zip(positions, hits):
        ply = sum(1 for v in b if v)
        n, k = acc.get(ply, (0, 0))
        acc[ply] = (n + 1, k + bool(h))
    return {p: (n, k / n) for p, (n, k) in sorted(acc.items())}


def bootstrap_diff_ci(a, b, n_boot=10000, seed=0, alpha=0.05):
    """(mean(b) - mean(a), lo, hi): percentile bootstrap CI for the difference of means. With a handful of
    seeds this is only approximate."""
    rng = random.Random(seed)
    diffs = sorted(sum(rng.choices(b, k=len(b))) / len(b) - sum(rng.choices(a, k=len(a))) / len(a)
                   for _ in range(n_boot))
    return (statistics.fmean(b) - statistics.fmean(a),
            diffs[int(alpha / 2 * n_boot)], diffs[int((1 - alpha / 2) * n_boot) - 1])


def _stats(xs):
    return {"mean": statistics.fmean(xs), "sd": statistics.stdev(xs) if len(xs) > 1 else 0.0,
            "min": min(xs), "max": max(xs)}


METRICS = (("p_loss_white", "P(loss to perfect) as white"), ("p_loss_black", "P(loss to perfect) as black"),
           ("blunder_positions", "blunder positions (both colors)"), ("rnd_draw", "draw rate vs Random (avg colors)"),
           ("rnd_loss", "loss rate vs Random (avg colors)"), ("test_opt", "held-out optimal-move rate"),
           ("test_vclass", "held-out value-class accuracy"))


def summarize_runs(runs, out=print, n_boot=10000):
    by = {}
    for r in runs:
        by.setdefault(r["source"], []).append(r)
    for rs in by.values():
        rs.sort(key=lambda r: r["seed"])
    res = {"stats": {}, "diff_ci": {}}
    for src, rs in by.items():
        out(f"\n--- per-seed results, net trained on {src} labels ---")
        out("seed | P(loss) white | P(loss) black | blunder pos | draw vs Rnd | loss vs Rnd | test opt-move | test value-cls")
        for r in rs:
            out(f"{r['seed']:4d} | {r['p_loss_white']:13.4f} | {r['p_loss_black']:13.4f} | {r['blunder_positions']:11d} | "
                f"{r['rnd_draw']:11.4f} | {r['rnd_loss']:11.4f} | {r['test_opt']:13.3f} | {r['test_vclass']:14.3f}")
    out("\n--- across seeds (mean, sd, min, max) ---")
    for src, rs in by.items():
        n = len(rs)
        zero = sum(1 for r in rs if r["p_loss_white"] == 0 and r["p_loss_black"] == 0)
        out(f"[{src}] n = {n} seeds; {zero} of {n} never lose to a perfect player")
        res["stats"][src] = {}
        for key, label in METRICS:
            s = _stats([r[key] for r in rs])
            res["stats"][src][key] = s
            out(f"    {label:36s} {s['mean']:.4f}  {s['sd']:.4f}  {s['min']:.4f}  {s['max']:.4f}")
    if "solver" in by and "mcts" in by:
        out("\n--- mcts minus solver, difference of means with percentile-bootstrap 95% CI (approximate, few seeds) ---")
        for key, label in METRICS[:1] + METRICS[3:5]:
            d = bootstrap_diff_ci([r[key] for r in by["solver"]], [r[key] for r in by["mcts"]], n_boot)
            res["diff_ci"][key] = d
            out(f"    {label:36s} {d[0]:+.4f}  [{d[1]:+.4f}, {d[2]:+.4f}]")
    for split in ("train", "test"):
        out(f"\n--- optimal-move rate by number of stones on the board ({split} positions; mean over seeds) ---")
        plies = sorted({p for rs in by.values() for r in rs for p in r[f"ply_{split}"]})
        out("stones |   n | " + " | ".join(f"{s:>6s}" for s in by))
        for p in plies:
            n = max((r[f"ply_{split}"][p][0] for rs in by.values() for r in rs if p in r[f"ply_{split}"]), default=0)
            cells = []
            for rs in by.values():
                v = [r[f"ply_{split}"][p][1] for r in rs if p in r[f"ply_{split}"]]
                cells.append(f"{statistics.fmean(v):6.3f}" if v else "     -")
            out(f"{p:6d} | {n:3d} | " + " | ".join(cells))
    return res
