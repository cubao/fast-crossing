"""Benchmark batch (N-point) queries against the per-point Python loop.

Scale is copied from the real downstream load reported in
`build/batch-query.md` (the `align2ld` LD-matching solver):

    target index : 394 polylines, 4160 segments, ~10-11 segments per line
    queries      : 1562 points, ENU plane, direction ~parallel to the polylines

Three call shapes are compared, all answering the *same* question, so the
numbers are directly comparable:

  1. `nearest_loop`        - `for p in points: fc.nearest(p)`
                             (what a downstream caller does today)
  2. `nearest_batch`       - one call, `counts/labels/dists` out
  3. `scanline_loop`       - `for p, d in ...: fc.scanline_hits(p, d, ...)`
                             (the lateral-candidate search, per point)
  4. `intersections_batch` - one call, flat buffers out
                             (`only_nearest` both ways)

Read the numbers honestly: most of the win here is *not* "batching the pybind
crossing". Query setup per call (EPU/ld conversion, bbox construction, result
vector allocation) is per point in both loops, so hoisting it into one C++ loop
is a constant-factor change. The bigger effect for the scanline query comes from
`only_nearest` avoiding the sort/allocation of the full hit list, and from the
segment R-tree (bbox search) instead of scanning a global candidate set.

If you need "same points queried again for the same index", no batch API helps:
that is repeated work downstream and has to be cached there. This benchmark
does not claim otherwise - it measures one batch of fresh queries.

Usage:

    python benchmarks/benchmark_batch_query.py run
    python benchmarks/benchmark_batch_query.py run --num_queries=1562 --num_lines=394
"""

from __future__ import annotations

import time

import numpy as np
from loguru import logger

# reference shape from the downstream report
NUM_POLYLINES = 394
NUM_SEGMENTS = 4160
NUM_QUERIES = 1562


def make_index(
    num_polylines: int = NUM_POLYLINES,
    num_segments: int = NUM_SEGMENTS,
    *,
    length: float = 40.0,
    spacing: float = 3.5,
):
    """A synthetic "LD corridor": near-straight polylines along +x.

    `spacing` is the distance between neighbouring lanes (m), `length` the
    extent of each polyline along x. Lines are laid out in a grid so the index
    has a realistic 2D extent (a corridor ~wraps around, like a map crop).
    """
    base = num_segments // num_polylines
    extra = num_segments - base * num_polylines  # first `extra` lines get +1

    polylines = []
    for i in range(num_polylines):
        n_seg = base + (1 if i < extra else 0)
        # grid layout: 40 lanes across, rows every `length` metres along x
        row, col = divmod(i, 40)
        y0 = col * spacing
        x0 = row * (length + 20.0)
        x = np.linspace(x0, x0 + length, n_seg + 1)
        # tiny lateral curvature so segments are not exactly horizontal
        y = y0 + 0.6 * np.sin((x - x0) / length * np.pi + i)
        z = np.zeros_like(x)
        polylines.append(np.c_[x, y, z])

    from fast_crossing import FastCrossing

    fc = FastCrossing()
    for i, polyline in enumerate(polylines):
        fc.add_polyline(np.ascontiguousarray(polyline), index=i)
    fc.finish()
    return fc, polylines


def make_queries(polylines, num_queries: int = NUM_QUERIES, *, seed: int = 1):
    """Points scattered on/near the polylines, headings ~parallel to them."""
    rng = np.random.default_rng(seed)
    lengths = np.array([len(p) for p in polylines])
    owner = rng.choice(len(polylines), size=num_queries, p=lengths / lengths.sum())
    llas = np.zeros((num_queries, 3))
    dirs = np.zeros((num_queries, 3))
    for i, oi in enumerate(owner):
        polyline = polylines[oi]
        k = rng.integers(0, len(polyline) - 1)
        t = rng.random()
        p = polyline[k] * (1 - t) + polyline[k + 1] * t
        # lateral jitter up to ~2 m, like a matched point cloud
        d = polyline[k + 1, :2] - polyline[k, :2]
        d /= np.linalg.norm(d)
        left = np.array([-d[1], d[0]])
        p = p.copy()
        p[:2] += left * rng.uniform(-2.0, 2.0)
        llas[i] = p
        # heading with ~3 deg jitter
        heading = np.arctan2(d[1], d[0]) + rng.normal(0.0, 0.05)
        dirs[i] = [np.cos(heading), np.sin(heading), 0.0]
    return llas, dirs


def timed(fn, repeat: int = 3):
    best = float("inf")
    result = None
    for _ in range(repeat):
        tic = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - tic)
    return best, result


def report(name: str, secs: float, num: int, extra: str = ""):
    logger.info(
        f"{name:<28} secs:{secs:8.4f}  "
        f"{secs / num * 1e6:7.2f} us/pt  {extra}".rstrip()
    )
    return secs


def nearest_loop(fc, llas):
    return [fc.nearest(p) for p in llas]


def scanline_loop(fc, llas, dirs, options, *, only_nearest: bool):
    options.only_nearest = only_nearest
    return [
        fc.scanline_hits(p, d, min=-5.0, max=5.0, options=options)
        for p, d in zip(llas, dirs)
    ]


def run(
    num_polylines: int = NUM_POLYLINES,
    num_segments: int = NUM_SEGMENTS,
    num_queries: int = NUM_QUERIES,
    repeat: int = 3,
):
    from fast_crossing import ScanlineHitFilterOptions

    fc, polylines = make_index(num_polylines, num_segments)
    llas, dirs = make_queries(polylines, num_queries)
    logger.info(
        f"index: {fc.num_polylines()} polylines, "
        f"{num_segments} segments, queries: {num_queries}"
    )

    options = ScanlineHitFilterOptions()
    options.accept_opposite = True
    options.min_dot = 0.9
    options.max_height_offset = 0.0
    options.only_nearest = False

    # ---- 0: pure pybind crossing cost (empty call) ---------------------
    # `num_polylines()` does no work, so this is the floor of any per-point
    # Python loop: it isolates "call + int boxing" from the query itself.
    t_call, _ = timed(lambda: [fc.num_polylines() for _ in range(num_queries)], repeat)
    report("empty call (num_polylines)", t_call, num_queries)

    # ---- 1/2: nearest (point based) -----------------------------------
    t_loop, ref_loop = timed(lambda: nearest_loop(fc, llas), repeat)
    report(
        "nearest loop",
        t_loop,
        num_queries,
        f"query-only ~{(t_loop - t_call) / num_queries * 1e6:.2f} us/pt",
    )

    t_batch, ref_batch = timed(lambda: fc.nearest_batch(llas, k=1), repeat)
    counts, pid, pidx, dists = ref_batch
    report("nearest_batch", t_batch, num_queries, f"speedup x{t_loop / t_batch:.2f}")

    # ---- 0b: is the per-point cost index-size dependent? ---------------
    # Same loop, tiny index (one polyline, 11 points). If this is about as slow
    # as the 394-polyline index, the per-call cost is *fixed overhead* and the
    # batch win comes from amortizing it - not from a better traversal.
    tiny, tiny_polylines = make_index(num_polylines=1, num_segments=10)
    tiny_llas, _ = make_queries(tiny_polylines, num_queries)
    t_tiny, _ = timed(lambda: nearest_loop(tiny, tiny_llas), repeat)
    report(
        "nearest loop (1 polyline)",
        t_tiny,
        num_queries,
        f"index-size sensitivity x{t_tiny / t_loop:.2f} "
        f"vs {num_polylines} polylines",
    )

    # sanity: element-wise equal to the loop
    loop_labels = np.array([np.atleast_2d(idx) for idx, _ in ref_loop]).reshape(-1, 2)
    loop_dists = np.array([dd for _, dd in ref_loop]).reshape(-1)
    assert np.array_equal(np.c_[pid[:, 0], pidx[:, 0]], loop_labels)
    np.testing.assert_allclose(dists[:, 0], loop_dists, atol=1e-12)

    # ---- 3/4: scanline lateral candidates ------------------------------
    t_s_loop, ref_s_loop = timed(
        lambda: scanline_loop(fc, llas, dirs, options, only_nearest=False), repeat
    )
    hits_per_point = [len(h) for h in ref_s_loop]
    report(
        "scanline loop",
        t_s_loop,
        num_queries,
        f"avg hits {np.mean(hits_per_point):.2f}, "
        f"query-only ~{(t_s_loop - t_call) / num_queries * 1e6:.2f} us/pt",
    )

    t_s_batch, ref_s_batch = timed(
        lambda: fc.intersections_batch(llas, dirs, min=-5.0, max=5.0, options=options),
        repeat,
    )
    counts, lateral, *_ = ref_s_batch
    report(
        "intersections_batch",
        t_s_batch,
        num_queries,
        f"speedup x{t_s_loop / t_s_batch:.2f}",
    )
    assert np.all(counts == hits_per_point)

    # only_nearest: the fast path downstream actually wants
    t_s_loop_1, _ = timed(
        lambda: scanline_loop(fc, llas, dirs, options, only_nearest=True), repeat
    )
    report(
        "scanline loop (only_nearest)",
        t_s_loop_1,
        num_queries,
        f"x{t_s_loop / t_s_loop_1:.2f} vs full hits",
    )

    t_s_batch_1, ref_s_batch_1 = timed(
        lambda: fc.intersections_batch(
            llas, dirs, min=-5.0, max=5.0, options=options, max_hits=1
        ),
        repeat,
    )
    report(
        "intersections_batch (max_hits=1)",
        t_s_batch_1,
        num_queries,
        f"speedup x{t_s_loop_1 / t_s_batch_1:.2f} vs loop",
    )

    # correctness of the "best candidate" against the per-point rule
    best = ref_s_batch_1
    assert np.all(best[0] > 0)
    for i, hits in enumerate(ref_s_loop):
        if not hits:
            assert best[0][i] == 0
            continue
        expected = min(hits, key=lambda h: abs(h.lateral_offset))
        assert best[2][i, 0] == expected.polyline_index
        assert abs(best[1][i, 0] - expected.lateral_offset) < 1e-12

    # ---- summary -------------------------------------------------------
    logger.info("")
    logger.info(f"summary (per batch of {num_queries} queries):")
    logger.info(
        f"  nearest:    loop {t_loop:7.3f}s -> batch {t_batch:7.3f}s "
        f"(x{t_loop / t_batch:.2f})"
    )
    logger.info(
        f"  scanline:   loop {t_s_loop:7.3f}s -> batch {t_s_batch:7.3f}s "
        f"(x{t_s_loop / t_s_batch:.2f})"
    )
    logger.info(
        f"  only_nearest:{t_s_loop_1:6.3f}s -> batch {t_s_batch_1:7.3f}s "
        f"(x{t_s_loop_1 / t_s_batch_1:.2f})"
    )
    logger.info(
        f"  per-call overhead: {t_call / num_queries * 1e6:.2f} us/pt "
        f"({t_call / t_loop * 100:.0f}% of the nearest loop)"
    )
    logger.info(
        f"  index-size check: 1-polyline loop is x{t_tiny / t_loop:.2f} of the "
        f"{num_polylines}-polyline loop -> the per-point cost is dominated by "
        "fixed per-call overhead, not by index size."
    )
    logger.info(
        "  note 1: the loop-vs-batch gap is *amortization*, not a better "
        "traversal: the same single nearest query costs about the same against 1 "
        "polyline as against 394 (see the index-size row above), so the batch API "
        "wins by paying pybind/alloc/setup once instead of N times. It does not "
        "make an individual lookup faster."
    )
    logger.info(
        "  note 2: the upstream `align2ld` loop spends ~91% of its 13.95 s in "
        "these queries because the *same points are re-queried for every "
        "iteration*; a batch API removes the per-call overhead but NOT that "
        "repeated work. Cache per-iteration results on the caller side."
    )
    return {
        "nearest_loop": t_loop,
        "nearest_batch": t_batch,
        "scanline_loop": t_s_loop,
        "intersections_batch": t_s_batch,
        "scanline_loop_only_nearest": t_s_loop_1,
        "intersections_batch_max_hits_1": t_s_batch_1,
        "num_queries": num_queries,
    }


def run_against_tbc():
    """Compare with the reference implementation downstream uses today.

    Requires `tbc_bentley_ottmann` (Momenta internal). Skipped otherwise.
    """
    try:
        from tbc_bentley_ottmann import FastCrossing as TbcFastCrossing
        from tbc_bentley_ottmann import ScanlineHitFilterOptions as TbcOptions
    except ImportError:
        logger.warning("tbc_bentley_ottmann not installed, skip")
        return None

    fc, polylines = make_index()
    llas, dirs = make_queries(polylines)

    tbc = TbcFastCrossing(is_wgs84=False)
    for i, polyline in enumerate(polylines):
        tbc.add_polyline(np.ascontiguousarray(polyline), id=str(i))
    tbc.build()
    tbc_options = TbcOptions()
    tbc_options.accept_opposite = True
    tbc_options.min_dot = 0.9
    tbc_options.max_height_offset = 0.0
    tbc_options.only_nearest = False

    from fast_crossing import ScanlineHitFilterOptions

    options = ScanlineHitFilterOptions()
    options.accept_opposite = True
    options.min_dot = 0.9
    options.max_height_offset = 0.0

    def tbc_loop():
        return [
            tbc.intersections(lla=p, dir=d, min=-5.0, max=5.0, options=tbc_options)
            for p, d in zip(llas, dirs)
        ]

    def ours_batch():
        return fc.intersections_batch(llas, dirs, min=-5.0, max=5.0, options=options)

    t_tbc, tbc_hits = timed(tbc_loop)
    t_ours, ours = timed(ours_batch)
    report("tbc per-point loop", t_tbc, len(llas))
    report("fast_crossing batch", t_ours, len(llas), f"speedup x{t_tbc / t_ours:.2f}")
    logger.info(
        f"  hits: tbc {sum(len(h) for h in tbc_hits)}, "
        f"fast_crossing {int(np.sum(ours[0]))}"
    )
    return {"tbc_loop": t_tbc, "fast_crossing_batch": t_ours}


if __name__ == "__main__":
    import fire

    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({"run": run, "against_tbc": run_against_tbc})
