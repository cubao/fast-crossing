# Release Notes

---

## Upgrading

To upgrade `fast-crossing` to the latest version, use pip:

```bash
pip install -U fast-crossing
```

## Version 0.1.7 (2026-10-28)

*   Add batch query APIs, so N points can be queried against one index in a
    single C++ loop:
    *   `FastCrossing.nearest_batch` - `(counts, polyline_index, point_index, dists)`,
        `k` or `radius` mode, optional `max_hits`.
    *   `FastCrossing.intersections_batch` - perpendicular scanline / lateral
        candidate search for a batch of `(point, direction)` pairs, returns flat
        preallocated buffers (`counts/lateral/polyline_id/seg_idx/t/dot/lla/dir`).
    *   `FastCrossing.scanline_hits` - single-point version of the above (same
        filters, same ordering).
    *   `ScanlineHitFilterOptions` / `ScanlineHit` - `max_height_offset`,
        `min_dot` + `accept_opposite` (same/opposite accepted, perpendicular
        rejected), `filter_out_hits_by_gap`, `only_nearest`.
*   Batch semantics are fixed and tested: miss = `-1` / `inf` / `NaN`,
    `max_hits` truncation keeps the smallest `|lateral|` first, `counts` is
    always the pre-truncation count.
*   Add `benchmarks/benchmark_batch_query.py` (394 polylines / 4160 segments /
    1562 queries), reporting per-call overhead separately from query cost.

## Version 0.1.6 (2026-10-03)

*   Build a pyodide (emscripten/wasm32) wheel so `import fast_crossing` works
    inside a browser, and publish it to PyPI together with the native wheels.
*   Add `tests/pyodide/index.html`, which loads pyodide, installs the wheel and
    runs `tests/test_basic.py` in the browser.

## Version 0.1.1 (2025-06-27)

*   Use sckit-build-core, ditch python 3.7

## Version 0.1.0 (2024-10-03)

*   Add typing stubs

## Version 0.0.9 (2024-09-07)

*   Update pybind11 (for python 3.11, 3.12), ditch python 3.6, 3.7

## Version 0.0.8 (2023-06-23)

*   Sync with headers

## Version 0.0.7 (2023-03-18)

*   Add polyline-in-polygon test, [youtube](https://www.youtube.com/watch?v=1dPJ3P84FxE), [bilibili](https://www.bilibili.com/video/BV1D24y1u7uB)

## Version 0.0.6 (2023-03-12)

*   Integrate KdQuiver (kdtree + dirs)

## Version 0.0.5 (2023-03-08)

*   Add nanaflann KdTree, use identical interface to `scipy.spatial.cKDTree`

## Version 0.0.4 (2023-03-07)

*   Integrate `point_in_polygon` test
*   Filter intersections by z, by self_intersection

## Version 0.0.3 (2023-03-05)

*   Fix CI
*   Use GitHub workflow to release to pypi

## Version 0.0.2 (2023-01-22)

*

## Version 0.0.1 (2023-01-22)

*   First release

---

You can also checkout releases on:

-   GitHub: <https://github.com/cubao/polyline-ruler/releases>
-   PyPi: <https://pypi.org/project/polyline-ruler>
