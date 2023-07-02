# point-in-polygon

[wikipedia:Point_in_polygon](https://en.wikipedia.org/wiki/Point_in_polygon):

>   In computational geometry, the point-in-polygon (PIP) problem asks whether
>   a given point in the plane lies inside, outside, or on the boundary of a
>   polygon.  It is a special case of point location problems and finds
>   applications in areas that deal with processing geometrical data, such as
>   computer graphics, computer vision, geographic information systems (GIS),
>   motion planning, and computer-aided design (CAD).

## super-fast implementation from matplotlib

[`shapely`](https://shapely.readthedocs.io) (>=2.0.1 already integrated C++ GEOS) version:

```python
import shapely
from shapely.geometry import Polygon
polygon = Polygon(polygon)
points = shapely.points(points)
mask = shapely.contains(polygon, points).astype(np.int32)
```

[`matplotlib`](https://matplotlib.org/stable/api/path_api.html) version:

```python
from matplotlib.path import Path
path = Path(polygon)
mask = path.contains_points(points).astype(np.int32)
```

matplotlib version is 50x faster than shapely version. Why???

>   Need to mention that, shapely version can handle multi-polygon (thought that's not a quite common need)

We integrated matplotlib version:

```python
from fast_crossing import point_in_polygon
mask = point_in_polygon(points=points, polygon=polygon)
```

It's a little bit faster than matplotlib. (Maybe because I used `-O3` compile flag?)

## benchmarks

code:

```python
{%
    include-markdown "../benchmarks/benchmark_point_in_polygon.py"
    comments=false
%}
```

test:

```makefile
{%
    include-markdown "../Makefile"
    start="benchmark_point_in_polygon:"
    end=".PHONY: benchmark_point_in_polygon"
    comments=false
%}
```

result:

```
python3 benchmarks/benchmark_point_in_polygon.py generate_test_data -o dist/point_in_polygon
2023-03-07 16:04:39.661 | INFO     | __main__:generate_test_data:206 - wrote to dist/point_in_polygon/random_num_10000__bbox_800.00x600.00__radius_250.00__points.npy
2023-03-07 16:04:39.661 | INFO     | __main__:generate_test_data:219 - wrote to dist/point_in_polygon/random_num_10000__bbox_800.00x600.00__radius_250.00__polygon.npy
python3 benchmarks/benchmark_point_in_polygon.py shapely \
        dist/point_in_polygon/random_num_10000__bbox_800.00x600.00__radius_250.00__points.npy \
        dist/point_in_polygon/random_num_10000__bbox_800.00x600.00__radius_250.00__polygon.npy \
        dist/mask_shapely.npy
2023-03-07 16:04:39.911 | INFO     | __main__:point_in_polygon_shapely:59 - point_in_polygon_shapely, secs:0.114556074 (3,207/10000)
2023-03-07 16:04:39.912 | INFO     | __main__:write_mask:78 - wrote to dist/mask_shapely.npy
python3 benchmarks/benchmark_point_in_polygon.py matplotlib \
        dist/point_in_polygon/random_num_10000__bbox_800.00x600.00__radius_250.00__points.npy \
        dist/point_in_polygon/random_num_10000__bbox_800.00x600.00__radius_250.00__polygon.npy \
        dist/mask_matplotlib.npy
2023-03-07 16:04:40.086 | INFO     | __main__:point_in_polygon_matplotlib:17 - point_in_polygon_matplotlib, secs:0.001869917 (3,207/10000)
2023-03-07 16:04:40.086 | INFO     | __main__:write_mask:78 - wrote to dist/mask_matplotlib.npy
python3 benchmarks/benchmark_point_in_polygon.py cubao \
        dist/point_in_polygon/random_num_10000__bbox_800.00x600.00__radius_250.00__points.npy \
        dist/point_in_polygon/random_num_10000__bbox_800.00x600.00__radius_250.00__polygon.npy \
        dist/mask_cubao.npy
2023-03-07 16:04:40.224 | INFO     | __main__:point_in_polygon_cubao:31 - point_in_polygon_cubao, secs:0.001556635 (3,207/10000)
2023-03-07 16:04:40.225 | INFO     | __main__:write_mask:78 - wrote to dist/mask_cubao.npy
```

| implementation | time (seconds) | speed up |
| :-- | :-- | :-- |
| shapely | 0.114556074 | |
| matplotlib | 0.001869917 | 61x |
| cubao | 0.001556635 | 73x |
