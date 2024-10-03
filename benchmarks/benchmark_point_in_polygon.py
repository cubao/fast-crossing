from __future__ import annotations

import math
import os
import random
import time

import numpy as np
from loguru import logger


def point_in_polygon_matplotlib(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    from matplotlib.path import Path

    tic = time.time()
    path = Path(polygon)
    mask = path.contains_points(points).astype(np.int32)
    logger.info(
        f"point_in_polygon_matplotlib, secs:{time.time() - tic:.9f} ({mask.sum():,}/{len(points)})"
    )
    return mask


def point_in_polygon_cubao(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """
    it's migrated from matplotlib
    """
    from fast_crossing import point_in_polygon

    tic = time.time()
    mask = point_in_polygon(points=points, polygon=polygon)
    logger.info(
        f"point_in_polygon_cubao, secs:{time.time() - tic:.9f} ({mask.sum():,}/{len(points)})"
    )
    return mask


def point_in_polygon_polygons(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """
    not ready
    """
    import polygons

    polygon = polygon.tolist()
    num_edges_children = 4
    num_nodes_children = 4
    tree = polygons.build_search_tree(polygon, num_edges_children, num_nodes_children)
    return polygons.points_are_inside(tree, points).astype(np.int32)


def point_in_polygon_shapely(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    import shapely
    from shapely.geometry import Polygon

    tic = time.time()
    polygon = Polygon(polygon)
    points = shapely.points(points)
    mask = shapely.contains(polygon, points).astype(np.int32)
    logger.info(
        f"point_in_polygon_shapely, secs:{time.time() - tic:.9f} ({mask.sum():,}/{len(points)})"
    )
    return mask


def load_points(path: str):
    return np.load(path)


def load_polygon(path: str):
    if path.endswith((".npy", ".pcd")):
        load_points(path)


def write_mask(mask: np.ndarray, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.save(path, mask)
    logger.info(f"wrote to {path}")


def wrapping(fn):
    def wrapped_fn(input_points: str, input_polygon: str, output_path: str):
        points = load_points(input_points)[:, :2]
        polygon = load_polygon(input_polygon)[:, :2]
        mask = fn(points, polygon)
        write_mask(mask, output_path)

    return wrapped_fn


# https://stackoverflow.com/questions/8997099/algorithm-to-generate-random-2d-polygon
def generate_polygon(
    center: tuple[float, float],
    avg_radius: float,
    irregularity: float,
    spikiness: float,
    num_vertices: int,
) -> list[tuple[float, float]]:
    """
    Start with the center of the polygon at center, then creates the
    polygon by sampling points on a circle around the center.
    Random noise is added by varying the angular spacing between
    sequential points, and by varying the radial distance of each
    point from the centre.

    Args:
        center (Tuple[float, float]):
            a pair representing the center of the circumference used
            to generate the polygon.
        avg_radius (float):
            the average radius (distance of each generated vertex to
            the center of the circumference) used to generate points
            with a normal distribution.
        irregularity (float):
            variance of the spacing of the angles between consecutive
            vertices.
        spikiness (float):
            variance of the distance of each vertex to the center of
            the circumference.
        num_vertices (int):
            the number of vertices of the polygon.
    Returns:
        List[Tuple[float, float]]: list of vertices, in CCW order.
    """
    # Parameter check
    if irregularity < 0 or irregularity > 1:
        raise ValueError("Irregularity must be between 0 and 1.")
    if spikiness < 0 or spikiness > 1:
        raise ValueError("Spikiness must be between 0 and 1.")

    irregularity *= 2 * math.pi / num_vertices
    spikiness *= avg_radius
    angle_steps = random_angle_steps(num_vertices, irregularity)

    # now generate the points
    points = []
    angle = random.uniform(0, 2 * math.pi)
    for i in range(num_vertices):
        radius = clip(random.gauss(avg_radius, spikiness), 0, 2 * avg_radius)
        point = (
            center[0] + radius * math.cos(angle),
            center[1] + radius * math.sin(angle),
        )
        points.append(point)
        angle += angle_steps[i]

    return points


def random_angle_steps(steps: int, irregularity: float) -> list[float]:
    """Generates the division of a circumference in random angles.

    Args:
        steps (int):
            the number of angles to generate.
        irregularity (float):
            variance of the spacing of the angles between consecutive vertices.
    Returns:
        List[float]: the list of the random angles.
    """
    # generate n angle steps
    angles = []
    lower = (2 * math.pi / steps) - irregularity
    upper = (2 * math.pi / steps) + irregularity
    cumsum = 0
    for _ in range(steps):
        angle = random.uniform(lower, upper)
        angles.append(angle)
        cumsum += angle

    # normalize the steps so that point 0 and point n+1 are the same
    cumsum /= 2 * math.pi
    for i in range(steps):
        angles[i] /= cumsum
    return angles


def clip(value, lower, upper):
    """
    Given an interval, values outside the interval are clipped to the interval
    edges.
    """
    return min(upper, max(value, lower))


def generate_test_data(
    output_dir: str = ".",
    *,
    label: str = None,
    num: int = 10000,
    width: float = 800,
    height: float = 600,
    radius: float = 250,
):
    os.makedirs(os.path.abspath(output_dir), exist_ok=True)

    if not label:
        label = f"random_num_{num}__bbox_{width:.2f}x{height:.2f}__radius_{radius:.2f}"
    xs = (np.random.random(num) - 0.5) * width
    ys = (np.random.random(num) - 0.5) * height
    xyzs = np.zeros((num, 2))
    xyzs[:, 0] = xs
    xyzs[:, 1] = ys
    path = f"{output_dir}/{label}__points.npy"
    np.save(path, xyzs)
    logger.info(f"wrote to {path}")

    polygon = np.array(
        generate_polygon(
            [0.0, 0.0],
            avg_radius=radius,
            irregularity=1.0,
            spikiness=0.6,
            num_vertices=60,
        )
    )
    path = f"{output_dir}/{label}__polygon.npy"
    np.save(path, polygon)
    logger.info(f"wrote to {path}")


def point_in_polygon_all():
    pass


if __name__ == "__main__":
    import fire

    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire(
        {
            "all": point_in_polygon_all,
            "matplotlib": wrapping(point_in_polygon_matplotlib),
            "cubao": wrapping(point_in_polygon_cubao),
            "polygons": wrapping(point_in_polygon_polygons),
            "shapely": wrapping(point_in_polygon_shapely),
            "generate_test_data": generate_test_data,
        }
    )
