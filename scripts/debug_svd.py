from typing import Tuple, Union

import numpy as np
from vedo import show  # noqa
from vedo.pointcloud import Points  # noqa
from vedo.shapes import Arrow, Cross3D, Line, Polygon  # noqa


def cartesian_product(*arrays):
    return np.stack(np.meshgrid(*arrays), axis=-1).reshape(-1, len(arrays))


xs = np.linspace(0, 400, 41)
ys = np.linspace(0, 300, 31)

xyzs = cartesian_product(xs, ys)
xyzs = np.column_stack((xyzs, np.zeros(len(xyzs))))
# print(xyzs.shape)

if False:
    theta = np.radians(30)
    xdir = np.array([np.cos(theta), np.sin(theta), 0.0])
    ydir = np.array([-xdir[1], xdir[0], 0.0])
    zdir = np.array([0.0, 0.0, 1.0])
else:
    # have some randomness
    theta = np.radians(np.random.random() * 360.0)
    phi = np.radians(np.random.random() * 90.0)
    xdir = np.array(
        [np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)]
    )
    ydir = np.array([-xdir[1], xdir[0], 0.0])
    ydir /= np.linalg.norm(ydir)
    zdir = np.cross(xdir, ydir)

R_world_local = np.c_[xdir, ydir, zdir]
xyzs = (R_world_local @ xyzs.T).T

center = xyzs.mean(axis=0)
u, sig, vT = np.linalg.svd(xyzs - center)
xdir2, ydir2, zdir2 = vT[0], vT[1], vT[2]
# assert:
#   xdir @ xdir2 == 1.0
#   ydir @ ydir2 == 1.0
#   zdir @ zdir2 == 1.0


def buildAxesHelper(
    pose: np.ndarray, *, scale: Union[float, Tuple[float, float, float]] = None
):
    """
    pose: 4x4 matrix, (pose@local_xyz.T).T -> global_xyz
        [ |  ,   | ,  |  ,  |     ]
        [xdir, ydir, zdir, center ]
        [ |  ,   | ,  |  ,  |     ]
        [ 0  ,   0 ,  0  ,  1     ]
    """
    xdir = pose[:3, 0]
    ydir = pose[:3, 1]
    zdir = pose[:3, 2]
    center = pose[:3, 3]
    if scale is None:
        sx, sy, sz = 1.0, 1.0, 1.0
    elif isinstance(scale, (list, tuple)):
        sx, sy = scale[:2]
        sz = scale[2] if len(scale) > 2 else 1.0
    else:
        sx, sy, sz = scale, scale, scale
    ax = Arrow(center, center + sx * xdir, c="r")
    ay = Arrow(center, center + sy * ydir, c="g")
    az = Arrow(center, center + sz * zdir, c="b")
    return ax + ay + az


axesHelper = buildAxesHelper(np.c_[xdir2, ydir2, zdir2, center], scale=100)

points = Points(xyzs)
show(points, axesHelper, axes=2)
