# based on https://raw.githubusercontent.com/marcomusy/vedo/master/examples/basic/spline_tool.py
import math
import random
from typing import List, Tuple

import numpy as np
from vedo import Circle, show  # noqa
from vedo.plotter import Event  # noqa
from vedo.pointcloud import Points  # noqa
from vedo.shapes import Line, Polygon  # noqa


# https://stackoverflow.com/questions/8997099/algorithm-to-generate-random-2d-polygon
def generate_polygon(
    *,
    center: Tuple[float, float] = (0.0, 0.0),
    avg_radius: float = 100.0,
    irregularity: float = 1.0,
    spikiness: float = 0.1,
    num_vertices: int = 100,
    close: bool = True,
) -> List[Tuple[float, float]]:
    if irregularity < 0 or irregularity > 1:
        raise ValueError("Irregularity must be between 0 and 1.")
    if spikiness < 0 or spikiness > 1:
        raise ValueError("Spikiness must be between 0 and 1.")
    if close:
        num_vertices -= 1
    irregularity *= 2 * math.pi / num_vertices
    spikiness *= avg_radius

    def random_angle_steps(steps: int, irregularity: float) -> List[float]:
        angles = []
        lower = (2 * math.pi / steps) - irregularity
        upper = (2 * math.pi / steps) + irregularity
        cumsum = 0
        for _ in range(steps):
            angle = random.uniform(lower, upper)
            angles.append(angle)
            cumsum += angle
        cumsum /= 2 * math.pi
        for i in range(steps):
            angles[i] /= cumsum
        return angles

    angle_steps = random_angle_steps(num_vertices, irregularity)
    points = []
    angle = random.uniform(0, 2 * math.pi)

    def clip(value, lower, upper):
        return min(upper, max(value, lower))

    for i in range(num_vertices):
        radius = clip(random.gauss(avg_radius, spikiness), 0, 2 * avg_radius)
        point = (
            center[0] + radius * math.cos(angle),
            center[1] + radius * math.sin(angle),
        )
        points.append(point)
        angle += angle_steps[i]
    if close:
        points.append(points[0])
    return np.array(points)


def on_key_press(evt):
    if evt.keypress == "c":
        print("==== Cleared all points ====", c="r", invert=True)


# def update(self):
#     self.remove([self.spline, self.points])  # remove old points and spline
#     self.points = Points(self.cpoints).ps(10).c("purple5")
#     self.points.pickable(False)  # avoid picking the same point
#     if len(self.cpoints) > 2:
#         self.spline = Spline(self.cpoints, closed=False).c("yellow5").lw(3)
#         self.add(self.points, self.spline)
#     else:
#         self.add(self.points)


radius = 100

# Create a set of points in space
pts = Circle(r=radius, res=8).extrude(zshift=0.5).ps(8)
print(type(pts))

pts2 = generate_polygon(avg_radius=radius, irregularity=1.0, spikiness=0.2)
# pts2 = Points(pts2)
pts2 = Line(pts2)
print(pts2.points().shape)

# Visualize the points
plt = show([pts, pts2], __doc__, interactive=False, axes=1)

# Add the spline tool using the same points and interact with it
sptool = plt.add_spline_tool(pts, closed=True)
# sptool = plt.add_spline_tool(pts2, closed=True)

# spline = sptool.spline()
# print(spline, type(spline))


def callback(evt: Event):
    if not evt.actor:
        return
    print("point coords =", evt.picked3d)
    sptool.spline()


plt.add_callback("mouse hovering", callback)

plt.interactive()
