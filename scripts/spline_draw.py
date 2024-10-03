from __future__ import annotations

from vedo import Picture, dataurl
from vedo.applications import SplinePlotter  # ready to use class!

pic = Picture(dataurl + "images/embryo.jpg")

plt = SplinePlotter(pic)
plt.show(mode="image", zoom="tight")
print("Npts =", len(plt.cpoints), "NSpline =", plt.line.npoints)
