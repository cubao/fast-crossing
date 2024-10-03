from __future__ import annotations

import numpy as np

from ._core import KdTree as _KdTree


class KDTree:
    def __init__(self, data: np.ndarray, leafsize: int = 10, *args, **kwargs):
        data = np.asarray(data, dtype=np.float64)
        self.tree: _KdTree = _KdTree(data)
        self.tree.set_leafsize(leafsize)

    @staticmethod
    def vec3(arr: np.ndarray):
        return np.r_[arr, 0.0] if len(arr) == 2 else np.asarray(arr, dtype=np.float64)

    def count_neighbors(self, *args, **kwargs):
        raise NotImplementedError

    def query(self, x, k=1, *args, **kwargs):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            xyz = self.vec3(x)
            ii, dd = self.tree.nearest(xyz, k=k)
            return dd, ii
        if isinstance(k, (int, np.integer)):
            ret_ii, ret_dd = [], []
            for xyz in x:
                xyz = self.vec3(xyz)  # noqa: PLW2901
                if k == 1:
                    ii, dd = self.tree.nearest(xyz)
                else:
                    ii, dd = self.tree.nearest(xyz, k=k)
                    ii = ii.tolist()
                    dd = dd.tolist()
                ret_ii.append(ii)
                ret_dd.append(dd)
            return ret_dd, ret_ii
        K = max(k)
        ret_ii, ret_dd = [], []
        for xyz in x:
            xyz = self.vec3(xyz)  # noqa: PLW2901
            ii, dd = self.tree.nearest(xyz, k=K)
            ii = [ii[kk - 1] for kk in k]
            dd = [dd[kk - 1] for kk in k]
            ret_ii.append(ii)
            ret_dd.append(dd)
        return ret_dd, ret_ii

    def query_ball_point(
        self,
        x,
        r,
        p=2.0,
        eps=0,
        workers=1,
        return_sorted=None,
        return_length=False,
        *args,
        **kwargs,
    ):
        """
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query_ball_point.html#scipy.spatial.cKDTree.query_ball_point
        """
        x = np.asarray(x, dtype=np.float64)
        if return_sorted is None:
            # If None, does not sort single point queries, but does sort
            # multi-point queries which was the behavior before this option was
            # added.
            return_sorted = x.ndim != 1
        if x.ndim == 1:
            xyz = self.vec3(x)
            ii, _ = self.tree.nearest(
                xyz,
                radius=r,
                sort=return_sorted,
                return_squared_l2=True,
            )
            if return_length:
                return len(ii)
            return ii.tolist()
        if return_sorted is None:
            return_sorted = True
        if isinstance(r, (int, float, np.number)):
            r = [r] * len(x)
        ret_ii = []
        for pp, rr in zip(x, r):
            xyz = self.vec3(pp)
            ii, _ = self.tree.nearest(
                xyz,
                radius=rr,
                sort=return_sorted,
                return_squared_l2=True,
            )
            ret_ii.append(ii.tolist())
        if return_length:
            ret_ii = [len(ii) for ii in ret_ii]
        return ret_ii

    def query_ball_tree(self, *args, **kwargs):
        raise NotImplementedError

    def query_pairs(self, *args, **kwargs):
        raise NotImplementedError

    def query_distance_matrix(self, *args, **kwargs):
        raise NotImplementedError


# create alias
cKDTree = KDTree
