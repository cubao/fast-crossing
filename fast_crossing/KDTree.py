import numpy as np
from _pybind11_fast_crossing import KdTree as cKDTree


class KDTree:
    def __init__(self, data: np.ndarray, leafsize: int = 10, *args, **kwargs):
        data = np.asarray(data, dtype=np.float64)
        self.tree: cKDTree = cKDTree(data)
        self.tree.set_leafsize(leafsize)
    
    @staticmethod
    def vec3(arr: np.ndarray):
        return np.r_[arr, 0.0] if len(arr) == 2 else np.asarray(arr, dtype=np.float64)

    def count_neighbors(self, *args, **kwargs):
        raise NotImplementedError

    def query(self, x, k=1, *args, **kwargs):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = self.vec3(x)
            ii, dd = self.tree.nearest(x, k=k)
            return dd, ii
        for row in x:
            pass

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
        if x.ndim == 1:
            if return_sorted is None:
                return_sorted = False
            ii, dd = self.vec3(x, radius=r)
            return dd, ii
            self.tree.nearest(x)
        if return_sorted is None:
            return_sorted = True
        raise NotImplementedError

    def query_ball_tree(self, *args, **kwargs):
        raise NotImplementedError

    def query_pairs(self, *args, **kwargs):
        raise NotImplementedError

    def query_distance_matrix(self, *args, **kwargs):
        raise NotImplementedError
