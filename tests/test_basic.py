import os
import sys

import numpy as np
import pytest

from fast_crossing import (
    Arrow,
    FastCrossing,
    KdTree,
    Quiver,
    densify_polyline,
    point_in_polygon,
)


def test_fast_crossing():
    fc = FastCrossing()

    # add your polylines
    """
                    2 C
                    |
                    1 D
    0               |                  5
    A---------------o------------------B
                    |
                    |
                    -2 E
    """
    fc.add_polyline(np.array([[0.0, 0.0], [5.0, 0.0]]))  # AB
    fc.add_polyline(np.array([[2.5, 2.0], [2.5, 1.0], [2.5, -2.0]]))  # CDE

    # build index
    fc.finish()

    # num_poylines
    assert 2 == fc.num_poylines()
    rulers = fc.polyline_rulers()
    assert len(rulers) == 2
    ruler0 = fc.polyline_ruler(0)
    ruler1 = fc.polyline_ruler(1)
    assert not ruler0.is_wgs84()
    assert not ruler1.is_wgs84()
    assert ruler0.length() == 5
    assert ruler1.length() == 4
    assert fc.polyline_ruler(10) is None

    # intersections
    ret = fc.intersections([1.5, 0], [3.5, 2])
    assert len(ret) == 2
    assert np.linalg.norm(fc.coordinates(ret[0]) - [1.5, 0, 0]) < 1e-15
    assert np.linalg.norm(fc.coordinates(ret[1]) - [2.5, 1, 0]) < 1e-15
    xyz = fc.coordinates(0, 0, 0.2)
    assert np.linalg.norm(xyz - [1.0, 0, 0]) < 1e-15
    with pytest.raises(IndexError) as excinfo:
        xyz = fc.coordinates(2, 0, 0.5)
    assert "map::at" in str(excinfo)

    # query all line segment intersections
    # [
    #    (array([2.5, 0. ]),
    #     array([0.5       , 0.33333333]),
    #     array([0, 0], dtype=int32),
    #     array([1, 1], dtype=int32))
    # ]
    ret = fc.intersections()
    # print(ret)
    assert len(ret) == 1
    for xy, ts, label1, label2 in ret:
        # xy 是交点，即图中的 o 点坐标
        # t,s 是分位点，(0.5, 0.33)
        #        0.5  ->   o 在 AB 1/2 处
        #        0.33 ->   o 在 DE 1/3 处
        # label1 是 line segment 索引，(polyline_index, point_index)
        #        e.g. (0, 0)，polyline AB 的第一段
        # label2 是另一个条 line seg 的索引
        #        e.g. (1, 1)，polyline CDE 的第二段（DE 段）
        # print(xy)
        # print(ts)
        # print(label1)
        # print(label2)
        assert np.all(xy == [2.5, 0])
        assert np.all(ts == [0.5, 1 / 3.0])
        assert np.all(label1 == [0, 0])
        assert np.all(label2 == [1, 1])

    # query intersections against provided polyline
    polyline = np.array([[-6.0, -1.0], [-5.0, 1.0], [5.0, -1.0]])
    ret = fc.intersections(polyline)
    ret = np.array(ret)  # 还是转化成 numpy 比较好用
    xy = ret[:, 0]  # 直接取出所有交点
    ts = ret[:, 1]  # 所有分位点
    label1 = ret[:, 2]  # 所有 label1（当前 polyline 的 label）
    label2 = ret[:, 3]  # tree 中 line segs 的 label
    # print(ret, xy, ts, label1, label2)
    assert np.all(xy[0] == [0, 0])
    assert np.all(xy[1] == [2.5, -0.5])
    assert np.all(ts[0] == [0.5, 0])
    assert np.all(ts[1] == [0.75, 0.5])
    assert np.all(label1 == [[0, 1], [0, 1]])
    assert np.all(label2 == [[0, 0], [1, 1]])

    polyline2 = np.column_stack((polyline, np.zeros(len(polyline))))
    ret2 = np.array(fc.intersections(polyline2[:, :2]))
    assert str(ret) == str(ret2)


def test_fast_crossing_intersection3d():
    fc = FastCrossing()
    """
                    2 C
                    |
                    1 D
    0               |                  5
    A---------------o------------------B
                    |
                    |
                    -2 E
    """
    fc.add_polyline(np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 100]]))  # AB
    fc.add_polyline(np.array([[2.5, 2.0, 0.0], [2.5, 1.0, 100], [2.5, -2.0, 0]]))  # CDE
    fc.finish()
    ret = fc.intersections()
    assert len(ret) == 1
    ret = ret[0]
    xyz1 = fc.coordinates(ret, second=False)
    xyz2 = fc.coordinates(ret)
    assert np.linalg.norm(xyz1 - [2.5, 0, 50]) < 1e-10
    assert np.linalg.norm(xyz2 - [2.5, 0, 2 / 3 * 100.0]) < 1e-10


def test_fast_crossing_auto_rebuild_flatbush():
    fc = FastCrossing()
    fc.add_polyline(np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 100]]))  # AB
    fc.add_polyline(np.array([[2.5, 2.0, 0.0], [2.5, 1.0, 100], [2.5, -2.0, 0]]))  # CDE
    ret = fc.intersections()
    assert len(ret) == 1

    fc.add_polyline([[1.5, 0], [3.5, 2]])
    ret = fc.intersections()
    assert len(ret) == 4  # should dedup to 3?


def test_fast_crossing_filter_by_z():
    fc = FastCrossing()
    fc.add_polyline([[0, 0, 0], [1, 0, 0]])
    fc.add_polyline([[0, 0, 10], [1, 0, 10]])
    fc.add_polyline([[0, 0, 20], [1, 0, 20]])
    ret = fc.intersections([[0.5, -1], [0.5, 1]])
    assert len(ret) == 3

    ret = fc.intersections([[0.5, -1], [0.5, 1]], z_min=-1, z_max=1)
    assert len(ret) == 1
    assert fc.coordinates(ret[0])[2] == 0

    ret = fc.intersections([[0.5, -1, 10.5], [0.5, 1, 10.5]], z_min=-1, z_max=1)
    assert len(ret) == 1
    assert np.all(fc.coordinates(ret[0]) == [0.5, 0, 10])
    assert np.all(fc.coordinates(ret[0], second=True) == [0.5, 0, 10])
    # don't do this, first point is not in fc!!!
    # assert np.all(fc.coordinates(ret[0], second=False) == [0.5, 0, 10.5])

    ret = fc.intersections([[0.5, -1, 20], [0.5, 1, 20]], z_min=-1, z_max=1)
    assert len(ret) == 1
    assert fc.coordinates(ret[0])[2] == 20

    ret = fc.intersections([[0.5, -1, 15], [0.5, 1, 15]], z_min=-6, z_max=6)
    assert len(ret) == 2
    assert fc.coordinates(ret[0])[2] == 10
    assert fc.coordinates(ret[1])[2] == 20

    assert 3 == len(fc.intersections())  # 3 == C(3,2): choose 2 from 3
    # for row in fc.intersections():
    #     print(row)


def test_fast_crossing_dedup():
    # should be stable
    for _ in range(100):
        fc = FastCrossing()
        fc.add_polyline([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        fc.add_polyline([[0, 1, 0], [1, 1, 0], [2, 1, 0]])

        ret = fc.intersections([[1, -1], [1, 1]])
        assert len(ret) == 2
        assert np.all(ret[0][-1] == [0, 0]), ret
        assert np.all(ret[1][-1] == [1, 0]), ret
        assert ret[0][1][1] == 1.0, ret
        assert ret[1][1][1] == 1.0, ret

        ret = fc.intersections([[1, -1], [1, 1]], dedup=False)
        # for idx, row in enumerate(ret):
        #     print(idx, row)
        assert len(ret) == 4


def test_fast_crossing_single_polyline():
    fc = FastCrossing()
    fc.add_polyline([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    assert not fc.intersections()


def test_fast_crossing_single_polyline_self_intersection():
    fc = FastCrossing()
    fc.add_polyline([[0, 0, 0], [1, 0, 0], [1, 1, 0], [-1, -1, 0]])
    assert len(fc.intersections()) == 1
    assert not fc.intersections(z_offset_range=[0.0, 10.0], self_intersection=0)
    assert fc.intersections(z_offset_range=[0.0, 0.0])
    assert not fc.intersections(z_offset_range=[1e-15, 1.0])

    fc = FastCrossing()
    fc.add_polyline([[0, 0, 0], [1, 0, 0], [1, 1, 0], [-1, -1, 0]])
    fc.add_polyline(np.array([[0.0, 0.0], [5.0, 0.0]]))  # AB
    fc.add_polyline(np.array([[2.5, 2.0], [2.5, 1.0], [2.5, -2.0]]))  # CDE
    self_inter = fc.intersections(z_offset_range=[-1.0, 1e10], self_intersection=1)
    assert len(self_inter) == 1


def test_densify():
    coords = np.array(
        [
            [0, 0, 0],
            [2, 0, 0],
            [5, 0, 0],
        ],
        dtype=np.float64,
    )
    dense = densify_polyline(coords, max_gap=1.0)
    assert len(dense) == 6
    dense = densify_polyline(coords, max_gap=2.0)
    assert len(dense) == 4
    dense = densify_polyline(coords, max_gap=3.0)
    assert len(dense) == 3
    assert 3 == len(densify_polyline(coords, max_gap=3.0 + 1e-3))
    assert 4 == len(densify_polyline(coords, max_gap=3.0 - 1e-3))


def test_point_in_polygon():
    polygon = [[0, 0], [1, 0], [1, 1], [0, 1]]
    points = [[0.5, 0.5], [10, 0]]
    mask = point_in_polygon(points=points, polygon=polygon)
    assert np.all(mask == [1, 0])


def test_kdtree():
    xs = np.linspace(0, 10, 101)
    xyzs = np.zeros((len(xs), 3))
    xyzs[:, 0] = xs
    tree = KdTree(xyzs)
    assert tree.points().shape == (101, 3)
    idx, dist = tree.nearest(0)
    assert idx == 1, dist == 0.1
    idx, dist = tree.nearest([0.0, 0.0, 0.0])
    assert idx == 0, dist == 0.0
    idx, dist = tree.nearest([0.0, 0.0, 0.0], k=4)
    assert np.all(idx == [0, 1, 2, 3])
    np.testing.assert_allclose(dist, [0.0, 0.1, 0.2, 0.3], atol=1e-15)
    idx, dist = tree.nearest([0.0, 0.0, 0.0], radius=0.25)
    assert np.all(idx == [0, 1, 2])
    np.testing.assert_allclose(dist, [0.0, 0.1, 0.2], atol=1e-15)


def _test_cKDTree_query(KDTree):
    x, y = np.mgrid[0:5, 2:8]
    tree = KDTree(np.c_[x.ravel(), y.ravel()])

    expected_k1 = [[2.0, 0.2236068], [0, 13]]
    expected_k2 = [[[2.0, 2.23606798], [0.2236068, 0.80622577]], [[0, 6], [13, 19]]]
    for k, expected in zip([1, 2], [expected_k1, expected_k2]):  # noqa
        dd, ii = tree.query([[0, 0], [2.2, 2.9]], k=k)
        DD, II = expected
        np.testing.assert_allclose(dd, DD, atol=1e-6)
        np.testing.assert_allclose(ii, II, atol=1e-6)

    expected_k1 = [
        [[2.0], [0.22360679774997916]],
        [[0], [13]],
    ]
    expected_k2 = [
        [[2.23606797749979], [0.8062257748298548]],
        [[6], [19]],
    ]
    expected_k1_k2 = [
        [[2.0, 2.23606797749979], [0.22360679774997916, 0.8062257748298548]],
        [[0, 6], [13, 19]],
    ]
    for k, expected in zip(  # noqa
        [[1], [2], [1, 2]], [expected_k1, expected_k2, expected_k1_k2]
    ):
        dd, ii = tree.query([[0, 0], [2.2, 2.9]], k=k)
        DD, II = expected
        np.testing.assert_allclose(dd, DD, atol=1e-6)
        np.testing.assert_allclose(ii, II, atol=1e-6)

    x, y = np.mgrid[0:4, 0:4]
    points = np.c_[x.ravel(), y.ravel()]
    tree = KDTree(points)
    ret = sorted(tree.query_ball_point([2 + 1e-15, 1e-15], 1 + 1e-9))
    assert np.all([4, 8, 9, 12] == np.array(ret))

    ret = tree.query_ball_point([[2, 0], [3, 0]], 1 + 1e-9)
    assert np.all([4, 8, 9, 12] == np.array(sorted(ret[0])))
    assert np.all([8, 12, 13] == np.array(sorted(ret[1])))

    ret = tree.query_ball_point([[2, 0], [3, 0]], 1 + 1e-9, return_length=True)
    assert np.all([4, 3] == np.array(ret))


def test_scipy_cKDTree():
    from scipy.spatial import cKDTree

    _test_cKDTree_query(cKDTree)


def test_nanoflann_KDTree():
    from fast_crossing.spatial import KDTree

    _test_cKDTree_query(KDTree)


def test_arrow():
    arrow = Arrow()
    assert np.all(arrow.label() == [-1, -1])
    assert arrow.t() == -1
    assert arrow.range() == -1
    assert np.all(arrow.position() == [0, 0, 0])
    assert np.all(arrow.direction() == [0, 0, 1])
    # https://stackoverflow.com/questions/47909048/what-will-be-atan2-output-for-both-x-and-y-as-0
    h = arrow.heading()
    assert h == 0.0 or np.isnan(h) or np.isinf(h)

    arrow.position([1, 2, 3])
    assert np.all(arrow.position() == [1, 2, 3])
    arrow.direction([3, 4, 12])
    assert np.all(arrow.direction() == [3, 4, 12])
    arrow.direction([3, 4, 0], True)
    np.testing.assert_allclose(arrow.direction(), [3 / 5, 4 / 5, 0], atol=1e-6)

    print(arrow.label())
    arrow.label([5, 10])
    assert np.all(arrow.label() == [5, 10])
    arrow.label(5, 20)
    assert np.all(arrow.label() == [5, 20])
    arrow.label(5, 20, t=0.3, range=23.0)
    assert arrow.t() == 0.3
    assert arrow.range() == 23.0
    arrow.label(5, 20, range=46.0)
    assert arrow.t() == 0.3
    assert arrow.range() == 46.0

    arrow.heading(90.0)
    np.testing.assert_allclose(arrow.direction(), [1, 0, 0], atol=1e-8)
    np.testing.assert_allclose(arrow.heading(), 90.0, atol=1e-8)
    arrow.heading(0.0)
    np.testing.assert_allclose(arrow.direction(), [0, 1, 0], atol=1e-8)
    np.testing.assert_allclose(arrow.heading(), 0.0, atol=1e-8)


def test_quiver():
    quiver = Quiver()
    assert quiver is not None


def pytest_main(dir: str, *, test_file: str = None):

    os.chdir(dir)
    sys.exit(
        pytest.main(
            [
                dir,
                *(["-k", test_file] if test_file else []),
                "--capture",
                "tee-sys",
                "-vv",
                "-x",
            ]
        )
    )


if __name__ == "__main__":
    pwd = os.path.abspath(os.path.dirname(__file__))
    pytest_main(pwd, test_file=os.path.basename(__file__))
