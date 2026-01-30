from __future__ import annotations

import os
import sys
import time

import numpy as np
import pytest

from fast_crossing import (
    Arrow,
    FastCrossing,
    FlatBush,
    KdQuiver,
    KdTree,
    PolylineRuler,
    Quiver,
    densify_polyline,
    point_in_polygon,
    polyline_in_polygon,
    tf,
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

    # num_polylines
    assert fc.num_polylines() == 2
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

    xyz, dir = fc.arrow(polyline_index=0, point_index=0)
    assert np.all(xyz == [0, 0, 0])
    assert np.all(dir == [1, 0, 0])

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
        assert np.all(xy == [2.5, 0])
        assert np.all(ts == [0.5, 1 / 3.0])
        assert np.all(label1 == [0, 0])
        assert np.all(label2 == [1, 1])

    # query intersections against provided polyline
    polyline = np.array([[-6.0, -1.0], [-5.0, 1.0], [5.0, -1.0]])
    ret = fc.intersections(polyline)
    ret = np.array(ret)
    xy = ret[:, 0]
    ts = ret[:, 1]
    label1 = ret[:, 2]
    label2 = ret[:, 3]
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

    assert len(fc.intersections()) == 3  # 3 == C(3,2): choose 2 from 3
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
    assert len(densify_polyline(coords, max_gap=3.0 + 1e-3)) == 3
    assert len(densify_polyline(coords, max_gap=3.0 - 1e-3)) == 4


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
    for k, expected in zip([1, 2], [expected_k1, expected_k2]):
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
    for k, expected in zip(
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
    assert np.all(np.array(ret) == [4, 8, 9, 12])

    ret = tree.query_ball_point([[2, 0], [3, 0]], 1 + 1e-9)
    assert np.all(np.array(sorted(ret[0])) == [4, 8, 9, 12])
    assert np.all(np.array(sorted(ret[1])) == [8, 12, 13])

    ret = tree.query_ball_point([[2, 0], [3, 0]], 1 + 1e-9, return_length=True)
    assert np.all(np.array(ret) == [4, 3])


def test_scipy_cKDTree():
    from scipy.spatial import cKDTree

    _test_cKDTree_query(cKDTree)


def test_nanoflann_KDTree():
    from fast_crossing.spatial import KDTree

    _test_cKDTree_query(KDTree)


def test_arrow():
    arrow = Arrow()
    assert np.all(arrow.label() == [-1, -1])
    assert np.isnan(arrow.t())
    assert np.isnan(arrow.range())
    assert not arrow.has_index()
    assert np.all(arrow.position() == [0, 0, 0])
    assert np.all(arrow.direction() == [1, 0, 0])
    h = arrow.heading()
    assert h == 90.0

    arrow.label([2, 3]).t(0.5).range(23.0)
    assert arrow.has_index()

    arrow.position([1, 2, 3])
    assert np.all(arrow.position() == [1, 2, 3])
    arrow.direction([3, 4, 12])
    assert np.all(arrow.direction() == [3, 4, 12])
    arrow.direction(Arrow._unit_vector([3, 4, 0]))
    np.testing.assert_allclose(arrow.direction(), [3 / 5, 4 / 5, 0], atol=1e-6)

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

    arrow.t(0.000001)
    assert "label:(5/20/0.000)" in repr(arrow)
    # same like f-string format
    assert f"{0.009:.2f}" == "0.01"
    assert f"{0.001:.2f}" == "0.00"

    arrow2 = arrow.copy()
    arrow2.t(0.3)
    assert arrow.t() == 0.000001

    arrow = Arrow().label([0, 0]).t(0)
    assert not arrow.has_index()
    assert arrow.has_index(check_range=False)
    arrow.range(10)
    assert arrow.has_index()
    assert arrow.has_index(check_range=False)
    arrow.reset_index()
    assert not arrow.has_index()


def test_quiver():
    quiver = Quiver()
    assert not quiver.is_wgs84()
    assert np.all(quiver.k() == [1, 1, 1])
    assert np.all(quiver.anchor() == [0, 0, 0])

    quiver = Quiver([123, 45, 6])
    assert quiver.is_wgs84()
    assert np.all(quiver.anchor() == [123, 45, 6])
    k = quiver.k()
    np.testing.assert_allclose(k, [78846.8350939781, 111131.7774141756, 1.0], atol=1e-6)
    np.testing.assert_allclose(np.sum(k @ quiver.inv_k()), 3.0, atol=1e-16)

    arrow = Arrow([123, 45, 6])
    updated = quiver.update(arrow, [1, 1, 1], update_direction=True)
    np.testing.assert_allclose(
        updated.position(), [123.00001268281724, 45.000008998326344, 7], atol=1e-8
    )
    np.testing.assert_allclose(updated.heading(), 45, atol=1e-8)

    updated = quiver.update(arrow, [1, 1, 0], update_direction=False)
    np.testing.assert_allclose(updated.direction(), [1, 0, 0], atol=1e-8)

    # delta
    #   ^ y (north)
    #   |
    #   @->
    #   |
    #   o---------> x (east)
    #
    quiver = Quiver()
    arrow = Arrow([0, 1, 0], direction=[1, 0, 0])
    # towards (delta in Frenet, x->forwards, y->leftwards, z->upwards)
    updated = quiver.towards(arrow, [2, 0, 0])
    np.testing.assert_allclose(updated.position(), [2, 1, 0], atol=1e-8)
    # update (delta in EUN, x->east, y->north, z->up)
    updated = quiver.update(arrow, [2, 0, 0])
    np.testing.assert_allclose(updated.position(), [2, 1, 0], atol=1e-8)
    arrow = Arrow([0, 1, 0], direction=Arrow._unit_vector([1, 1, 0]))
    updated = quiver.towards(arrow, [3, 3, 0])
    sqrt2 = np.sqrt(2)
    np.testing.assert_allclose(updated.position(), [0, 3 * sqrt2 + 1, 0], atol=1e-8)
    np.testing.assert_allclose(updated.direction(), [0, 1, 0], atol=1e-8)
    updated = quiver.update(arrow, [3, 3, 0])
    np.testing.assert_allclose(updated.position(), [3, 4, 0], atol=1e-8)
    np.testing.assert_allclose(
        updated.direction(), [sqrt2 / 2, sqrt2 / 2, 0], atol=1e-8
    )


def test_kdquiver():
    quiver = KdQuiver()
    """
    #  0   1   2   3   4   5   6  7   8   9   10   11  12  13   14
    #                                                  o
    #                                                 /
    #0 o-------------o-------------------------------+-----------o
    #1        o-------------o-----------------o     /
    #2                   o-------------o----------o
    #3 o-------------o-------------------------------------------o right-to-left
    #4 o-------------o-------------------------------------------o z += 10
    """
    y = 5.0
    quiver.add([[0, y], [3.5, y], [14.0, y]])  # 0
    y -= 1
    quiver.add([[1.8, y], [5.2, y], [10.0, y]])  # 1
    y -= 1
    quiver.add([[4.5, y], [8.2, y], [10.8, y], [12.0, y + 5]])  # 2
    y -= 1
    quiver.add([[0, y], [3.5, y], [14.0, y]][::-1])  # 3
    y -= 1
    quiver.add([[0, y, 10.0], [3.5, y, 10.0], [14.0, y, 10.0]])  # 4

    # nearest
    i, d = quiver.nearest([0.0, 5.0, 0.0])
    a = quiver.arrow(i)
    # print(i, d, a)
    assert i == 0 and d == 0.0 and a.polyline_index() == 0 and a.segment_index() == 0

    i, d = quiver.nearest(0)
    a = quiver.arrow(i)
    # print(i, d, a)
    assert i == 3 and a.polyline_index() == 1 and a.segment_index() == 0

    ii, dd = quiver.nearest([4.0, 2.5, 0.0], radius=3)
    arrows = quiver.arrows(ii)
    # print(ii, dd, arrows)
    # for arrow in arrows:
    #     print(arrow)
    assert len(ii) == len(dd) == len(arrows) == 5
    assert np.all(ii == [6, 11, 4, 1, 3])
    ii2, dd2 = quiver.nearest([4.0, 2.5, 0.0], k=5)
    assert np.all(ii == ii2)
    assert np.all(dd == dd2)

    filter = Quiver.FilterParams()
    assert filter.x_slots() is None
    assert filter.y_slots() is None
    assert filter.z_slots() is None
    assert filter.angle_slots() is None
    filter.x_slots([1, 2, 3, 4]).angle_slots([-50, 50])
    assert np.all(filter.x_slots() == [1, 2, 3, 4])
    assert np.all(filter.angle_slots() == [-50, 50])

    arrow = Arrow([4.0, 2.5, 0.0], [1.0, 0.0, 0.0])
    ii, dd = quiver.nearest(arrow.position(), radius=3)
    assert len(ii) == 5
    # for arr in quiver.arrows(ii):
    #     print('\n', arr, sep=';')
    ii2 = quiver.filter(
        hits=ii,
        arrow=arrow,
        params=Quiver.FilterParams(),
    )
    assert np.all(ii == ii2)
    ii_lefts = quiver.filter(
        hits=ii,
        arrow=arrow,
        params=Quiver.FilterParams().y_slots([0.0, 10]),  # only lefts
    )
    assert len(ii_lefts) == 4
    ii_rights = quiver.filter(
        hits=ii,
        arrow=arrow,
        params=Quiver.FilterParams().y_slots([-10, 0]),  # only rights
    )
    assert len(ii_rights) == 1
    assert np.all(np.sort(ii) == np.sort([*ii_lefts, *ii_rights]))
    ii, dd = quiver.nearest(arrow.position(), radius=100)
    assert len(ii) == 16
    ii, dd = quiver.nearest(arrow.position(), radius=100)
    ii = quiver.filter(
        hits=ii,
        arrow=arrow,
        params=Quiver.FilterParams().z_slots([9, 11]),
    )
    assert len(ii) == 3


def test_kdquiver_filter_by_angle():
    quiver = KdQuiver()
    vecs = {}
    headings = [0, 30, 60, 90, 120]
    for heading in headings:
        rad = np.radians(90.0 - heading)
        vec = np.array([np.cos(rad), np.sin(rad), 0.0])
        vecs[heading] = vec
        polyline = np.array([vec * 80, vec * 100])
        quiver.add(polyline, index=heading)
    np.testing.assert_allclose(30, Arrow._angle(vecs[30], ref=vecs[60]), atol=1e-15)
    np.testing.assert_allclose(-30, Arrow._angle(vecs[90], ref=vecs[60]), atol=1e-15)

    arrow = Arrow([0, 0, 0]).heading(60)
    ii, dd = quiver.nearest(arrow.position(), radius=90)
    assert len(ii) == len(headings)
    np.testing.assert_allclose(dd, [80.0] * len(ii), atol=1e-15)

    ii60 = quiver.filter(
        hits=ii,
        arrow=arrow,
        params=Quiver.FilterParams().angle_slots([-1, 1]),
    )
    hits = np.array([a.heading() for a in quiver.arrows(ii60)])
    np.testing.assert_allclose(hits, [60.0], atol=1e-15)

    ii30_60 = quiver.filter(
        hits=ii,
        arrow=arrow,
        params=Quiver.FilterParams().angle_slots([-1, 31]),
    )
    hits = np.array([a.heading() for a in quiver.arrows(ii30_60)])
    np.testing.assert_allclose(sorted(hits), [30.0, 60.0], atol=1e-15)

    ii30_60_120 = quiver.filter(
        hits=ii,
        arrow=arrow,
        params=Quiver.FilterParams().angle_slots([-1, 31, -61, -59]),
    )
    hits = np.array([a.heading() for a in quiver.arrows(ii30_60_120)])
    np.testing.assert_allclose(sorted(hits), [30.0, 60.0, 120.0], atol=1e-15)


def test_within():
    fc = FastCrossing()
    """
          0A      ┌────────┐
           ****   │    *2C │
                  │    *   │
    1B  ┌─────────o    *   │
      * │                  │
       *│               **********3D
        *         *        │
        └─*───────*────────┘
           *      *
            *     *4E
    """
    polygon = np.array(
        [[0, 0], [-10, 0], [-10, -10], [10, -10], [10, 10], [0, 10]], dtype=np.float64
    )
    fc.add_polyline([[-8, 9], [-2, 9]])  # 0A
    fc.add_polyline([[-12, -1], [-8, -8], [-5, -15]])  # 1B
    fc.add_polyline([[6, 0], [6, 9]])  # 2C
    fc.add_polyline([[7, -5], [18, -5]])  # 3D
    fc.add_polyline([[0, -5], [0, -15]])  # 4E
    hits = np.array(fc.within(polygon=polygon))
    assert np.all(
        hits
        == [
            [1, 0],
            [1, 1],
            [2, 0],
            [3, 0],
            [4, 0],
        ]
    )
    hits = np.array(fc.within(polygon=polygon, segment_wise=False))
    assert np.all(
        hits
        == [
            [1, 1],
            [2, 0],
            [2, 1],
            [3, 0],
            [4, 0],
        ]
    )

    hits = np.array(fc.within(min=np.array([0.0, 0.0]), max=np.array([10.0, 10.0])))
    assert np.all(hits == [[2, 0]])
    hits = np.array(fc.within(min=np.array([-10.0, -10.0]), max=np.array([10.0, 10.0])))
    assert np.all(
        hits
        == [
            [0, 0],
            [1, 0],
            [1, 1],
            [2, 0],
            [3, 0],
            [4, 0],
        ]
    )

    # fc.add_polyline([[-8, 9], [-2, 9]])  # 0A
    hits = np.array(fc.within(center=np.array([-5.0, 10.0]), width=6.0, height=0.5))
    assert not len(hits)
    hits = np.array(
        fc.within(center=np.array([-5.0, 10.0]), width=6.0, height=0.5, heading=30.0)
    )
    assert np.all(hits == [[0, 0]])


def test_nearst():
    fc = FastCrossing()
    tree = fc.quiver()
    assert tree is None
    bush = fc.bush()
    assert bush is None
    """
          0A      ┌────────┐
           ****   │    *2C │
                  │    *   │
    1B  ┌─────────o    *   │
      * │                  │
       *│               **********3D
        *         *        │
        └─*───────*────────┘
           *      *
            *     *4E
    """
    polygon = np.array(
        [[0, 0], [-10, 0], [-10, -10], [10, -10], [10, 10], [0, 10]], dtype=np.float64
    )
    assert len(polygon)
    fc.add_polyline([[-8, 9], [-2, 9]])  # 0A
    fc.add_polyline([[-12, -1], [-8, -8], [-5, -15]])  # 1B
    fc.add_polyline([[6, 0], [6, 9]])  # 2C
    fc.add_polyline([[7, -5], [18, -5]])  # 3D
    fc.add_polyline([[0, -5], [0, -15]])  # 4E

    # nearest
    idx, dist = fc.nearest(np.array([0.0, 0.0, 0.0]))
    assert np.all(idx == [4, 0]) and np.fabs(dist - 5.0) < 1e-6
    idx, dist = fc.nearest(np.array([0.0, 0.0, 0.0]), return_squared_l2=True)
    assert np.all(idx == [4, 0]) and np.fabs(dist - 25.0) < 1e-6

    # nearest
    idx, dist = fc.nearest(np.array([0, 0]))
    assert np.all(idx == [0, 1]) and np.fabs(dist - 6.0) < 1e-6
    idx, dist = fc.nearest(np.array([0, 1]))
    assert np.all(idx == [0, 0]) and np.fabs(dist - 6.0) < 1e-6

    # nearest k
    idx, dist = fc.nearest(np.array([0.0, 0.0, 0.0]), k=2)
    assert np.all(idx == [[4, 0], [2, 0]])
    np.testing.assert_allclose(dist, [5, 6], atol=1e-15)
    # nearest radius
    idx, dist = fc.nearest(np.array([0.0, 0.0, 0.0]), radius=6 + 1e-3)
    assert np.all(idx == [[4, 0], [2, 0]])
    np.testing.assert_allclose(dist, [5, 6], atol=1e-15)
    idx, dist = fc.nearest(np.array([0.0, 0.0, 0.0]), radius=5 + 1e-3)
    assert np.all(idx == [[4, 0]])
    np.testing.assert_allclose(dist, [5], atol=1e-15)

    arrow = Arrow([-4.0, 0.0, 0.0], [0, -1, 0])
    idx, dist = fc.nearest(arrow.position(), radius=10.0)
    # print(idx, dist)
    assert len(idx) == 5
    idx, dist = fc.nearest(
        arrow.position(),
        radius=10.0,
        filter=[
            arrow.direction(),
            Quiver.FilterParams().angle_slots([-1, 1]),
        ],
    )
    # print(idx, dist)
    assert len(idx) == 1
    assert np.all(idx == [[4, 0]])

    tree = fc.quiver()
    assert tree is not None
    bush = fc.bush(autobuild=False)
    assert bush is None
    bush = fc.bush()
    assert bush is not None


def test_nearst_wgs84():
    fc = FastCrossing()
    fc.add_polyline([[-8, 9], [-2, 9]])  # 0A
    fc.add_polyline([[-12, -1], [-8, -8], [-5, -15]])  # 1B
    fc.add_polyline([[6, 0], [6, 9]])  # 2C
    fc.add_polyline([[7, -5], [18, -5]])  # 3D
    fc.add_polyline([[0, -5], [0, -15]])  # 4E
    arrow = Arrow([-4.0, 0.0, 0.0], [0, -1, 0])
    idx, dist = fc.nearest(arrow.position(), radius=10.0)
    expected_ii = np.array([[4, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    expected_dd = np.array([6.40312424, 8.06225775, 8.94427191, 9.21954446, 9.8488578])
    assert np.all(idx == expected_ii)
    np.testing.assert_allclose(dist, expected_dd, atol=1e-8)
    assert len(idx) == 5
    idx, dist = fc.nearest(
        arrow.position(),
        radius=10.0,
        filter=[
            arrow.direction(),
            Quiver.FilterParams().angle_slots([-1, 1]),
        ],
    )
    assert len(idx) == 1
    assert np.all(idx == [[4, 0]])

    xyzs0 = fc.quiver().positions()
    # print(xyzs0)

    fc2 = FastCrossing(is_wgs84=True)
    anchor_lla = np.array([123.4, 56.7, 8.9])
    for index, ruler in fc.polyline_rulers().items():
        enus = ruler.polyline()
        llas = tf.enu2lla(enus, anchor_lla=anchor_lla)
        fc2.add_polyline(llas, index=index)
    # print(fc2.quiver().anchor())
    xyzs1 = tf.lla2enu(
        tf.enu2lla(fc2.quiver().positions(), anchor_lla=fc2.quiver().anchor()),
        anchor_lla=anchor_lla,
    )
    # print(xyzs1)
    np.testing.assert_allclose(xyzs0, xyzs1, atol=1e-8)

    position = tf.enu2lla([-4, 0, 0], anchor_lla=anchor_lla).reshape(-1)
    arrow = Arrow(position, [0, -1, 0])
    idx, dist = fc2.nearest(arrow.position(), radius=10.0)
    # precision error??
    # print(idx, dist)
    np.testing.assert_allclose(dist[:4], expected_dd[:4], atol=1e-4)
    # assert len(idx) == 5


def __print_chunks(chunks):
    for cidx, ((seg1, t1, r1, seg2, t2, r2), polyline) in enumerate(chunks.items()):
        print(f"\nchunk index: {cidx}", sep=", ")
        print(f"length: {r2 - r1:.3f}")
        print(f"seg={seg1},t={t1:.3f}, r={r1:.2f}", sep="; ")
        print(f"seg={seg2},t={t2:.3f}, r={r2:.2f}")
        print(polyline.round(2).tolist())


def test_polyline_in_polygon():
    """
           2    4
           *    *
      A   /|   /|
       o---+--/-+--------------o D
       |/  | /  |              |
       /   |/   |              |
      /|   *    *              |
    1* |   3    5              |
      Bo-----------------------o
                               C
    """
    polygon_ABCD = np.array(
        [
            [0.0, 0.0],  # A
            [0.0, -10.0],  # B
            [20.0, -10.0],  # C
            [20.0, 0.0],  # D
            [0.0, 0.0],  # A
        ]
    )
    polyline_12345 = np.array(
        [
            [-2.0, -9.0, 0.0],  # 1
            [3.0, 7.0, 1.0],  # 2
            [3.0, -7.0, 2.0],  # 3
            [8.0, 7.0, 3.0],  # 4
            [8.0, -7.0, 4.0],  # 5
        ]
    )
    ruler = PolylineRuler(polyline_12345, is_wgs84=False)
    chunks = polyline_in_polygon(polyline_12345, polygon_ABCD)
    ranges = []
    for _, _, r1, _, _, r2 in chunks:
        ranges.append(r2 - r1)
    expected_ranges = [2.72883, 14.4676666, 7.01783]
    np.testing.assert_allclose(ranges, expected_ranges, atol=1e-4)

    # test inside
    polyline = next(iter(chunks.values()))
    polyline_updated = np.copy(polyline_12345)
    polyline_updated[0] = polyline.mean(axis=0)
    chunks2 = polyline_in_polygon(polyline_updated, polygon_ABCD)
    __print_chunks(chunks)
    __print_chunks(chunks2)

    chunks1 = list(chunks.items())
    chunks2 = list(chunks2.items())
    assert len(chunks1) == len(chunks2)
    assert np.fabs(chunks1[0][1][-1] - chunks2[0][1][-1]).max() == 0.0
    assert np.fabs(chunks1[1][1] - chunks2[1][1]).sum() < 1e-6
    assert np.fabs(chunks1[2][1] - chunks2[2][1]).sum() < 1e-6

    chunks3 = polyline_in_polygon(ruler.lineSliceAlong(1, 2), polygon_ABCD)
    assert len(chunks3) == 0
    chunks3 = polyline_in_polygon(ruler.lineSliceAlong(7, 8), polygon_ABCD)
    assert len(chunks3) == 1

    # test fc
    fc = FastCrossing()
    fc.add_polyline(polygon_ABCD)
    chunks2 = polyline_in_polygon(polyline_12345, polygon_ABCD, fc=fc)
    assert list(chunks.keys()) == list(chunks2.keys())
    N = 1000
    tick = time.time()
    for _ in range(N):
        polyline_in_polygon(polyline_12345, polygon_ABCD)
    delta1 = time.time() - tick
    tick = time.time()
    for _ in range(N):
        polyline_in_polygon(polyline_12345, polygon_ABCD, fc=fc)
    delta2 = time.time() - tick
    print(delta1, delta2)
    # assert delta2 < delta1

    anchor_lla = [123.4, 56.7, 8.9]
    chunks = polyline_in_polygon(
        tf.enu2lla(polyline_12345, anchor_lla=anchor_lla),
        tf.enu2lla(
            np.c_[polygon_ABCD, np.zeros(len(polygon_ABCD))], anchor_lla=anchor_lla
        )[:, :2],
        is_wgs84=True,
    )
    ranges = []
    for _, _, r1, _, _, r2 in chunks:
        ranges.append(r2 - r1)
    np.testing.assert_allclose(ranges, expected_ranges, atol=1e-4)

    chunks = polyline_in_polygon([[100, 0, 0], [200, 0, 0]], polygon_ABCD)
    assert len(chunks) == 0

    # TODO, test touches


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


def test_duplicates():
    fc = FastCrossing()
    fc.add_polyline(np.array([[0.0, 0.0], [5.0, 0.0]]))
    assert len(fc.intersections()) == 0
    fc.add_polyline(np.array([[0.0, 0.0], [5.0, 0.0]]))
    assert len(fc.intersections()) == 1


def test_flatbush():
    polyline = [
        [0, 0],
        [10, 0],
        [10, 5],
        [20, 5],
    ]
    fb = FlatBush()
    fb.add(polyline)
    fb.add(polyline[::-1])
    assert np.all(
        fb.labels()
        == [
            [-1, 0],
            [-1, 1],
            [-1, 2],
            [-1, 0],
            [-1, 1],
            [-1, 2],
        ]
    )
    assert np.all(
        fb.boxes()
        == [
            [0.0, 0.0, 10.0, 0.0],
            [10.0, 0.0, 10.0, 5.0],
            [10.0, 5.0, 20.0, 5.0],
            [20.0, 5.0, 10.0, 5.0],
            [10.0, 5.0, 10.0, 0.0],
            [10.0, 0.0, 0.0, 0.0],
        ]
    )

    assert not fb.search([0, 0], [10, 10])
    fb.finish()
    assert fb.search([0, 0], [10, 10])

    assert sorted(
        fb.search(
            [
                0,
                0,
            ],
            [0, 0],
        )
    ) == [0, 5]
    assert np.all(fb.box(0)[:2] == fb.box(5)[-2:])
    assert np.all(fb.box(0)[-2:] == fb.box(5)[:2])

    fb = FlatBush()
    fb.add(polyline)
    fb.add(polyline[::-1])
    fb.add(polyline)
    fb.add(polyline[::-1])
    fb.finish()
    assert sorted(
        fb.search(
            [
                0,
                0,
            ],
            [0, 0],
        )
    ) == [0, 5, 6, 11]


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    pwd = os.path.abspath(os.path.dirname(__file__))
    pytest_main(pwd, test_file=os.path.basename(__file__))
