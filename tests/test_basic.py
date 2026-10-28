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
    ScanlineHitFilterOptions,
    crop,
    crop_labels,
    densify_polyline,
    point_in_polygon,
    polyline_chunks_in_polygon,
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
    pytest.importorskip("scipy")
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


def test_polyline_chunks_in_polygon():
    """
    polyline_chunks_in_polygon should return the same chunk boundaries (labels)
    as polyline_in_polygon but without the coordinate data.
    """
    polygon_ABCD = np.array(
        [
            [0.0, 0.0],
            [0.0, -10.0],
            [20.0, -10.0],
            [20.0, 0.0],
            [0.0, 0.0],  # closed
        ]
    )
    polyline_12345 = np.array(
        [
            [-2.0, -9.0, 0.0],
            [3.0, 7.0, 1.0],
            [3.0, -7.0, 2.0],
            [8.0, 7.0, 3.0],
            [8.0, -7.0, 4.0],
        ]
    )
    # polyline_in_polygon returns chunks with coordinates
    chunks = polyline_in_polygon(polyline_12345, polygon_ABCD)
    assert len(chunks) == 3

    # polyline_chunks_in_polygon returns labels only, no coordinates
    labels_nfc = polyline_chunks_in_polygon(polyline_12345, polygon_ABCD)
    assert len(labels_nfc) == 3

    # with pre-built FC
    fc = FastCrossing()
    fc.add_polyline(polygon_ABCD)
    fc.finish()
    labels_fc = polyline_chunks_in_polygon(polyline_12345, polygon_ABCD, fc=fc)
    assert len(labels_fc) == 3

    # labels should match chunks keys
    chunk_keys = list(chunks.keys())
    for i, label in enumerate(labels_nfc):
        assert label == chunk_keys[i], f"{label} != {chunk_keys[i]}"
    for i, label in enumerate(labels_fc):
        assert label == chunk_keys[i], f"{label} != {chunk_keys[i]}"

    # confirm range values match expected
    expected_ranges = [2.72883, 14.4676666, 7.01783]
    for i, (_, _, r1, _, _, r2) in enumerate(labels_nfc):
        np.testing.assert_allclose(r2 - r1, expected_ranges[i], atol=1e-4)


def test_crop():
    """crop() batches polyline_in_polygon across all polylines in an FC."""
    polygon = np.array(
        [
            [0.0, 0.0],
            [0.0, -10.0],
            [20.0, -10.0],
            [20.0, 0.0],
            [0.0, 0.0],  # closed
        ]
    )
    # build obstacle FC with 3 polylines, 2 intersect the polygon
    obs_fc = FastCrossing()
    obs_fc.add_polyline(  # polyline 0: fully inside
        np.array([[1.0, -1.0, 0], [5.0, -5.0, 0], [10.0, -1.0, 0]])
    )
    obs_fc.add_polyline(  # polyline 1: crosses the polygon boundary
        np.array([[-5.0, -5.0, 0], [5.0, -5.0, 0], [25.0, -5.0, 0]])
    )
    obs_fc.add_polyline(  # polyline 2: fully outside
        np.array([[-5.0, -5.0, 0], [-3.0, -5.0, 0]])
    )
    obs_fc.finish()

    # test crop with auto-build polygon FC
    result = crop(obs_fc, polygon, is_wgs84=False)
    assert 0 in result  # polyline 0 fully inside
    assert 1 in result  # polyline 1 crosses polygon
    assert 2 not in result  # polyline 2 fully outside

    # polyline 0 is fully inside: single chunk spanning whole polyline
    assert len(result[0]) == 1
    # polyline 1 crosses: 1 chunk (the inside portion)
    assert len(result[1]) == 1
    # check the inside portion of polyline 1
    (_, _, r1, _, _, r2) = next(iter(result[1].keys()))
    np.testing.assert_allclose(r1, 5.0, atol=1e-6)  # starts at x=0 boundary
    np.testing.assert_allclose(r2, 25.0, atol=1e-6)  # ends at x=20 boundary

    # test crop with pre-built polygon FC
    poly_fc = FastCrossing()
    poly_fc.add_polyline(polygon)
    poly_fc.finish()
    result2 = crop(obs_fc, polygon, polygon_fc=poly_fc)
    assert result.keys() == result2.keys()
    for pid in result:
        assert list(result[pid].keys()) == list(result2[pid].keys())

    # test crop_labels - should match same boundaries
    result_l = crop_labels(obs_fc, polygon, is_wgs84=False)
    assert result_l.keys() == result.keys()
    for pid in result:
        assert list(result_l[pid]) == list(result[pid].keys())


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


def __batch_test_index():
    """A small "LD corridor": parallel polylines plus a few crossing ones.

    Scale is deliberately small (index semantics, not speed): 12 polylines,
    ~500 segments. `benchmarks/benchmark_batch_query.py` uses the real scale
    (394 polylines / 4160 segments) instead.
    """
    polylines = []
    for i in range(10):
        x = np.arange(0.0, 60.0, 1.5)
        y = -5.0 + i * 1.1 + 0.3 * np.sin(x / 7.0 + i)
        polylines.append(np.c_[x, y, np.zeros_like(x)])
    for j in range(4):
        x = 15.0 * j + 3.0
        polylines.append(np.array([[x, -9.0, 0.0], [x, 9.0, 0.0]], dtype=np.float64))
    fc = FastCrossing()
    for i, polyline in enumerate(polylines):
        fc.add_polyline(np.ascontiguousarray(polyline), index=i)
    fc.finish()
    return fc, polylines


def __hit_options(**kwargs) -> ScanlineHitFilterOptions:
    from fast_crossing import ScanlineHitFilterOptions

    options = ScanlineHitFilterOptions()
    options.accept_opposite = True
    options.min_dot = 0.9
    options.max_height_offset = 0.0
    for key, value in kwargs.items():
        setattr(options, key, value)
    return options


def __queries(num: int = 300, seed: int = 11):
    rng = np.random.default_rng(seed)
    llas = np.c_[
        rng.uniform(0.0, 50.0, num),
        rng.uniform(-8.0, 8.0, num),
        np.zeros(num),
    ]
    heading = rng.uniform(-0.25, 0.25, num)
    dirs = np.c_[np.cos(heading), np.sin(heading), np.zeros(num)]
    return llas, dirs


def test_nearest_batch():
    fc, _ = __batch_test_index()
    llas, _ = __queries()

    # k mode: element-wise equal to the per-point loop
    for k in (1, 2, 5):
        expected = [np.atleast_2d(fc.nearest(p, k=k)[0]) for p in llas]
        expected_dist = [np.atleast_1d(fc.nearest(p, k=k)[1]) for p in llas]
        counts, pid, pidx, dists = fc.nearest_batch(llas, k=k)
        assert dists.shape == (len(llas), k)
        assert np.all(counts == [len(e) for e in expected])
        for i, (label, dist) in enumerate(zip(expected, expected_dist)):
            n = counts[i]
            assert np.array_equal(np.c_[pid[i, :n], pidx[i, :n]], label)
            np.testing.assert_allclose(dists[i, :n], dist, atol=1e-12)
            # padding
            assert np.all(pid[i, n:] == -1) and np.all(pidx[i, n:] == -1)
            assert np.all(np.isinf(dists[i, n:]))

    # radius mode: same story, plus `max_hits` truncation
    for radius in (1.5, 3.0):
        expected = [np.atleast_2d(fc.nearest(p, radius=radius)[0]) for p in llas]
        expected_dist = [np.atleast_1d(fc.nearest(p, radius=radius)[1]) for p in llas]
        counts, pid, pidx, dists = fc.nearest_batch(llas, radius=radius)
        assert np.all(counts == [len(e) for e in expected])
        for i, (label, dist) in enumerate(zip(expected, expected_dist)):
            n = counts[i]
            assert np.array_equal(np.c_[pid[i, :n], pidx[i, :n]], label)
            np.testing.assert_allclose(dists[i, :n], dist, atol=1e-12)

    full = fc.nearest_batch(llas, radius=3.0)
    counts, pid, pidx, dists = fc.nearest_batch(llas, radius=3.0, max_hits=2)
    assert dists.shape == (len(llas), 2)
    # counts is the pre-truncation count
    assert np.array_equal(counts, full[0])
    truncated = counts > 2
    assert truncated.any()
    np.testing.assert_allclose(dists[truncated], full[3][truncated][:, :2])
    assert np.array_equal(pid[truncated], full[1][truncated][:, :2])

    # return_squared_l2
    _, _, _, d2 = fc.nearest_batch(llas, radius=3.0, return_squared_l2=True)
    _, _, _, d1 = fc.nearest_batch(llas, radius=3.0)
    finite = np.isfinite(d1)
    np.testing.assert_allclose(d2[finite], d1[finite] ** 2, atol=1e-12)
    assert np.all(np.isinf(d2[~finite]))

    # k and radius are mutually exclusive; k must be positive
    with pytest.raises(ValueError, match="k / radius"):
        fc.nearest_batch(llas, k=1, radius=2.0)
    with pytest.raises(ValueError, match="k should be positive"):
        fc.nearest_batch(llas, k=0)

    # empty index: predictable shapes, everything misses
    empty = FastCrossing()
    counts, pid, pidx, dists = empty.nearest_batch(llas, k=2)
    assert counts.shape == (len(llas),) and np.all(counts == 0)
    assert pid.shape == pidx.shape == dists.shape == (len(llas), 2)
    assert np.all(pid == -1) and np.all(np.isinf(dists))


def test_intersections_batch():
    fc, _ = __batch_test_index()
    llas, dirs = __queries()

    for min_, max_ in ((-1.2, 1.2), (-5.0, 5.0), (-10.0, 10.0)):
        options = __hit_options()
        expected = [
            fc.scanline_hits(p, d, min=min_, max=max_, options=options)
            for p, d in zip(llas, dirs)
        ]
        counts, lateral, pid, seg_idx, t, dot, lla, direction = fc.intersections_batch(
            llas, dirs, min=min_, max=max_, options=options
        )
        width = lateral.shape[1]
        assert width == max(len(h) for h in expected)
        assert lla.shape == direction.shape == (len(llas) * width, 3)
        lla3 = lla.reshape(len(llas), width, 3)
        dir3 = direction.reshape(len(llas), width, 3)
        assert np.all(counts == [len(h) for h in expected])
        for i, hits in enumerate(expected):
            n = counts[i]
            for j, hit in enumerate(hits):
                assert pid[i, j] == hit.polyline_index
                assert seg_idx[i, j] == hit.segment_index
                np.testing.assert_allclose(lateral[i, j], hit.lateral_offset)
                np.testing.assert_allclose(t[i, j], hit.t)
                np.testing.assert_allclose(dot[i, j], hit.dot)
                np.testing.assert_allclose(lla3[i, j], hit.lla, atol=1e-12)
                np.testing.assert_allclose(dir3[i, j], hit.dir, atol=1e-12)
            # unused slots
            assert np.all(pid[i, n:] == -1) and np.all(seg_idx[i, n:] == -1)
            assert np.all(np.isinf(lateral[i, n:]))
            assert np.all(np.isnan(t[i, n:])) and np.all(np.isnan(dot[i, n:]))
            assert np.all(np.isnan(lla3[i, n:]))

    # |lateral| ascending, and the first hit is the `min(abs(lateral))` one
    options = __hit_options()
    counts, lateral, pid, *_ = fc.intersections_batch(
        llas, dirs, min=-10.0, max=10.0, options=options
    )
    for i in range(len(llas)):
        n = counts[i]
        assert np.all(np.diff(np.abs(lateral[i, :n])) >= 0)
        hits = fc.scanline_hits(llas[i], dirs[i], min=-10.0, max=10.0, options=options)
        best = min(hits, key=lambda h: abs(h.lateral_offset))
        assert pid[i, 0] == best.polyline_index

    # only_nearest: at most one hit, and it is the closest-lateral one
    only = __hit_options(only_nearest=True)
    counts, lateral, pid, *_ = fc.intersections_batch(
        llas, dirs, min=-10.0, max=10.0, options=only
    )
    assert np.all(counts <= 1)
    assert np.sum(counts) > 0

    # max_hits truncation: counts stays the pre-truncation value, slots keep
    # the smallest |lateral| first
    full = fc.intersections_batch(llas, dirs, min=-10.0, max=10.0, options=options)
    for max_hits in (1, 2, 3):
        counts, lateral, pid, seg_idx, t, dot, lla, direction = fc.intersections_batch(
            llas,
            dirs,
            min=-10.0,
            max=10.0,
            options=options,
            max_hits=max_hits,
        )
        assert lateral.shape == (len(llas), max_hits)
        assert lla.shape == (len(llas) * max_hits, 3)
        assert np.array_equal(counts, full[0])
        truncated = counts > max_hits
        assert truncated.any()
        np.testing.assert_allclose(lateral[truncated], full[1][truncated][:, :max_hits])
        assert np.array_equal(pid[truncated], full[2][truncated][:, :max_hits])

    # NaN dir row == "no direction": no direction filtering for that row
    nan_dirs = dirs.copy()
    nan_dirs[::7] = np.nan
    counts, lateral, pid, *_ = fc.intersections_batch(
        llas, nan_dirs, min=-3.0, max=3.0, options=options, max_hits=16
    )
    for i in range(0, len(llas), 7):
        reference = fc.scanline_hits(
            llas[i],
            np.array([np.nan, np.nan, 0.0]),
            min=-3.0,
            max=3.0,
            options=options,
        )
        assert counts[i] == len(reference)
        for j, hit in enumerate(reference):
            assert pid[i, j] == hit.polyline_index
            np.testing.assert_allclose(lateral[i, j], hit.lateral_offset)

    # miss semantics with an empty index
    empty = FastCrossing()
    out = empty.intersections_batch(llas, dirs, min=-5.0, max=5.0, max_hits=4)
    counts, lateral, pid, seg_idx, t, dot, lla, direction = out
    assert np.all(counts == 0)
    assert lateral.shape == (len(llas), 4)
    assert np.all(np.isinf(lateral)) and np.all(pid == -1) and np.all(seg_idx == -1)
    assert np.all(np.isnan(t)) and np.all(np.isnan(dot))
    assert lla.shape == direction.shape == (len(llas) * 4, 3)
    assert np.all(np.isnan(lla))
    # without max_hits an empty batch has zero width
    counts, lateral, *_ = empty.intersections_batch(llas, dirs)
    assert np.all(counts == 0) and lateral.shape == (len(llas), 0)

    # argument validation
    with pytest.raises(ValueError, match="#llas != #dirs"):
        fc.intersections_batch(llas, dirs[:-1])
    with pytest.raises(ValueError, match="min should be <= max"):
        fc.intersections_batch(llas, dirs, min=5.0, max=-5.0)
    with pytest.raises(ValueError, match="max_hits"):
        fc.intersections_batch(llas, dirs, max_hits=-1)


def test_hit_filter_options_semantics():
    """`accept_opposite` + `min_dot`: same/opposite accepted, perpendicular not.

    Geometry: query at the origin pointing along (1, 1)/sqrt(2), so the scanline
    runs along (-1, 1)/sqrt(2) (the leftward normal) and lateral offsets are
    measured along it.
    """
    r2 = 2.0**0.5
    dir_ = np.array([1.0 / r2, 1.0 / r2, 0.0])
    left = np.array([-1.0 / r2, 1.0 / r2, 0.0])
    lla = np.array([0.0, 0.0, 0.0])

    # a point `lateral` meters to the left of the query point
    def at(lateral: float, z: float = 0.0) -> np.ndarray:
        return (left * lateral + np.array([0.0, 0.0, z])).tolist()

    fc = FastCrossing()
    fc.add_polyline(np.array([at(0.0, 0.0), at(0.0, 0.0)], dtype=float))
    # ^ placeholder, replaced below

    fc = FastCrossing()
    # A: along +query direction (dot = +1)
    a = np.array([-10.0 / r2, -10.0 / r2, 0.0])
    b = np.array([10.0 / r2, 10.0 / r2, 0.0])
    fc.add_polyline(np.array([a, b]), index=0)
    # B: same geometry, digitized the other way round (dot = -1)
    fc.add_polyline(np.array([b, a]), index=1)
    # C: 60 degrees off the query direction (dot = 0.5), crossing the scanline
    # 3 meters to the left of the query point. Below min_dot=0.9, so it must be
    # filtered out by the direction gate and kept when the gate is disabled.
    # (A segment exactly perpendicular to the scanline is collinear with it, so
    # the "dot == 0 crossing" case does not exist outside that degenerate
    # overlap.)
    c = np.array(at(3.0))
    d3 = np.cos(np.pi / 3.0) * dir_ + np.sin(np.pi / 3.0) * left
    off = d3 * 5.0
    fc.add_polyline(np.array([np.array(c) - off, np.array(c) + off]), index=2)
    fc.finish()

    def hits(**kwargs):
        return fc.scanline_hits(
            lla, dir_, min=-5.0, max=5.0, options=__hit_options(**kwargs)
        )

    # accept_opposite=True, min_dot=0.9: dot = +1 / -1 pass, dot = 0.5 fails
    assert sorted(h.polyline_index for h in hits(min_dot=0.9)) == [0, 1]
    assert {h.polyline_index: round(h.dot, 6) for h in hits(min_dot=0.9)} == {
        0: 1.0,
        1: -1.0,
    }

    # same direction only: the reversed polyline is dropped
    assert [h.polyline_index for h in hits(min_dot=0.9, accept_opposite=False)] == [0]

    # accept_opposite=False + negative min_dot disables the direction gate
    assert sorted(
        h.polyline_index for h in hits(min_dot=-1.0, accept_opposite=False)
    ) == [0, 1, 2]
    # accept_opposite=True + min_dot <= 0 disables it as well
    assert sorted(h.polyline_index for h in hits(min_dot=0.0)) == [0, 1, 2]

    # lateral sign: left of the query direction is positive
    by_id = {h.polyline_index: h.lateral_offset for h in hits(min_dot=0.0)}
    assert by_id[0] == pytest.approx(0.0, abs=1e-9)
    assert by_id[1] == pytest.approx(0.0, abs=1e-9)
    assert by_id[2] == pytest.approx(3.0)

    # max_height_offset
    fc_z = FastCrossing()
    fc_z.add_polyline(np.array([a, b]), index=0)
    fc_z.add_polyline(np.array([a, b]) + np.array([0.0, 0.0, 5.0]), index=1)
    fc_z.finish()
    z_hits = fc_z.scanline_hits(
        lla, dir_, min=-5.0, max=5.0, options=__hit_options(max_height_offset=1.0)
    )
    assert [h.polyline_index for h in z_hits] == [0]
    z_hits = fc_z.scanline_hits(
        lla, dir_, min=-5.0, max=5.0, options=__hit_options(max_height_offset=0.0)
    )
    assert sorted(h.polyline_index for h in z_hits) == [0, 1]

    # filter_out_hits_by_gap keeps the cluster next to lateral offset 0
    fc_g = FastCrossing()
    for i, lateral in enumerate((-0.5, 0.5, 4.0)):
        p = np.array(at(lateral))
        fc_g.add_polyline(np.array([p - off, p + off]), index=i)
    fc_g.finish()
    g_hits = fc_g.scanline_hits(
        lla,
        dir_,
        min=-5.0,
        max=5.0,
        options=__hit_options(min_dot=0.0, filter_out_hits_by_gap=2.0),
    )
    assert sorted(h.polyline_index for h in g_hits) == [0, 1]


def test_intersections_batch_wgs84():
    """Batch and per-point agree in WGS84 too, and offsets stay in meters."""
    from fast_crossing import tf

    anchor = np.array([121.5, 31.25, 0.0])
    enus = []
    for i in range(10):
        x = np.arange(0.0, 60.0, 1.5)
        y = -5.0 + i * 1.1 + 0.3 * np.sin(x / 7.0 + i)
        enus.append(np.c_[x, y, np.zeros_like(x)])
    # digitized with the opposite orientation, as LD data often is
    enus.append(np.c_[np.arange(60.0, 0.0, -1.5), np.full(40, -6.5), np.zeros(40)])

    fc = FastCrossing(is_wgs84=True)
    for i, enu in enumerate(enus):
        fc.add_polyline(
            np.ascontiguousarray(tf.enu2lla(enu, anchor_lla=anchor)), index=i
        )
    fc.finish()

    enu_llas, dirs = __queries()
    llas = tf.enu2lla(enu_llas, anchor_lla=anchor)
    options = __hit_options()
    counts, lateral, pid, seg_idx, t, dot, lla, direction = fc.intersections_batch(
        llas, dirs, min=-5.0, max=5.0, options=options
    )
    assert np.sum(counts) > 0
    # finite exactly on the valid slots
    valid = np.arange(lateral.shape[1])[None, :] < counts[:, None]
    assert np.all(np.isfinite(lateral[valid]))
    assert np.all(~np.isfinite(lateral[~valid]))
    # lateral offsets are metric: the widest spacing here is ~1.1 m
    assert np.all(np.abs(lateral[np.isfinite(lateral)]) <= 5.0 + 1e-9)
    width = lateral.shape[1]
    lla3 = lla.reshape(len(llas), width, 3)
    for i in range(len(llas)):
        reference = fc.scanline_hits(
            llas[i], dirs[i], min=-5.0, max=5.0, options=options
        )
        assert counts[i] == len(reference)
        for j, hit in enumerate(reference):
            assert pid[i, j] == hit.polyline_index
            assert seg_idx[i, j] == hit.segment_index
            np.testing.assert_allclose(lateral[i, j], hit.lateral_offset)
            np.testing.assert_allclose(lla3[i, j], hit.lla, atol=1e-12)


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    pwd = os.path.abspath(os.path.dirname(__file__))
    pytest_main(pwd, test_file=os.path.basename(__file__))
