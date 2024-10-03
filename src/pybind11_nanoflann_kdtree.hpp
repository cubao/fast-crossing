// should sync
// -
// https://github.com/cubao/fast-crossing/blob/master/src/pybind11_nanoflann_kdtree.hpp
// -
// https://github.com/cubao/headers/tree/main/include/cubao/pybind11_nanoflann_kdtree.hpp

#pragma once

#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "cubao_inline.hpp"
#include "nanoflann_kdtree.hpp"

namespace cubao
{
namespace py = pybind11;
using namespace pybind11::literals;
using rvp = py::return_value_policy;

CUBAO_INLINE void bind_nanoflann_kdtree(py::module &m)
{
    using KdTree = cubao::KdTree;
    py::class_<KdTree>(m, "KdTree", py::module_local())
        .def(py::init<int>(), "leafsize"_a = 10,
             "Initialize KdTree with specified leaf size.\n\n"
             ":param leafsize: Maximum number of points in leaf node, defaults "
             "to 10")
        .def(py::init<const RowVectors &>(), "points"_a,
             "Initialize KdTree with 3D points.\n\n"
             ":param points: 3D points to initialize the tree")
        .def(py::init<const Eigen::Ref<const RowVectorsNx2> &>(), "points"_a,
             "Initialize KdTree with 2D points.\n\n"
             ":param points: 2D points to initialize the tree")
        //
        .def("points", &KdTree::points, rvp::reference_internal,
             "Get the points in the KdTree.\n\n"
             ":return: Reference to the points in the tree")
        //
        .def("add", py::overload_cast<const RowVectors &>(&KdTree::add),
             "points"_a,
             "Add 3D points to the KdTree.\n\n"
             ":param points: 3D points to add")
        .def("add",
             py::overload_cast<const Eigen::Ref<const RowVectorsNx2> &>(
                 &KdTree::add),
             "points"_a,
             "Add 2D points to the KdTree.\n\n"
             ":param points: 2D points to add")
        //
        .def("reset", &KdTree::reset, "Reset the KdTree, clearing all points.")
        .def("reset_index", &KdTree::reset_index,
             "Reset the index of the KdTree.")
        .def("build_index", &KdTree::build_index, "force_rebuild"_a = false,
             "Build the KdTree index.\n\n"
             ":param force_rebuild: Force rebuilding the index even if already "
             "built, defaults to false")
        //
        .def("leafsize", &KdTree::leafsize,
             "Get the current leaf size of the KdTree.\n\n"
             ":return: Current leaf size")
        .def("set_leafsize", &KdTree::set_leafsize, "value"_a,
             "Set the leaf size of the KdTree.\n\n"
             ":param value: New leaf size value")
        //
        .def("nearest",
             py::overload_cast<const Eigen::Vector3d &, bool>(&KdTree::nearest,
                                                              py::const_),
             "position"_a, py::kw_only(), "return_squared_l2"_a = false,
             "Find the nearest point to the given position.\n\n"
             ":param position: Query position\n"
             ":param return_squared_l2: If true, return squared L2 distance, "
             "defaults to false\n"
             ":return: Tuple of (index, distance)")
        .def("nearest",
             py::overload_cast<int, bool>(&KdTree::nearest, py::const_),
             "index"_a, py::kw_only(), "return_squared_l2"_a = false,
             "Find the nearest point to the point at the given index.\n\n"
             ":param index: Index of the query point\n"
             ":param return_squared_l2: If true, return squared L2 distance, "
             "defaults to false\n"
             ":return: Tuple of (index, distance)")
        //
        .def(
            "nearest",
            py::overload_cast<const Eigen::Vector3d &, int, bool, bool>(
                &KdTree::nearest, py::const_),
            "position"_a, py::kw_only(),
            "k"_a,           //
            "sort"_a = true, //
            "return_squared_l2"_a = false,
            "Find k nearest points to the given position.\n\n"
            ":param position: Query position\n"
            ":param k: Number of nearest neighbors to find\n"
            ":param sort: If true, sort results by distance, defaults to true\n"
            ":param return_squared_l2: If true, return squared L2 distances, "
            "defaults to false\n"
            ":return: Tuple of (indices, distances)")
        .def(
            "nearest",
            py::overload_cast<const Eigen::Vector3d &, double, bool, bool>(
                &KdTree::nearest, py::const_),
            "position"_a, py::kw_only(),
            "radius"_a,      //
            "sort"_a = true, //
            "return_squared_l2"_a = false,
            "Find all points within a given radius of the query position.\n\n"
            ":param position: Query position\n"
            ":param radius: Search radius\n"
            ":param sort: If true, sort results by distance, defaults to true\n"
            ":param return_squared_l2: If true, return squared L2 distances, "
            "defaults to false\n"
            ":return: Tuple of (indices, distances)")
        //
        ;
}
} // namespace cubao
