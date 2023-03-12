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
        .def(py::init<int>(), "leafsize"_a = 10)
        .def(py::init<const RowVectors &>(), "points"_a)
        .def(py::init<const Eigen::Ref<const RowVectorsNx2> &>(), "points"_a)
        //
        .def("points", &KdTree::points, rvp::reference_internal)
        //
        .def("add", py::overload_cast<const RowVectors &>(&KdTree::add),
             "points"_a)
        .def("add",
             py::overload_cast<const Eigen::Ref<const RowVectorsNx2> &>(
                 &KdTree::add),
             "points"_a)
        //
        .def("reset", &KdTree::reset)
        .def("reset_index", &KdTree::reset_index)
        .def("build_index", &KdTree::build_index, "force_rebuild"_a = false)
        //
        .def("leafsize", &KdTree::leafsize)
        .def("set_leafsize", &KdTree::set_leafsize, "value"_a)
        //
        .def("nearest",
             py::overload_cast<const Eigen::Vector3d &, bool>(&KdTree::nearest,
                                                              py::const_),
             "position"_a, py::kw_only(), "return_squared_l2"_a = false)
        .def("nearest",
             py::overload_cast<int, bool>(&KdTree::nearest, py::const_),
             "index"_a, py::kw_only(), "return_squared_l2"_a = false)
        //
        .def("nearest",
             py::overload_cast<const Eigen::Vector3d &, int, bool, bool>(
                 &KdTree::nearest, py::const_),
             "position"_a, py::kw_only(),
             "k"_a,           //
             "sort"_a = true, //
             "return_squared_l2"_a = false)
        .def("nearest",
             py::overload_cast<const Eigen::Vector3d &, double, bool, bool>(
                 &KdTree::nearest, py::const_),
             "position"_a, py::kw_only(),
             "radius"_a,      //
             "sort"_a = true, //
             "return_squared_l2"_a = false)
        //
        ;
}
} // namespace cubao
