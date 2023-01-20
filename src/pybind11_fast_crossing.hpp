// should sync
// -
// https://github.com/cubao/fast-crossing/blob/master/src/pybind11_fast_crossing.hpp
// -
// https://github.com/cubao/headers/tree/main/include/cubao/pybind11_fast_crossing.hpp

#pragma once

#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "cubao_inline.hpp"
#include "fast_crossing.hpp"

namespace cubao
{
namespace py = pybind11;
using namespace pybind11::literals;
using rvp = py::return_value_policy;

CUBAO_INLINE void bind_fast_crossing(py::module &m)
{
    py::class_<FastCrossing>(m, "FastCrossing", py::module_local())
        .def(py::init<bool, bool>(), py::kw_only(), "is_wgs84"_a = false,
             "use_polyline_rulers"_a = true)
        .def("add_polyline",

             py::overload_cast<const FastCrossing::FlatBush::PolylineType &,
                               int>(&FastCrossing::add_polyline),
             "polyline"_a, py::kw_only(), "index"_a = -1,
             "add polyline to tree, you can "
             "provide your own polyline index, "
             "default: -1")
        .def("finish", &FastCrossing::finish, "finish to finalize indexing")
        .def("intersections",
             py::overload_cast<>(&FastCrossing::intersections, py::const_),
             "all segment intersections in tree")
        .def("intersections",
             py::overload_cast<const Eigen::Vector2d &, const Eigen::Vector2d &,
                               bool>(&FastCrossing::intersections, py::const_),
             "from"_a, "to"_a, py::kw_only(), "dedup"_a = true,
             "crossing intersections with [from, to] segment "
             "(sorted by t ratio)")
        .def("intersections",
             py::overload_cast<
                 const Eigen::Ref<const FastCrossing::PolylineType> &, bool>(
                 &FastCrossing::intersections, py::const_),
             "polyline"_a, py::kw_only(), "dedup"_a = true,
             "crossing intersections with polyline (sorted by t ratio)")
        .def("intersections",
             py::overload_cast<const FastCrossing::FlatBush::PolylineType &>(
                 &FastCrossing::intersections, py::const_),
             "polyline"_a,
             "crossing intersections with polyline (sorted by t ratio)")
        //
        .def("bush", &FastCrossing::bush, rvp::reference_internal)
        .def("num_poylines", &FastCrossing::num_poylines)
        //
        ;
}
} // namespace cubao