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
        .def(py::init<bool>(), py::kw_only(), "is_wgs84"_a = false)

        // add polyline
        .def("add_polyline",

             py::overload_cast<const FastCrossing::PolylineType &, int>(
                 &FastCrossing::add_polyline),
             "polyline"_a, py::kw_only(), "index"_a = -1,
             "add polyline to tree, you can "
             "provide your own polyline index, "
             "default: -1")

        .def("add_polyline",

             py::overload_cast<
                 const Eigen::Ref<const FastCrossing::FlatBush::PolylineType> &,
                 int>(&FastCrossing::add_polyline),
             "polyline"_a, py::kw_only(), "index"_a = -1,
             "add polyline to tree, you can "
             "provide your own polyline index, "
             "default: -1")
        // finish

        .def("finish", &FastCrossing::finish, "finish to finalize indexing")
        // intersections
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
             py::overload_cast<const FastCrossing::PolylineType &, bool>(
                 &FastCrossing::intersections, py::const_),
             "polyline"_a, py::kw_only(), "dedup"_a = true,
             "crossing intersections with polyline (sorted by t ratio)")
        .def("intersections",
             py::overload_cast<
                 const Eigen::Ref<const FastCrossing::FlatBush::PolylineType> &,
                 bool>(&FastCrossing::intersections, py::const_),
             "polyline"_a, py::kw_only(), "dedup"_a = true,
             "crossing intersections with polyline (sorted by t ratio)")
        .def("intersections",
             py::overload_cast<const FastCrossing::PolylineType &, double,
                               double, bool>(&FastCrossing::intersections,
                                             py::const_),
             "polyline"_a, py::kw_only(), "z_min"_a, "z_max"_a,
             "dedup"_a = true,
             "crossing intersections with polyline (sorted by t ratio)")
        .def(
            "intersections",
            py::overload_cast<
                const Eigen::Ref<const FastCrossing::FlatBush::PolylineType> &,
                double, double, bool>(&FastCrossing::intersections, py::const_),
            "polyline"_a, py::kw_only(), "z_min"_a, "z_max"_a, "dedup"_a = true,
            "crossing intersections with polyline (sorted by t ratio)")
        // coordinates
        .def("coordinates",
             py::overload_cast<int, int, double>(&FastCrossing::coordinates,
                                                 py::const_),
             "polyline_index"_a, "segment_index"_a, "ratio"_a)
        .def("coordinates",
             py::overload_cast<const Eigen::Vector2i &, double>(
                 &FastCrossing::coordinates, py::const_),
             "index"_a, "ratio"_a)
        .def("coordinates",
             py::overload_cast<const FastCrossing::IntersectionType &, bool>(
                 &FastCrossing::coordinates, py::const_),
             "intersection"_a, "second"_a = true)
        //
        .def("bush", &FastCrossing::bush, rvp::reference_internal)
        .def("is_wgs84", &FastCrossing::is_wgs84)
        .def("num_poylines", &FastCrossing::num_poylines)
        .def("polyline_rulers", &FastCrossing::polyline_rulers)
        .def("polyline_ruler", &FastCrossing::polyline_ruler, "index"_a,
             rvp::reference_internal)
        //
        ;
}
} // namespace cubao
