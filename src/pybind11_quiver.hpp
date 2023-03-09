// should sync
// -
// https://github.com/cubao/fast-crossing/blob/master/src/pybind11_quiver.hpp
// -
// https://github.com/cubao/headers/tree/main/include/cubao/pybind11_quiver.hpp

#pragma once

#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "cubao_inline.hpp"
#include "quiver.hpp"

namespace cubao
{
namespace py = pybind11;
using namespace pybind11::literals;
using rvp = py::return_value_policy;

CUBAO_INLINE void bind_quiver(py::module &m)
{
    using Arrow = cubao::Arrow;
    py::class_<Arrow>(m, "Arrow", py::module_local())
        //
        .def(py::init<>())
        .def(py::init<const Eigen::Vector3d &, const Eigen::Vector3d &>(),
             "position"_a, //
             "direction"_a = Eigen::Vector3d(0.0, 0.0, 1.0))
        //
        .def("label", py::overload_cast<>(&Arrow::label, py::const_))
        .def("label",
             py::overload_cast<int,                   //
                               int,                   //
                               std::optional<double>, //
                               std::optional<double>>(&Arrow::label),
             "polyline_index"_a, "segment_index"_a, //
             py::kw_only(),
             "t"_a = std::nullopt, //
             "range"_a = std::nullopt)
        //
        ;

    using Quiver = cubao::Quiver;
    py::class_<Quiver>(m, "Quiver", py::module_local())
        //
        ;
}
} // namespace cubao
