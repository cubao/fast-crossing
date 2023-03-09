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

#include <spdlog/spdlog.h>

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
        .def("label", py::overload_cast<const Eigen::Vector2i &>(&Arrow::label),
             "new_value"_a, rvp::reference_internal)
        .def("label",
             py::overload_cast<int,                   //
                               int,                   //
                               std::optional<double>, //
                               std::optional<double>>(&Arrow::label),
             "polyline_index"_a, "segment_index"_a, //
             py::kw_only(),
             "t"_a = std::nullopt, //
             "range"_a = std::nullopt, rvp::reference_internal)
        //
        .def("polyline_index",
             py::overload_cast<>(&Arrow::polyline_index, py::const_))
        .def("polyline_index", py::overload_cast<int>(&Arrow::polyline_index),
             "new_value"_a, rvp::reference_internal)
        .def("segment_index",
             py::overload_cast<>(&Arrow::segment_index, py::const_))
        .def("segment_index", py::overload_cast<int>(&Arrow::segment_index),
             "new_value"_a, rvp::reference_internal)
        //
        .def("t", py::overload_cast<>(&Arrow::t, py::const_))
        .def("t", py::overload_cast<double>(&Arrow::t), "new_value"_a,
             rvp::reference_internal)
        .def("range", py::overload_cast<>(&Arrow::range, py::const_))
        .def("range", py::overload_cast<double>(&Arrow::range), "new_value"_a,
             rvp::reference_internal)
        //
        .def("reset_index", py::overload_cast<>(&Arrow::reset_index))
        .def("has_index",
             py::overload_cast<bool>(&Arrow::has_index, py::const_),
             "check_range"_a = true)
        //
        .def("position", py::overload_cast<>(&Arrow::position, py::const_))
        .def("position",
             py::overload_cast<const Eigen::Vector3d &>(&Arrow::position),
             "new_value"_a, rvp::reference_internal)
        .def("direction", py::overload_cast<>(&Arrow::direction, py::const_))
        .def(
            "direction",
            py::overload_cast<const Eigen::Vector3d &, bool>(&Arrow::direction),
            "new_value"_a, "need_normalize"_a = false, rvp::reference_internal)
        .def("leftward", py::overload_cast<>(&Arrow::leftward, py::const_))
        //
        .def("heading", py::overload_cast<>(&Arrow::heading, py::const_))
        .def("heading", py::overload_cast<double>(&Arrow::heading),
             "new_value"_a)
        //
        .def("__repr__",
             [](const Arrow &self) {
                 return fmt::format("label:({}/{}/{:.3f}),range:{},"
                                    "xyz:({},{},{}),"
                                    "dir:({:.3f},{:.3f},{:.1f}),heading:{:.2f}",
                                    self.polyline_index_, self.segment_index_,
                                    self.t_, self.range_, self.position_[0],
                                    self.position_[1], self.position_[2],
                                    self.direction_[0], self.direction_[1],
                                    self.direction_[2], self.heading());
             })
        .def("copy", [](const Arrow &self) -> Arrow { return self; })
        .def("__copy__",
             [](const Arrow &self, py::dict) -> Arrow { return self; })
        //
        ;

    using Quiver = cubao::Quiver;
    py::class_<Quiver>(m, "Quiver", py::module_local())
        //
        .def(py::init<>())
        .def(py::init<const Eigen::Vector3d &>(), "anchor_lla"_a)
        //
        .def_static("_k", &Quiver::k)
        .def("k", [](const Quiver &self) { return self.k_; })
        .def("inv_k", [](const Quiver &self) { return self.inv_k_; })
        .def("anchor", [](const Quiver &self) { return self.anchor_; })
        .def("is_wgs84", [](const Quiver &self) { return self.is_wgs84_; })
        //
        .def("forward", &Quiver::forward, "arrow"_a, "delta"_a)
        .def("leftward", &Quiver::leftward, "arrow"_a, "delta"_a)
        .def("upward", &Quiver::upward, "arrow"_a, "delta"_a)
        .def("update", &Quiver::update, "arrow"_a, "delta"_a, py::kw_only(),
             "keep_direction"_a = false)
        //
        //
        ;
}
} // namespace cubao
