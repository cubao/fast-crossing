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
#include "kd_quiver.hpp"

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
        .def(py::init<const Eigen::Vector3d &>(), "position"_a)
        .def(py::init<const Eigen::Vector3d &, const Eigen::Vector3d &>(),
             "position"_a, "direction"_a)
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
        .def("direction",
             py::overload_cast<const Eigen::Vector3d &>(&Arrow::direction),
             rvp::reference_internal)
        .def("forward", py::overload_cast<>(&Arrow::direction, py::const_))
        .def("leftward", py::overload_cast<>(&Arrow::leftward, py::const_))
        .def("upward", py::overload_cast<>(&Arrow::upward, py::const_))
        .def("Frenet", py::overload_cast<>(&Arrow::Frenet, py::const_))
        //
        .def("heading", py::overload_cast<>(&Arrow::heading, py::const_))
        .def("heading", py::overload_cast<double>(&Arrow::heading),
             "new_value"_a, rvp::reference_internal)
        .def_static("_heading", py::overload_cast<double>(&Arrow::_heading),
                    "heading"_a)
        .def_static("_heading",
                    py::overload_cast<double, double>(&Arrow::_heading),
                    "east"_a, "north"_a)
        //
        .def_static("_unit_vector", &Arrow::_unit_vector, "vector"_a,
                    "with_eps"_a = true)
        .def_static("_angle", &Arrow::_angle, "vec"_a, py::kw_only(), "ref"_a)
        //
        .def("__repr__",
             [](const Arrow &self) {
                 auto &pos = self.position();
                 auto &dir = self.direction();
                 return fmt::format(
                     "Arrow(label:({}/{}/{:.3f}),range:{},"
                     "xyz:({},{},{}),"
                     "dir:({:.3f},{:.3f},{:.1f}),heading:{:.2f})",
                     self.polyline_index_, self.segment_index_, //
                     self.t_,
                     self.range_,            //
                     pos[0], pos[1], pos[2], //
                     dir[0], dir[1], dir[2], //
                     self.heading());
             })
        .def("copy", [](const Arrow &self) -> Arrow { return self; })
        .def("__copy__",
             [](const Arrow &self, py::dict) -> Arrow { return self; })
        //
        ;

    using Quiver = cubao::Quiver;
    auto pyQuiver =
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
            .def("forwards", &Quiver::forwards, "arrow"_a, "delta_x"_a)
            .def("leftwards", &Quiver::leftwards, "arrow"_a, "delta_y"_a)
            .def("upwards", &Quiver::upwards, "arrow"_a, "delta_z"_a)
            .def("towards", &Quiver::towards, "arrow"_a, "delta_frenet"_a,
                 py::kw_only(), "update_direction"_a = true)
            //
            .def("update", &Quiver::update, "arrow"_a, "delta_enu"_a,
                 py::kw_only(), "update_direction"_a = true)
            //
            .def("enu2lla",
                 py::overload_cast<const Eigen::Vector3d &>(&Quiver::enu2lla,
                                                            py::const_),
                 "coords"_a)
            .def("enu2lla",
                 py::overload_cast<const RowVectors &>(&Quiver::enu2lla,
                                                       py::const_),
                 "coords"_a)
            .def("lla2enu",
                 py::overload_cast<const Eigen::Vector3d &>(&Quiver::lla2enu,
                                                            py::const_),
                 "coords"_a)
            .def("lla2enu",
                 py::overload_cast<const RowVectors &>(&Quiver::lla2enu,
                                                       py::const_),
                 "coords"_a)
        //
        ;

    // FilterParams
    using FilterParams = Quiver::FilterParams;
    py::class_<FilterParams>(pyQuiver, "FilterParams", py::module_local())
        .def(py::init<>())
        .def("x_slots", py::overload_cast<>(&FilterParams::x_slots, py::const_))
        .def("x_slots",
             py::overload_cast<const std::optional<Eigen::VectorXd> &>(
                 &FilterParams::x_slots),
             rvp::reference_internal)
        .def("y_slots", py::overload_cast<>(&FilterParams::y_slots, py::const_))
        .def("y_slots",
             py::overload_cast<const std::optional<Eigen::VectorXd> &>(
                 &FilterParams::y_slots),
             rvp::reference_internal)
        .def("z_slots", py::overload_cast<>(&FilterParams::z_slots, py::const_))
        .def("z_slots",
             py::overload_cast<const std::optional<Eigen::VectorXd> &>(
                 &FilterParams::z_slots),
             rvp::reference_internal)
        .def("angle_slots",
             py::overload_cast<>(&FilterParams::angle_slots, py::const_))
        .def("angle_slots",
             py::overload_cast<const std::optional<Eigen::VectorXd> &>(
                 &FilterParams::angle_slots),
             rvp::reference_internal)
        .def("is_trivial", &FilterParams::is_trivial)
        //
        ;

    using KdQuiver = cubao::KdQuiver;
    py::class_<KdQuiver, Quiver>(m, "KdQuiver", py::module_local())
        .def(py::init<>())
        .def(py::init<const Eigen::Vector3d &>(), "anchor_lla"_a)
        // add
        .def("add", py::overload_cast<const RowVectors &, int>(&KdQuiver::add),
             "polyline"_a, "index"_a = -1)
        .def("add",
             py::overload_cast<const Eigen::Ref<const RowVectorsNx2> &, int>(
                 &KdQuiver::add),
             "polyline"_a, "index"_a = -1)
        // nearest
        .def("nearest",
             py::overload_cast<const Eigen::Vector3d &, bool>(
                 &KdQuiver::nearest, py::const_),
             "position"_a, py::kw_only(), //
             "return_squared_l2"_a = false)
        .def("nearest",
             py::overload_cast<int, bool>(&KdQuiver::nearest, py::const_),
             "index"_a, py::kw_only(), //
             "return_squared_l2"_a = false)
        .def("nearest",
             py::overload_cast<const Eigen::Vector3d &, int, bool, bool>(
                 &KdQuiver::nearest, py::const_),
             "position"_a, py::kw_only(), //
             "k"_a,                       //
             "sort"_a = true,             //
             "return_squared_l2"_a = false)
        .def("nearest",
             py::overload_cast<const Eigen::Vector3d &, double, bool, bool>(
                 &KdQuiver::nearest, py::const_),
             "position"_a, py::kw_only(), //
             "radius"_a,                  //
             "sort"_a = true,             //
             "return_squared_l2"_a = false)
        // positions
        .def("positions", py::overload_cast<>(&KdQuiver::positions, py::const_))
        .def("positions",
             py::overload_cast<const Eigen::VectorXi &>(&KdQuiver::positions,
                                                        py::const_),
             "indexes"_a)
        // directions
        .def("directions", &KdQuiver::directions, "indexes"_a)
        // arrows
        .def("arrows", &KdQuiver::arrows, "indexes"_a)
        // arrow
        .def("arrow", py::overload_cast<int>(&KdQuiver::arrow, py::const_),
             "point_index"_a)
        .def("arrow", py::overload_cast<int, int>(&KdQuiver::arrow, py::const_),
             "polyline_index"_a, "segment_index"_a)
        .def("arrow",
             py::overload_cast<int, int, double>(&KdQuiver::arrow, py::const_),
             "polyline_index"_a, "segment_index"_a, py::kw_only(), "t"_a)
        .def("arrow",
             py::overload_cast<int, double>(&KdQuiver::arrow, py::const_),
             "polyline_index"_a, py::kw_only(), "range"_a)
        // filter
        .def_static("_filter",
                    py::overload_cast<const std::vector<Arrow> &,   //
                                      const Arrow &,                //
                                      const Quiver::FilterParams &, //
                                      bool                          //
                                      >(&KdQuiver::filter),
                    py::kw_only(), //
                    "arrows"_a,    //
                    "arrow"_a,     //
                    "params"_a,    //
                    "is_wgs84"_a = false)
        .def("filter",
             py::overload_cast<const Eigen::VectorXi &,     //
                               const Arrow &,               //
                               const Quiver::FilterParams & //
                               >(&KdQuiver::filter, py::const_),
             py::kw_only(), //
             "hits"_a,      //
             "arrow"_a,     //
             "params"_a)
        .def("filter",
             py::overload_cast<const Eigen::VectorXi &,     //
                               const Eigen::VectorXd &,     //
                               const Arrow &,               //
                               const Quiver::FilterParams & //
                               >(&KdQuiver::filter, py::const_),
             py::kw_only(), //
             "hits"_a,      //
             "norms"_a,     //
             "arrow"_a,     //
             "params"_a)
        //
        .def("reset", &KdQuiver::reset)
        .def("index", py::overload_cast<int>(&KdQuiver::index, py::const_),
             "point_index"_a)
        .def("index", py::overload_cast<int, int>(&KdQuiver::index, py::const_),
             "polyline_index"_a, "segment_index"_a);
}
} // namespace cubao
