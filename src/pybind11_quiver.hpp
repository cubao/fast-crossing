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
        .def(py::init<>(), "Default constructor for Arrow")
        .def(py::init<const Eigen::Vector3d &>(), "position"_a,
             "Constructor for Arrow with position")
        .def(py::init<const Eigen::Vector3d &, const Eigen::Vector3d &>(),
             "position"_a, "direction"_a,
             "Constructor for Arrow with position and direction")
        //
        .def("label", py::overload_cast<>(&Arrow::label, py::const_),
             "Get the label of the Arrow")
        .def("label", py::overload_cast<const Eigen::Vector2i &>(&Arrow::label),
             "new_value"_a, rvp::reference_internal,
             "Set the label of the Arrow")
        .def("label",
             py::overload_cast<int,                   //
                               int,                   //
                               std::optional<double>, //
                               std::optional<double>>(&Arrow::label),
             "polyline_index"_a, "segment_index"_a, //
             py::kw_only(),
             "t"_a = std::nullopt, //
             "range"_a = std::nullopt, rvp::reference_internal,
             "Set the label of the Arrow with polyline and segment indices")
        //
        .def("polyline_index",
             py::overload_cast<>(&Arrow::polyline_index, py::const_),
             "Get the polyline index of the Arrow")
        .def("polyline_index", py::overload_cast<int>(&Arrow::polyline_index),
             "new_value"_a, rvp::reference_internal,
             "Set the polyline index of the Arrow")
        .def("segment_index",
             py::overload_cast<>(&Arrow::segment_index, py::const_),
             "Get the segment index of the Arrow")
        .def("segment_index", py::overload_cast<int>(&Arrow::segment_index),
             "new_value"_a, rvp::reference_internal,
             "Set the segment index of the Arrow")
        //
        .def("t", py::overload_cast<>(&Arrow::t, py::const_),
             "Get the t parameter of the Arrow")
        .def("t", py::overload_cast<double>(&Arrow::t), "new_value"_a,
             rvp::reference_internal, "Set the t parameter of the Arrow")
        .def("range", py::overload_cast<>(&Arrow::range, py::const_),
             "Get the range of the Arrow")
        .def("range", py::overload_cast<double>(&Arrow::range), "new_value"_a,
             rvp::reference_internal, "Set the range of the Arrow")
        //
        .def("reset_index", py::overload_cast<>(&Arrow::reset_index),
             "Reset the index of the Arrow")
        .def("has_index",
             py::overload_cast<bool>(&Arrow::has_index, py::const_),
             "check_range"_a = true, "Check if the Arrow has a valid index")
        //
        .def("position", py::overload_cast<>(&Arrow::position, py::const_),
             "Get the position of the Arrow")
        .def("position",
             py::overload_cast<const Eigen::Vector3d &>(&Arrow::position),
             "new_value"_a, rvp::reference_internal,
             "Set the position of the Arrow")
        .def("direction", py::overload_cast<>(&Arrow::direction, py::const_),
             "Get the direction of the Arrow")
        .def("direction",
             py::overload_cast<const Eigen::Vector3d &>(&Arrow::direction),
             rvp::reference_internal, "Set the direction of the Arrow")
        .def("forward", py::overload_cast<>(&Arrow::direction, py::const_),
             "Get the forward direction of the Arrow")
        .def("leftward", py::overload_cast<>(&Arrow::leftward, py::const_),
             "Get the leftward direction of the Arrow")
        .def("upward", py::overload_cast<>(&Arrow::upward, py::const_),
             "Get the upward direction of the Arrow")
        .def("Frenet", py::overload_cast<>(&Arrow::Frenet, py::const_),
             "Get the Frenet frame of the Arrow")
        //
        .def("heading", py::overload_cast<>(&Arrow::heading, py::const_),
             "Get the heading of the Arrow")
        .def("heading", py::overload_cast<double>(&Arrow::heading),
             "new_value"_a, rvp::reference_internal,
             "Set the heading of the Arrow")
        .def_static("_heading", py::overload_cast<double>(&Arrow::_heading),
                    "heading"_a, "Convert heading to unit vector")
        .def_static(
            "_heading", py::overload_cast<double, double>(&Arrow::_heading),
            "east"_a, "north"_a, "Convert east and north components to heading")
        //
        .def_static("_unit_vector", &Arrow::_unit_vector, "vector"_a,
                    "with_eps"_a = true, "Normalize a vector to unit length")
        .def_static("_angle", &Arrow::_angle, "vec"_a, py::kw_only(), "ref"_a,
                    "Calculate angle between two vectors")
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
        .def(
            "copy", [](const Arrow &self) -> Arrow { return self; },
            "Create a copy of the Arrow")
        .def(
            "__copy__",
            [](const Arrow &self, py::dict) -> Arrow { return self; },
            "Create a copy of the Arrow")
        //
        ;

    using Quiver = cubao::Quiver;
    auto pyQuiver =
        py::class_<Quiver>(m, "Quiver", py::module_local())
            //
            .def(py::init<>(), "Default constructor for Quiver")
            .def(py::init<const Eigen::Vector3d &>(), "anchor_lla"_a,
                 "Constructor for Quiver with anchor LLA coordinates")
            //
            .def_static("_k", &Quiver::k, "Get the constant k")
            .def(
                "k", [](const Quiver &self) { return self.k_; },
                "Get the k value of the Quiver")
            .def(
                "inv_k", [](const Quiver &self) { return self.inv_k_; },
                "Get the inverse k value of the Quiver")
            .def(
                "anchor", [](const Quiver &self) { return self.anchor_; },
                "Get the anchor point of the Quiver")
            .def(
                "is_wgs84", [](const Quiver &self) { return self.is_wgs84_; },
                "Check if the Quiver is using WGS84 coordinates")
            //
            .def("forwards", &Quiver::forwards, "arrow"_a, "delta_x"_a,
                 "Move the Arrow forward by delta_x")
            .def("leftwards", &Quiver::leftwards, "arrow"_a, "delta_y"_a,
                 "Move the Arrow leftward by delta_y")
            .def("upwards", &Quiver::upwards, "arrow"_a, "delta_z"_a,
                 "Move the Arrow upward by delta_z")
            .def("towards", &Quiver::towards, "arrow"_a, "delta_frenet"_a,
                 py::kw_only(), "update_direction"_a = true,
                 "Move the Arrow in Frenet coordinates")
            //
            .def("update", &Quiver::update, "arrow"_a, "delta_enu"_a,
                 py::kw_only(), "update_direction"_a = true,
                 "Update the Arrow's position and optionally direction")
            //
            .def("enu2lla",
                 py::overload_cast<const Eigen::Vector3d &>(&Quiver::enu2lla,
                                                            py::const_),
                 "coords"_a, "Convert ENU coordinates to LLA")
            .def("enu2lla",
                 py::overload_cast<const RowVectors &>(&Quiver::enu2lla,
                                                       py::const_),
                 "coords"_a, "Convert multiple ENU coordinates to LLA")
            .def("lla2enu",
                 py::overload_cast<const Eigen::Vector3d &>(&Quiver::lla2enu,
                                                            py::const_),
                 "coords"_a, "Convert LLA coordinates to ENU")
            .def("lla2enu",
                 py::overload_cast<const RowVectors &>(&Quiver::lla2enu,
                                                       py::const_),
                 "coords"_a, "Convert multiple LLA coordinates to ENU")
        //
        ;

    // FilterParams
    using FilterParams = Quiver::FilterParams;
    py::class_<FilterParams>(pyQuiver, "FilterParams", py::module_local())
        .def(py::init<>(), "Default constructor for FilterParams")
        .def("x_slots", py::overload_cast<>(&FilterParams::x_slots, py::const_),
             "Get the x slots of the FilterParams")
        .def("x_slots",
             py::overload_cast<const std::optional<Eigen::VectorXd> &>(
                 &FilterParams::x_slots),
             rvp::reference_internal, "Set the x slots of the FilterParams")
        .def("y_slots", py::overload_cast<>(&FilterParams::y_slots, py::const_),
             "Get the y slots of the FilterParams")
        .def("y_slots",
             py::overload_cast<const std::optional<Eigen::VectorXd> &>(
                 &FilterParams::y_slots),
             rvp::reference_internal, "Set the y slots of the FilterParams")
        .def("z_slots", py::overload_cast<>(&FilterParams::z_slots, py::const_),
             "Get the z slots of the FilterParams")
        .def("z_slots",
             py::overload_cast<const std::optional<Eigen::VectorXd> &>(
                 &FilterParams::z_slots),
             rvp::reference_internal, "Set the z slots of the FilterParams")
        .def("angle_slots",
             py::overload_cast<>(&FilterParams::angle_slots, py::const_),
             "Get the angle slots of the FilterParams")
        .def("angle_slots",
             py::overload_cast<const std::optional<Eigen::VectorXd> &>(
                 &FilterParams::angle_slots),
             rvp::reference_internal, "Set the angle slots of the FilterParams")
        .def("is_trivial", &FilterParams::is_trivial,
             "Check if the FilterParams is trivial")
        //
        ;

    using KdQuiver = cubao::KdQuiver;
    py::class_<KdQuiver, Quiver>(m, "KdQuiver", py::module_local())
        .def(py::init<>(), "Default constructor for KdQuiver")
        .def(py::init<const Eigen::Vector3d &>(), "anchor_lla"_a,
             "Constructor for KdQuiver with anchor LLA coordinates")
        // add
        .def("add", py::overload_cast<const RowVectors &, int>(&KdQuiver::add),
             "polyline"_a, "index"_a = -1, "Add a polyline to the KdQuiver")
        .def("add",
             py::overload_cast<const Eigen::Ref<const RowVectorsNx2> &, int>(
                 &KdQuiver::add),
             "polyline"_a, "index"_a = -1, "Add a 2D polyline to the KdQuiver")
        // nearest
        .def("nearest",
             py::overload_cast<const Eigen::Vector3d &, bool>(
                 &KdQuiver::nearest, py::const_),
             "position"_a, py::kw_only(), //
             "return_squared_l2"_a = false,
             "Find the nearest point to the given position")
        .def("nearest",
             py::overload_cast<int, bool>(&KdQuiver::nearest, py::const_),
             "index"_a, py::kw_only(), //
             "return_squared_l2"_a = false,
             "Find the nearest point to the point at the given index")
        .def("nearest",
             py::overload_cast<const Eigen::Vector3d &, int, bool, bool>(
                 &KdQuiver::nearest, py::const_),
             "position"_a, py::kw_only(), //
             "k"_a,                       //
             "sort"_a = true,             //
             "return_squared_l2"_a = false,
             "Find k nearest points to the given position")
        .def("nearest",
             py::overload_cast<const Eigen::Vector3d &, double, bool, bool>(
                 &KdQuiver::nearest, py::const_),
             "position"_a, py::kw_only(), //
             "radius"_a,                  //
             "sort"_a = true,             //
             "return_squared_l2"_a = false,
             "Find all points within a given radius of the query position")
        // positions
        .def("positions", py::overload_cast<>(&KdQuiver::positions, py::const_),
             "Get all positions in the KdQuiver")
        .def("positions",
             py::overload_cast<const Eigen::VectorXi &>(&KdQuiver::positions,
                                                        py::const_),
             "indexes"_a, "Get positions for the given indexes")
        // directions
        .def("directions", &KdQuiver::directions, "indexes"_a,
             "Get directions for the given indexes")
        // arrows
        .def("arrows", &KdQuiver::arrows, "indexes"_a,
             "Get arrows for the given indexes")
        // arrow
        .def("arrow", py::overload_cast<int>(&KdQuiver::arrow, py::const_),
             "point_index"_a, "Get the arrow at the given point index")
        .def("arrow", py::overload_cast<int, int>(&KdQuiver::arrow, py::const_),
             "polyline_index"_a, "segment_index"_a,
             "Get the arrow at the given polyline and segment indices")
        .def("arrow",
             py::overload_cast<int, int, double>(&KdQuiver::arrow, py::const_),
             "polyline_index"_a, "segment_index"_a, py::kw_only(), "t"_a,
             "Get the arrow at the given polyline, segment indices, and t "
             "parameter")
        .def("arrow",
             py::overload_cast<int, double>(&KdQuiver::arrow, py::const_),
             "polyline_index"_a, py::kw_only(), "range"_a,
             "Get the arrow at the given polyline index and range")
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
                    "is_wgs84"_a = false,
                    "Filter arrows based on the given parameters")
        .def("filter",
             py::overload_cast<const Eigen::VectorXi &,     //
                               const Arrow &,               //
                               const Quiver::FilterParams & //
                               >(&KdQuiver::filter, py::const_),
             py::kw_only(), //
             "hits"_a,      //
             "arrow"_a,     //
             "params"_a, "Filter hits based on the given parameters")
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
             "params"_a, "Filter hits and norms based on the given parameters")
        //
        .def("reset", &KdQuiver::reset, "Reset the KdQuiver")
        .def("index", py::overload_cast<int>(&KdQuiver::index, py::const_),
             "point_index"_a, "Get the index for the given point index")
        .def("index", py::overload_cast<int, int>(&KdQuiver::index, py::const_),
             "polyline_index"_a, "segment_index"_a,
             "Get the index for the given polyline and segment indices");
}
} // namespace cubao
