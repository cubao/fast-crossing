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
        .def(py::init<bool>(), py::kw_only(), "is_wgs84"_a = false,
             "Initialize FastCrossing object.\n\n"
             ":param is_wgs84: Whether coordinates are in WGS84 format, "
             "defaults to false")

        // add polyline
        .def("add_polyline",
             py::overload_cast<const FastCrossing::PolylineType &, int>(
                 &FastCrossing::add_polyline),
             "polyline"_a, py::kw_only(), "index"_a = -1,
             "Add polyline to the tree.\n\n"
             ":param polyline: The polyline to add\n"
             ":param index: Custom polyline index, defaults to -1\n"
             ":return: The index of the added polyline")

        .def("add_polyline",
             py::overload_cast<
                 const Eigen::Ref<const FastCrossing::FlatBush::PolylineType> &,
                 int>(&FastCrossing::add_polyline),
             "polyline"_a, py::kw_only(), "index"_a = -1,
             "Add polyline to the tree (alternative format).\n\n"
             ":param polyline: The polyline to add\n"
             ":param index: Custom polyline index, defaults to -1\n"
             ":return: The index of the added polyline")

        .def("finish", &FastCrossing::finish,
             "Finalize indexing after adding all polylines")

        // intersections
        .def("intersections",
             py::overload_cast<>(&FastCrossing::intersections, py::const_),
             "Get all segment intersections in the tree")

        .def(
            "intersections",
            py::overload_cast<const std::tuple<double, double> &, int>(
                &FastCrossing::intersections, py::const_),
            py::kw_only(), "z_offset_range"_a, "self_intersection"_a = 2,
            "Get segment intersections in the tree, filtered by conditions.\n\n"
            ":param z_offset_range: Z-offset range for filtering\n"
            ":param self_intersection: Self-intersection parameter, defaults "
            "to 2")

        .def("intersections",
             py::overload_cast<const Eigen::Vector2d &, const Eigen::Vector2d &,
                               bool>(&FastCrossing::intersections, py::const_),
             "start"_a, "to"_a, py::kw_only(), "dedup"_a = true,
             "Get crossing intersections with [start, to] segment.\n\n"
             ":param from: Start point of the segment\n"
             ":param to: End point of the segment\n"
             ":param dedup: Whether to remove duplicates, defaults to true\n"
             ":return: Sorted intersections by t ratio")

        .def("intersections",
             py::overload_cast<const FastCrossing::PolylineType &, bool>(
                 &FastCrossing::intersections, py::const_),
             "polyline"_a, py::kw_only(), "dedup"_a = true,
             "Get crossing intersections with a polyline.\n\n"
             ":param polyline: The polyline to check intersections with\n"
             ":param dedup: Whether to remove duplicates, defaults to true\n"
             ":return: Sorted intersections by t ratio")

        .def("intersections",
             py::overload_cast<
                 const Eigen::Ref<const FastCrossing::FlatBush::PolylineType> &,
                 bool>(&FastCrossing::intersections, py::const_),
             "polyline"_a, py::kw_only(), "dedup"_a = true,
             "Get crossing intersections with a polyline (alternative "
             "format).\n\n"
             ":param polyline: The polyline to check intersections with\n"
             ":param dedup: Whether to remove duplicates, defaults to true\n"
             ":return: Sorted intersections by t ratio")

        .def("intersections",
             py::overload_cast<const FastCrossing::PolylineType &, double,
                               double, bool>(&FastCrossing::intersections,
                                             py::const_),
             "polyline"_a, py::kw_only(), "z_min"_a, "z_max"_a,
             "dedup"_a = true,
             "Get crossing intersections with a polyline, filtered by Z "
             "range.\n\n"
             ":param polyline: The polyline to check intersections with\n"
             ":param z_min: Minimum Z value for filtering\n"
             ":param z_max: Maximum Z value for filtering\n"
             ":param dedup: Whether to remove duplicates, defaults to true\n"
             ":return: Sorted intersections by t ratio")

        .def(
            "intersections",
            py::overload_cast<
                const Eigen::Ref<const FastCrossing::FlatBush::PolylineType> &,
                double, double, bool>(&FastCrossing::intersections, py::const_),
            "polyline"_a, py::kw_only(), "z_min"_a, "z_max"_a, "dedup"_a = true,
            "Get crossing intersections with a polyline, filtered by Z range "
            "(alternative format).\n\n"
            ":param polyline: The polyline to check intersections with\n"
            ":param z_min: Minimum Z value for filtering\n"
            ":param z_max: Maximum Z value for filtering\n"
            ":param dedup: Whether to remove duplicates, defaults to true\n"
            ":return: Sorted intersections by t ratio")

        // segment_index
        .def("segment_index",
             py::overload_cast<int>(&FastCrossing::segment_index, py::const_),
             "index"_a,
             "Get segment index for a given index.\n\n"
             ":param index: The index to query\n"
             ":return: The segment index")

        .def("segment_index",
             py::overload_cast<const Eigen::VectorXi &>(
                 &FastCrossing::segment_index, py::const_),
             "indexes"_a,
             "Get segment indexes for given indexes.\n\n"
             ":param indexes: The indexes to query\n"
             ":return: The segment indexes")

        // point_index
        .def("point_index",
             py::overload_cast<int>(&FastCrossing::point_index, py::const_),
             "index"_a,
             "Get point index for a given index.\n\n"
             ":param index: The index to query\n"
             ":return: The point index")

        .def("point_index",
             py::overload_cast<const Eigen::VectorXi &>(
                 &FastCrossing::point_index, py::const_),
             "indexes"_a,
             "Get point indexes for given indexes.\n\n"
             ":param indexes: The indexes to query\n"
             ":return: The point indexes")

        // within
        .def("within",
             py::overload_cast<const Eigen::Vector2d &, const Eigen::Vector2d &,
                               bool, bool>(&FastCrossing::within, py::const_),
             py::kw_only(),           //
             "min"_a,                 //
             "max"_a,                 //
             "segment_wise"_a = true, //
             "sort"_a = true,
             "Find polylines within a bounding box.\n\n"
             ":param min: Minimum corner of the bounding box\n"
             ":param max: Maximum corner of the bounding box\n"
             ":param segment_wise: Whether to return segment-wise results, "
             "defaults to true\n"
             ":param sort: Whether to sort the results, defaults to true\n"
             ":return: Polylines within the bounding box")

        .def("within",
             py::overload_cast<const Eigen::Ref<const RowVectorsNx2> &, bool,
                               bool>(&FastCrossing::within, py::const_),
             py::kw_only(),           //
             "polygon"_a,             //
             "segment_wise"_a = true, //
             "sort"_a = true,
             "Find polylines within a polygon.\n\n"
             ":param polygon: The polygon to check against\n"
             ":param segment_wise: Whether to return segment-wise results, "
             "defaults to true\n"
             ":param sort: Whether to sort the results, defaults to true\n"
             ":return: Polylines within the polygon")

        .def("within",
             py::overload_cast<const Eigen::Vector2d &, double, double, double,
                               bool, bool>(&FastCrossing::within, py::const_),
             py::kw_only(),           //
             "center"_a,              //
             "width"_a,               //
             "height"_a,              //
             "heading"_a = 0.0,       //
             "segment_wise"_a = true, //
             "sort"_a = true,
             "Find polylines within a rotated rectangle.\n\n"
             ":param center: Center of the rectangle\n"
             ":param width: Width of the rectangle\n"
             ":param height: Height of the rectangle\n"
             ":param heading: Heading angle of the rectangle, defaults to 0.0\n"
             ":param segment_wise: Whether to return segment-wise results, "
             "defaults to true\n"
             ":param sort: Whether to sort the results, defaults to true\n"
             ":return: Polylines within the rotated rectangle")

        // nearest
        .def("nearest",
             py::overload_cast<const Eigen::Vector3d &, bool>(
                 &FastCrossing::nearest, py::const_),
             "position"_a, py::kw_only(), //
             "return_squared_l2"_a = false,
             "Find the nearest point to a given position.\n\n"
             ":param position: The query position\n"
             ":param return_squared_l2: Whether to return squared L2 distance, "
             "defaults to false\n"
             ":return: Nearest point information")

        .def("nearest",
             py::overload_cast<const Eigen::Vector2i &, bool>(
                 &FastCrossing::nearest, py::const_),
             "index"_a, py::kw_only(), //
             "return_squared_l2"_a = false,
             "Find the nearest point to a given index.\n\n"
             ":param index: The query index\n"
             ":param return_squared_l2: Whether to return squared L2 distance, "
             "defaults to false\n"
             ":return: Nearest point information")

        .def("nearest",
             py::overload_cast<const Eigen::Vector3d &, //
                               std::optional<int>,      // k
                               std::optional<double>,   // radius
                               bool,                    // sort
                               bool,                    // return_squared_l2
                               std::optional<std::pair<Eigen::Vector3d,
                                                       Quiver::FilterParams>>>(
                 &FastCrossing::nearest, py::const_),
             "position"_a,                  //
             py::kw_only(),                 //
             "k"_a = std::nullopt,          //
             "radius"_a = std::nullopt,     //
             "sort"_a = true,               //
             "return_squared_l2"_a = false, //
             "filter"_a = std::nullopt,
             "Find k nearest points to a given position with optional "
             "filtering.\n\n"
             ":param position: The query position\n"
             ":param k: Number of nearest neighbors to find (optional)\n"
             ":param radius: Search radius (optional)\n"
             ":param sort: Whether to sort the results, defaults to true\n"
             ":param return_squared_l2: Whether to return squared L2 distance, "
             "defaults to false\n"
             ":param filter: Optional filter parameters\n"
             ":return: Nearest points information")

        // coordinates
        .def("coordinates",
             py::overload_cast<int, int, double>(&FastCrossing::coordinates,
                                                 py::const_),
             "polyline_index"_a, "segment_index"_a, "ratio"_a,
             "Get coordinates at a specific position on a polyline.\n\n"
             ":param polyline_index: Index of the polyline\n"
             ":param segment_index: Index of the segment within the polyline\n"
             ":param ratio: Ratio along the segment (0 to 1)\n"
             ":return: Coordinates at the specified position")

        .def("coordinates",
             py::overload_cast<const Eigen::Vector2i &, double>(
                 &FastCrossing::coordinates, py::const_),
             "index"_a, "ratio"_a,
             "Get coordinates at a specific position on a polyline "
             "(alternative format).\n\n"
             ":param index: Combined index of polyline and segment\n"
             ":param ratio: Ratio along the segment (0 to 1)\n"
             ":return: Coordinates at the specified position")

        .def("coordinates",
             py::overload_cast<const FastCrossing::IntersectionType &, bool>(
                 &FastCrossing::coordinates, py::const_),
             "intersection"_a, "second"_a = true,
             "Get coordinates of an intersection.\n\n"
             ":param intersection: The intersection object\n"
             ":param second: Whether to use the second polyline of the "
             "intersection, defaults to true\n"
             ":return: Coordinates of the intersection")

        // arrow
        .def("arrow",
             py::overload_cast<int, int>(&FastCrossing::arrow, py::const_),
             py::kw_only(), "polyline_index"_a, "point_index"_a,
             "Get an arrow (position and direction) at a specific point on a "
             "polyline.\n\n"
             ":param polyline_index: Index of the polyline\n"
             ":param point_index: Index of the point within the polyline\n"
             ":return: Arrow (position and direction)")

        //
        .def(
            "is_wgs84", &FastCrossing::is_wgs84,
            "Check if the coordinates are in WGS84 format.\n\n"
            ":return: True if coordinates are in WGS84 format, False otherwise")

        .def("num_poylines", &FastCrossing::num_poylines,
             "Get the number of polylines in the FastCrossing object.\n\n"
             ":return: Number of polylines")

        .def("polyline_rulers", &FastCrossing::polyline_rulers,
             rvp::reference_internal,
             "Get all polyline rulers.\n\n"
             ":return: Dictionary of polyline rulers")

        .def("polyline_ruler", &FastCrossing::polyline_ruler, "index"_a,
             rvp::reference_internal,
             "Get a specific polyline ruler.\n\n"
             ":param index: Index of the polyline\n"
             ":return: Polyline ruler for the specified index")

        // export
        .def("bush", &FastCrossing::export_bush, "autobuild"_a = true,
             rvp::reference_internal,
             "Export the internal FlatBush index.\n\n"
             ":param autobuild: Whether to automatically build the index if "
             "not already built, defaults to true\n"
             ":return: FlatBush index")

        .def("quiver", &FastCrossing::export_quiver, rvp::reference_internal,
             "Export the internal Quiver object.\n\n"
             ":return: Quiver object");
}
} // namespace cubao
