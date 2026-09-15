// https://github.com/microsoft/vscode-cpptools/issues/9692
#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif
#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <Eigen/Core>

#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>

#include "fast_crossing.hpp"
#include "polyline_in_polygon.hpp"
#include "pybind11_fast_crossing.hpp"
#include "pybind11_flatbush.hpp"
#include "pybind11_nanoflann_kdtree.hpp"
#include "pybind11_quiver.hpp"

#define CUBAO_ARGV_DEFAULT_NONE(argv) py::arg_v(#argv, std::nullopt, "None")

#include "pybind11_polyline_ruler.hpp"
#include "pybind11_crs_transform.hpp"

#include "point_in_polygon.hpp"
#include "densify_polyline.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(_core, m)
{
    cubao::bind_fast_crossing(m);
    cubao::bind_flatbush(m);
    cubao::bind_nanoflann_kdtree(m);
    cubao::bind_quiver(m);
    cubao::bind_polyline_ruler(m);

    auto tf = m.def_submodule("tf");
    cubao::bind_crs_transform(tf);

    m.def("point_in_polygon", &cubao::point_in_polygon, //
          py::kw_only(), "points"_a, "polygon"_a,
          "point-in-polygon test, returns 0-1 mask");
    m.def("densify_polyline", &cubao::densify_polyline, //
          "polyline"_a, py::kw_only(), "max_gap"_a,
          "densify polyline, interpolate to satisfy max_gap");
    m.def("polyline_in_polygon",
          py::overload_cast<const cubao::RowVectors &, //
                            const Eigen::Ref<const cubao::RowVectorsNx2> &,
                            const cubao::FastCrossing &>(
              &cubao::polyline_in_polygon), //
          "polyline"_a, "polygon"_a, py::kw_only(), "fc"_a);
    m.def("polyline_in_polygon",
          py::overload_cast<const cubao::RowVectors &, //
                            const Eigen::Ref<const cubao::RowVectorsNx2> &,
                            bool>(&cubao::polyline_in_polygon), //
          "polyline"_a, "polygon"_a, py::kw_only(), "is_wgs84"_a = false);
    m.def("polyline_chunks_in_polygon",
          py::overload_cast<const cubao::RowVectors &,
                            const Eigen::Ref<const cubao::RowVectorsNx2> &,
                            const cubao::FastCrossing &>(
              &cubao::polyline_chunks_in_polygon),
          "polyline"_a, "polygon"_a, py::kw_only(), "fc"_a,
          "Like polyline_in_polygon but returns labels only (no coordinates)");
    m.def(
        "polyline_chunks_in_polygon",
        py::overload_cast<const cubao::RowVectors &,
                          const Eigen::Ref<const cubao::RowVectorsNx2> &, bool>(
            &cubao::polyline_chunks_in_polygon),
        "polyline"_a, "polygon"_a, py::kw_only(), "is_wgs84"_a = false,
        "Like polyline_in_polygon but returns labels only (no coordinates)");
    m.def("crop",
          py::overload_cast<const cubao::FastCrossing &,
                            const Eigen::Ref<const cubao::RowVectorsNx2> &,
                            const cubao::FastCrossing &>(&cubao::crop),
          "polylines_fc"_a, "polygon"_a, py::kw_only(), "polygon_fc"_a,
          "Batch crop: find chunks of all polylines in polylines_fc inside "
          "polygon. Returns dict: polyline_index -> PolylineChunks");
    m.def(
        "crop",
        py::overload_cast<const cubao::FastCrossing &,
                          const Eigen::Ref<const cubao::RowVectorsNx2> &, bool>(
            &cubao::crop),
        "polylines_fc"_a, "polygon"_a, py::kw_only(), "is_wgs84"_a = false,
        "Batch crop: find chunks of all polylines in polylines_fc inside "
        "polygon (auto-builds polygon FC). Returns dict: polyline_index -> "
        "PolylineChunks");
    m.def("crop_labels",
          py::overload_cast<const cubao::FastCrossing &,
                            const Eigen::Ref<const cubao::RowVectorsNx2> &,
                            const cubao::FastCrossing &>(&cubao::crop_labels),
          "polylines_fc"_a, "polygon"_a, py::kw_only(), "polygon_fc"_a,
          "Batch crop (labels only): like crop but skips coordinate "
          "extraction. Returns dict: polyline_index -> list of labels");
    m.def(
        "crop_labels",
        py::overload_cast<const cubao::FastCrossing &,
                          const Eigen::Ref<const cubao::RowVectorsNx2> &, bool>(
            &cubao::crop_labels),
        "polylines_fc"_a, "polygon"_a, py::kw_only(), "is_wgs84"_a = false,
        "Batch crop (labels only): like crop but skips coordinate "
        "extraction (auto-builds polygon FC). Returns dict: polyline_index "
        "-> list of labels");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
