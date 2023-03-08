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
#include "pybind11_fast_crossing.hpp"
#include "pybind11_flatbush.hpp"
#include "pybind11_nano_kdtree.hpp"

#include "pybind11_polyline_ruler.hpp"

#include "point_in_polygon.hpp"
#include "densify_polyline.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(fast_crossing, m)
{
    cubao::bind_fast_crossing(m);
    cubao::bind_flatbush(m);
    cubao::bind_polyline_ruler(m);

    m.def("point_in_polygon", &cubao::point_in_polygon, //
          py::kw_only(), "points"_a, "polygon"_a,
          "point-in-polygon test, returns 0-1 mask");
    m.def("densify_polyline", &cubao::densify_polyline, //
          "polyline"_a, py::kw_only(), "max_gap"_a,
          "densify polyline, interpolate to satisfy max_gap");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
