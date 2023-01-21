// should sync
// -
// https://github.com/cubao/fast-crossing/blob/master/src/pybind11_flatbush.hpp
// -
// https://github.com/cubao/headers/tree/main/include/cubao/pybind11_flatbush.hpp

#pragma once

#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "cubao_inline.hpp"
#include "flatbush.h"

namespace cubao
{
namespace py = pybind11;
using namespace pybind11::literals;
using rvp = py::return_value_policy;

CUBAO_INLINE void bind_flatbush(py::module &m)
{
    using FlatBush = flatbush::FlatBush<double>;
    py::class_<FlatBush>(m, "FlatBush", py::module_local())
        .def(py::init<>())
        .def(py::init<int>(), "reserve"_a)
        .def("reserve", &FlatBush::Reserve)
        .def("add",
             py::overload_cast<double, double, //
                               double, double, //
                               int, int>(&FlatBush::Add),
             "minX"_a, "minY"_a, //
             "maxX"_a, "maxY"_a, //
             py::kw_only(),      //
             "label0"_a = -1, "label1"_a = -1)
        .def("add",
             py::overload_cast<const Eigen::Ref<const FlatBush::PolylineType> &,
                               int>(&FlatBush::Add),
             "polyline"_a, //
             py::kw_only(), "label0"_a)

        .def(
            "add",
            [](FlatBush &self,              //
               const Eigen::Vector4d &bbox, //
               int label0, int label1) -> size_t {
                return self.Add(bbox[0], bbox[1], bbox[2], bbox[3]);
            },
            "box"_a, py::kw_only(), //
            "label0"_a = -1, "label1"_a = -1)
        .def("finish", &FlatBush::Finish)
        //
        .def("boxes", &FlatBush::boxes, rvp::reference_internal)
        .def("labels", &FlatBush::labels, rvp::reference_internal)
        .def("box", &FlatBush::box, "index"_a)
        .def("label", &FlatBush::label, "index"_a)
        //
        .def(
            "search",
            [](const FlatBush &self,       //
               double minX, double minY,   //
               double maxX, double maxY) { //
                return self.Search(minX, minY, maxX, maxY);
            },
            "minX"_a, "minY"_a, //
            "maxX"_a, "maxY"_a)
        .def(
            "search",
            [](const FlatBush &self, const Eigen::Vector4d &bbox) {
                return self.Search(bbox[0], bbox[1], bbox[2], bbox[3]);
            },
            "bbox"_a)
        .def(
            "search",
            [](const FlatBush &self,       //
               const Eigen::Vector2d &min, //
               const Eigen::Vector2d &max) {
                return self.Search(min[0], min[1], max[0], max[1]);
            },
            "min"_a, "max"_a)
        .def("size", &FlatBush::Size)
        //
        ;
}
} // namespace cubao
