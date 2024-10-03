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
        .def(py::init<>(), "Initialize an empty FlatBush index.")
        .def(py::init<int>(), "reserve"_a,
             "Initialize a FlatBush index with a reserved capacity.\n\n"
             ":param reserve: Number of items to reserve space for")
        .def("reserve", &FlatBush::Reserve,
             "Reserve space for a number of items.\n\n"
             ":param n: Number of items to reserve space for")
        .def("add",
             py::overload_cast<double, double, //
                               double, double, //
                               int, int>(&FlatBush::Add),
             "minX"_a, "minY"_a, //
             "maxX"_a, "maxY"_a, //
             py::kw_only(),      //
             "label0"_a = -1, "label1"_a = -1,
             "Add a bounding box to the index.\n\n"
             ":param minX: Minimum X coordinate of the bounding box\n"
             ":param minY: Minimum Y coordinate of the bounding box\n"
             ":param maxX: Maximum X coordinate of the bounding box\n"
             ":param maxY: Maximum Y coordinate of the bounding box\n"
             ":param label0: First label (optional)\n"
             ":param label1: Second label (optional)\n"
             ":return: Index of the added item")
        .def("add",
             py::overload_cast<const Eigen::Ref<const FlatBush::PolylineType> &,
                               int>(&FlatBush::Add),
             "polyline"_a, //
             py::kw_only(), "label0"_a = -1,
             "Add a polyline to the index.\n\n"
             ":param polyline: Polyline coordinates\n"
             ":param label0: Label for the polyline (optional)\n"
             ":return: Index of the added item")
        .def(
            "add",
            [](FlatBush &self,              //
               const Eigen::Vector4d &bbox, //
               int label0, int label1) -> size_t {
                return self.Add(bbox[0], bbox[1], bbox[2], bbox[3], //
                                label0, label1);
            },
            "box"_a, py::kw_only(), //
            "label0"_a = -1, "label1"_a = -1,
            "Add a bounding box to the index using a vector.\n\n"
            ":param box: Vector of [minX, minY, maxX, maxY]\n"
            ":param label0: First label (optional)\n"
            ":param label1: Second label (optional)\n"
            ":return: Index of the added item")
        .def("finish", &FlatBush::Finish, "Finish the index construction.")
        .def("boxes", &FlatBush::boxes, rvp::reference_internal,
             "Get all bounding boxes in the index.\n\n"
             ":return: Reference to the vector of bounding boxes")
        .def("labels", &FlatBush::labels, rvp::reference_internal,
             "Get all labels in the index.\n\n"
             ":return: Reference to the vector of labels")
        .def("box", &FlatBush::box, "index"_a,
             "Get the bounding box for a specific index.\n\n"
             ":param index: Index of the item\n"
             ":return: Bounding box of the item")
        .def("label", &FlatBush::label, "index"_a,
             "Get the label for a specific index.\n\n"
             ":param index: Index of the item\n"
             ":return: Label of the item")
        .def(
            "search",
            [](const FlatBush &self,       //
               double minX, double minY,   //
               double maxX, double maxY) { //
                return self.Search(minX, minY, maxX, maxY);
            },
            "minX"_a, "minY"_a, //
            "maxX"_a, "maxY"_a,
            "Search for items within a bounding box.\n\n"
            ":param minX: Minimum X coordinate of the search box\n"
            ":param minY: Minimum Y coordinate of the search box\n"
            ":param maxX: Maximum X coordinate of the search box\n"
            ":param maxY: Maximum Y coordinate of the search box\n"
            ":return: Vector of indices of items within the search box")
        .def(
            "search",
            [](const FlatBush &self, const Eigen::Vector4d &bbox) {
                return self.Search(bbox[0], bbox[1], bbox[2], bbox[3]);
            },
            "bbox"_a,
            "Search for items within a bounding box using a vector.\n\n"
            ":param bbox: Vector of [minX, minY, maxX, maxY]\n"
            ":return: Vector of indices of items within the search box")
        .def(
            "search",
            [](const FlatBush &self,       //
               const Eigen::Vector2d &min, //
               const Eigen::Vector2d &max) {
                return self.Search(min[0], min[1], max[0], max[1]);
            },
            "min"_a, "max"_a,
            "Search for items within a bounding box using min and max "
            "vectors.\n\n"
            ":param min: Vector of [minX, minY]\n"
            ":param max: Vector of [maxX, maxY]\n"
            ":return: Vector of indices of items within the search box")
        .def("size", &FlatBush::Size,
             "Get the number of items in the index.\n\n"
             ":return: Number of items in the index");
}
} // namespace cubao
