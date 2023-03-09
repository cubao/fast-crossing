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
    using Quiver = cubao::Quiver;
    py::class_<Quiver>(m, "Quiver", py::module_local())
        //
        ;

    using Arrow = cubao::Arrow;
    py::class_<Arrow>(m, "Arrow", py::module_local())
        //
        ;
}
} // namespace cubao
