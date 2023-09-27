#include <pybind11/pybind11.h>
#include <xtensor-python/pyarray.hpp>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <string>
#include <node/node.hpp>
#include <layer/linear.hpp>
#define FORCE_IMPORT_ARRAY
#define USED_TYPE FLOAT32

using namespace xt;
namespace py = pybind11;

void BindLinear(py::module &m) {
    py::class_<Linear<USED_TYPE>, CalNode<USED_TYPE>>(m, "Linear")
        .def(py::init<INT32, UNSIGNED_INT32, UNSIGNED_INT32, bool>())
        .def("SetOutput", &Linear<USED_TYPE>::SetOutput)
        .def("AddInput", &Linear<USED_TYPE>::AddInput)
        .def("Forward", &Linear<USED_TYPE>::Forward)
        .def("Backward", &Linear<USED_TYPE>::Backward)
        .def("ZeroGrad", &Linear<USED_TYPE>::ZeroGrad)
        .def("UpdateParams", &Linear<USED_TYPE>::UpdateParams)
        .def_readwrite("id", &Linear<USED_TYPE>::id)
        .def_readwrite("pre_cnt", &Linear<USED_TYPE>::pre_cnt)
        .def_readwrite("inputs", &Linear<USED_TYPE>::inputs)
        .def_readwrite("pre_set", &Linear<USED_TYPE>::pre_set)
        .def_readwrite("output", &Linear<USED_TYPE>::output);
}

PYBIND11_MODULE(linear, m) {
    BindLinear(m);
}