#include <pybind11/pybind11.h>
#include <xtensor-python/pyarray.hpp>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <string>
#include <cal/activate_func.hpp>
#define FORCE_IMPORT_ARRAY
#define USED_TYPE FLOAT32

using namespace xt;
namespace py = pybind11;

void BindActFunc(py::module &m) {
    py::class_<Sigmoid<USED_TYPE>, CalNode<USED_TYPE>>(m, "Sigmoid")
        .def(py::init<INT32>())
        .def("SetOutput", &Sigmoid<USED_TYPE>::SetOutput)
        .def("AddInput", &Sigmoid<USED_TYPE>::AddInput)
        .def("Forward", &Sigmoid<USED_TYPE>::Forward)
        .def("Backward", &Sigmoid<USED_TYPE>::Backward)
        .def_readwrite("id", &Sigmoid<USED_TYPE>::id)
        .def_readwrite("pre_cnt", &Sigmoid<USED_TYPE>::pre_cnt)
        .def_readwrite("inputs", &Sigmoid<USED_TYPE>::inputs)
        .def_readwrite("pre_set", &Sigmoid<USED_TYPE>::pre_set)
        .def_readwrite("output", &Sigmoid<USED_TYPE>::output);

    py::class_<ReLU<USED_TYPE>, CalNode<USED_TYPE>>(m, "ReLU")
        .def(py::init<INT32>())
        .def("SetOutput", &ReLU<USED_TYPE>::SetOutput)
        .def("AddInput", &ReLU<USED_TYPE>::AddInput)
        .def("Forward", &ReLU<USED_TYPE>::Forward)
        .def("Backward", &ReLU<USED_TYPE>::Backward)
        .def_readwrite("id", &ReLU<USED_TYPE>::id)
        .def_readwrite("pre_cnt", &ReLU<USED_TYPE>::pre_cnt)
        .def_readwrite("inputs", &ReLU<USED_TYPE>::inputs)
        .def_readwrite("pre_set", &ReLU<USED_TYPE>::pre_set)
        .def_readwrite("output", &ReLU<USED_TYPE>::output);


    py::class_<Tanh<USED_TYPE>, CalNode<USED_TYPE>>(m, "Tanh")
        .def(py::init<INT32>())
        .def("SetOutput", &Tanh<USED_TYPE>::SetOutput)
        .def("AddInput", &Tanh<USED_TYPE>::AddInput)
        .def("Forward", &Tanh<USED_TYPE>::Forward)
        .def("Backward", &Tanh<USED_TYPE>::Backward)
        .def_readwrite("id", &Tanh<USED_TYPE>::id)
        .def_readwrite("pre_cnt", &Tanh<USED_TYPE>::pre_cnt)
        .def_readwrite("inputs", &Tanh<USED_TYPE>::inputs)
        .def_readwrite("pre_set", &Tanh<USED_TYPE>::pre_set)
        .def_readwrite("output", &Tanh<USED_TYPE>::output);
}

PYBIND11_MODULE(act_func, m) {
    BindActFunc(m);
}