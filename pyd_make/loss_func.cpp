#include <pybind11/pybind11.h>
#include <xtensor-python/pyarray.hpp>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <string>
#include <lossfunc/ce_loss.hpp>
#include <lossfunc/mse_loss.hpp>
#define FORCE_IMPORT_ARRAY
#define USED_TYPE FLOAT32

using namespace xt;
namespace py = pybind11;

void BindLoss(py::module &m) {
    py::class_<MSELoss<USED_TYPE>, CalNode<USED_TYPE>>(m, "MSELoss")
        .def(py::init<INT32>())
        .def("SetOutput", &MSELoss<USED_TYPE>::SetOutput)
        .def("AddInput", &MSELoss<USED_TYPE>::AddInput)
        .def("Forward", &MSELoss<USED_TYPE>::Forward)
        .def("Backward", &MSELoss<USED_TYPE>::Backward)
        .def_readwrite("id", &MSELoss<USED_TYPE>::id)
        .def_readwrite("pre_cnt", &MSELoss<USED_TYPE>::pre_cnt)
        .def_readwrite("inputs", &MSELoss<USED_TYPE>::inputs)
        .def_readwrite("pre_set", &MSELoss<USED_TYPE>::pre_set)
        .def_readwrite("output", &MSELoss<USED_TYPE>::output);

    py::class_<CELoss<USED_TYPE>, CalNode<USED_TYPE>>(m, "CELoss")
        .def(py::init<INT32>())
        .def("SetOutput", &CELoss<USED_TYPE>::SetOutput)
        .def("AddInput", &CELoss<USED_TYPE>::AddInput)
        .def("Forward", &CELoss<USED_TYPE>::Forward)
        .def("Backward", &CELoss<USED_TYPE>::Backward)
        .def_readwrite("id", &CELoss<USED_TYPE>::id)
        .def_readwrite("pre_cnt", &CELoss<USED_TYPE>::pre_cnt)
        .def_readwrite("inputs", &CELoss<USED_TYPE>::inputs)
        .def_readwrite("pre_set", &CELoss<USED_TYPE>::pre_set)
        .def_readwrite("output", &CELoss<USED_TYPE>::output);
}

PYBIND11_MODULE(loss_func, m) {
    BindLoss(m);
}