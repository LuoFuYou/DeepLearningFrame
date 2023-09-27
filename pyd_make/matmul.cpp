#include <pybind11/pybind11.h>
#include <xtensor-python/pyarray.hpp>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <string>
#include <utils/matmul.hpp>
#define FORCE_IMPORT_ARRAY
#define USED_TYPE FLOAT32

using namespace xt;
namespace py = pybind11;

void BindMatMul(py::module &m) {
    m.def("MatMul", &MatMul<USED_TYPE>);
}

PYBIND11_MODULE(matmul, m) {
    BindMatMul(m);
}