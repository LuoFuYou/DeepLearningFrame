#include <pybind11/pybind11.h>
#include <xtensor-python/pyarray.hpp>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <string>
#include <graph/graph.hpp>
#define FORCE_IMPORT_ARRAY
#define USED_TYPE FLOAT32

using namespace xt;
namespace py = pybind11;

void BindGraph(py::module &m) {
    py::class_<Graph<USED_TYPE>>(m, "Graph")
        .def(py::init<>())
        .def("AddBeginNode", &Graph<USED_TYPE>::AddBeginNode)
        .def("AddDataNode", &Graph<USED_TYPE>::AddDataNode)
        .def("AddCalNode", &Graph<USED_TYPE>::AddCalNode)
        .def("NextId", &Graph<USED_TYPE>::NextId)
        .def("Forward", &Graph<USED_TYPE>::Forward)
        .def("Backward", &Graph<USED_TYPE>::Backward)
        .def("ZeroGrad", &Graph<USED_TYPE>::ZeroGrad)
        .def("UpdateParams", &Graph<USED_TYPE>::UpdateParams, py::arg("lr"))
        .def_readwrite("next_id", &Graph<USED_TYPE>::next_id)
        .def_readwrite("begin_nodes", &Graph<USED_TYPE>::begin_nodes)
        .def_readwrite("end_nodes", &Graph<USED_TYPE>::end_nodes)
        .def_readwrite("data_nodes", &Graph<USED_TYPE>::data_nodes)
        .def_readwrite("cal_nodes", &Graph<USED_TYPE>::cal_nodes);
}

PYBIND11_MODULE(graph, m) {
    BindGraph(m);
}
