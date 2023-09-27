#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <xtensor-python/pyarray.hpp>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <string>
#include <node/node.hpp>
#define FORCE_IMPORT_ARRAY
#define USED_TYPE FLOAT32

using namespace xt;
namespace py = pybind11;

// 绑定DataNode类
void BindDataNode(py::module &m) {
    py::class_<DataNode<USED_TYPE>>(m, "DataNode")
        .def(py::init<INT32>())
        .def(py::init<INT32, xarray<USED_TYPE>>())
        .def("SetData", &DataNode<USED_TYPE>::SetData)
        .def("SetProducer", &DataNode<USED_TYPE>::SetProducer)
        .def("AddConsumer", &DataNode<USED_TYPE>::AddConsumer)
        .def("ZeroGrad", &DataNode<USED_TYPE>::ZeroGrad)
        .def_readwrite("id", &DataNode<USED_TYPE>::id)
        .def_readwrite("back_cnt", &DataNode<USED_TYPE>::back_cnt)
        .def_readwrite("data", &DataNode<USED_TYPE>::data)
        .def_readwrite("grad", &DataNode<USED_TYPE>::grad)
        .def_readwrite("producer", &DataNode<USED_TYPE>::producer)
        .def_readwrite("consumers", &DataNode<USED_TYPE>::consumers);
}

// 创建Python模块
PYBIND11_MODULE(data_node, m) {
    BindDataNode(m);
}