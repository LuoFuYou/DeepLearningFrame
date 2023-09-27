#include <pybind11/pybind11.h>
#include <xtensor-python/pyarray.hpp>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <string>
#include <node/node.hpp>
#define FORCE_IMPORT_ARRAY
#define USED_TYPE FLOAT32

using namespace xt;
namespace py = pybind11;

void BindCalNode(py::module &m) {
    class PyCalNode : public CalNode<USED_TYPE> {
    public:
        using CalNode<USED_TYPE>::CalNode;

        void Forward() override {
            PYBIND11_OVERRIDE_PURE(
                void,
                CalNode<USED_TYPE>,
                Forward
            );
        }

        void Backward() override {
            PYBIND11_OVERRIDE_PURE(
                void,
                CalNode<USED_TYPE>,
                Backward
            );
        }
    };

    py::class_<CalNode<USED_TYPE>, PyCalNode>(m, "CalNode")
        .def(py::init<INT32>())
        .def("SetOutput", &CalNode<USED_TYPE>::SetOutput)
        .def("AddInput", &CalNode<USED_TYPE>::AddInput)
        .def("Forward", &CalNode<USED_TYPE>::Forward)
        .def("Backward", &CalNode<USED_TYPE>::Backward)
        .def("ZeroGrad", &CalNode<USED_TYPE>::ZeroGrad)
        .def("UpdateParams", &CalNode<USED_TYPE>::UpdateParams)
        .def_readwrite("id", &CalNode<USED_TYPE>::id)
        .def_readwrite("pre_cnt", &CalNode<USED_TYPE>::pre_cnt)
        .def_readwrite("inputs", &CalNode<USED_TYPE>::inputs)
        .def_readwrite("pre_set", &CalNode<USED_TYPE>::pre_set)
        .def_readwrite("output", &CalNode<USED_TYPE>::output);
}

PYBIND11_MODULE(opt_node, m) {
    BindCalNode(m);
}
