cmake_minimum_required(VERSION 3.10)

message(${CMAKE_BINARY_DIR})

# include_directories(${CMAKE_BINARY_DIR}/../lib/)
include_directories(${CMAKE_BINARY_DIR}/../lib/xtensor/include/)
include_directories(${CMAKE_BINARY_DIR}/../lib/xtl/include/)
include_directories(${CMAKE_BINARY_DIR}/../lib/xtensor-blas/include/)
include_directories(${CMAKE_BINARY_DIR}/../lib/xtensor-python/include/)
include_directories(${CMAKE_BINARY_DIR}/../lib/pybind11/include/)
include_directories(${CMAKE_BINARY_DIR}/../src/include)
include_directories(D:/python/anaconda/Lib/site-packages/numpy/core/include)

set(PYD_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../PyPanda/bin)

project(data_node)
pybind11_add_module(data_node ${CMAKE_BINARY_DIR}/../pyd_make/data_node.cpp)
set_target_properties(data_node PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PYD_OUTPUT_DIRECTORY})

project(opt_node)
pybind11_add_module(opt_node ${CMAKE_BINARY_DIR}/../pyd_make/opt_node.cpp)
set_target_properties(opt_node PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PYD_OUTPUT_DIRECTORY})

project(graph)
pybind11_add_module(graph ${CMAKE_BINARY_DIR}/../pyd_make/graph.cpp)
set_target_properties(graph PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PYD_OUTPUT_DIRECTORY})

project(loss_func)
pybind11_add_module(loss_func ${CMAKE_BINARY_DIR}/../pyd_make/loss_func.cpp)
set_target_properties(loss_func PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PYD_OUTPUT_DIRECTORY})

project(act_func)
pybind11_add_module(act_func ${CMAKE_BINARY_DIR}/../pyd_make/act_func.cpp)
set_target_properties(act_func PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PYD_OUTPUT_DIRECTORY})

project(matmul)
pybind11_add_module(matmul ${CMAKE_BINARY_DIR}/../pyd_make/matmul.cpp)
set_target_properties(matmul PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PYD_OUTPUT_DIRECTORY})

project(linear)
pybind11_add_module(linear ${CMAKE_BINARY_DIR}/../pyd_make/linear.cpp)
set_target_properties(linear PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PYD_OUTPUT_DIRECTORY})