#pragma once
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <node/node.hpp>

using namespace xt;

template <typename T>
xarray<T> MatMul(const xarray<T>& A, const xarray<T>& B) {
    typename xarray<T>::shape_type shape = A.shape();
    *(shape.end() - 1) = *(B.shape().end() - 1);
    xarray<T> out(shape);
    if(shape.size() == 2) {
        if (A.shape()[1] != B.shape()[0]) {
            throw std::runtime_error("Matrix dimensions are not compatible for multiplication.");
        }

        for(UNSIGNED_INT32 i = 0; i < shape[1]; i++) {
            xarray<T> vector = view(B, all(), i);
            view(out, all(), i) = linalg::dot(A, vector); 
        }

        return out;
    }

    for(UNSIGNED_INT32 i = 0; i < shape[0]; i++) {
        xarray<T> A_slice = view(A, i);
        xarray<T> B_slice = view(B, i);
        view(out, i) = MatMul<T>(A_slice, B_slice);
    }

    return out;
}