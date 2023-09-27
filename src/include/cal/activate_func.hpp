#ifndef ACTIVATE_FUNC
#define ACTIVATE_FUNC
#include <utils/matmul.hpp>
#include <node/node.hpp>

using namespace xt;

template<typename T>
class Sigmoid : public CalNode<T> {
public:
    using CalNode<T>::CalNode;

    xarray<T> sigmoid(xarray<T> x) {
        return 1 / (1 + exp(- x));
    }

    void Forward() override {
        this->output->data = sigmoid(this->inputs[0]->data);
    }

    void Backward() override {
        xarray<T> &X = this->inputs[0]->data;
        typename xarray<T>::shape_type shape = X.shape();

        xarray<T> grad = zeros<T>({shape[0], view(X, 0).size(), view(X, 0).size()});
        xarray<T> X_slice, seq, slice_grad;
        for(UNSIGNED_INT32 i = 0; i < shape[0]; i++) {
            X_slice = view(X, i);
            seq = reshape_view(X_slice, {X_slice.size()});
            slice_grad = diag(sigmoid(seq) * (1 - sigmoid(seq)));
            view(grad, i) = slice_grad;
        }
        this->inputs[0]->grad += MatMul<T>(this->output->grad, grad);
    }
};

template<typename T>
class ReLU : public CalNode<T> {
public:
    using CalNode<T>::CalNode;

    void Forward() override {
        this->output->data = where(this->inputs[0]->data > 0.0, this->inputs[0]->data, 0.0);
    }

    void Backward() override {
        xarray<T> &X = this->inputs[0]->data;
        typename xarray<T>::shape_type shape = X.shape();

        xarray<T> grad = zeros<T>({shape[0], view(X, 0).size(), view(X, 0).size()});
        xarray<T> X_slice, seq, slice_grad;
        for(UNSIGNED_INT32 i = 0; i < shape[0]; i++) {
            X_slice = view(X, i);
            seq = reshape_view(X_slice, {X_slice.size()});
            slice_grad = diag(where(seq < 0.0, 0.0, 1));
            view(grad, i) = slice_grad;
        }
        this->inputs[0]->grad += MatMul<T>(this->output->grad, grad);
    }
};

template<typename T>
class Tanh : public CalNode<T> {
public:
    using CalNode<T>::CalNode;

    xarray<T> tanh(xarray<T> x) {
        return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
    }

    void Forward() override {
        this->output->data = tanh(this->inputs[0]->data);
    }

    void Backward() override {
        xarray<T> &X = this->inputs[0]->data;
        typename xarray<T>::shape_type shape = X.shape();

        xarray<T> grad = zeros<T>({shape[0], view(X, 0).size(), view(X, 0).size()});
        xarray<T> X_slice, seq, slice_grad;
        for(UNSIGNED_INT32 i = 0; i < shape[0]; i++) {
            X_slice = view(X, i);
            seq = reshape_view(X_slice, {X_slice.size()});
            slice_grad = diag(1 - pow(tanh(seq), 2));
            view(grad, i) = slice_grad;
        }
        this->inputs[0]->grad += MatMul<T>(this->output->grad, grad);
    }
};

#endif