#ifndef CELOSS
#define CELOSS
#include <xtensor/xmanipulation.hpp>
#include <xtensor/xview.hpp>
#include <node/node.hpp>
#include <utils/matmul.hpp>
using namespace xt;

template <typename T>
class CELoss : public CalNode<T> {
public:
    using CalNode<T>::CalNode;

    xarray<T> SoftMax(const xarray<T> &x) {
        xarray<T> sum_pow = expand_dims(sum(exp(x), 1), 1);
        return exp(x) / sum_pow;
    }

    xarray<T> SoftMaxDerivative(const xarray<T> &p) {
        INT32 batch = p.shape()[0];
        INT32 dim = p.shape()[1]; 
        
        xarray<T> A = broadcast(expand_dims(p, 2), {batch, dim, dim});
        xarray<T> B = broadcast(expand_dims(eye(dim), 0), {batch, dim, dim}) - broadcast(expand_dims(p, 1), {batch, dim, dim}); 

        return A * B;
    }

    void Forward() {
        typename xarray<T>::shape_type shape = this->inputs[0]->data.shape();
        xarray<T> predict = reshape_view(this->inputs[0]->data, {shape[0], shape[1]});
        xarray<T> target = this->inputs[1]->data;

        xarray<T> p = SoftMax(predict); 
        this->output->data = mean(sum(target * (- log(p)), 1), 0);
    }

    void Backward() {
        typename xarray<T>::shape_type shape = this->inputs[0]->data.shape();
        xarray<T> predict = reshape_view(this->inputs[0]->data, {shape[0], shape[1]});
        xarray<T> target = this->inputs[1]->data;

        xarray<T> p = SoftMax(predict);
        xarray<T> g0 = target / p;
        g0 = reshape_view(g0, {(INT32)shape[0], 1, (INT32)shape[1]});
        xarray<T> g1 = SoftMaxDerivative(p);

        this->inputs[0]->grad += MatMul<T>(g0, g1);
    }
};

#endif