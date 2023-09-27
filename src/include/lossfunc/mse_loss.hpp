#ifndef MSELOSS
#define MSELOSS
#include <xtensor/xview.hpp>
#include <node/node.hpp>
using namespace xt;

template <typename T>
class MSELoss : public CalNode<T> {
public:
    using CalNode<T>::CalNode;

    void Forward() {
        xarray<T> predict = reshape_view(this->inputs[0]->data, {this->inputs[0]->data.size()});
        xarray<T> target = this->inputs[1]->data;

        this->output->data = mean(pow(predict - target, 2), 0);
    }

    void Backward() {
        xarray<T> predict = reshape_view(this->inputs[0]->data, {this->inputs[0]->data.size()});
        xarray<T> target = this->inputs[1]->data;

        this->inputs[0]->grad += reshape_view(2 / predict.shape()[0] * (predict - target), {(INT32)predict.size(), 1, 1}); 
    }
};

#endif