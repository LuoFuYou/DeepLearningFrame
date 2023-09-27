#ifndef LINEAR
#define LINEAR
#include <xtensor/xrandom.hpp>
#include <utils/matmul.hpp>
#include <node/node.hpp>

using namespace xt;

template<typename T>
class Linear : public CalNode<T> {
public:
    INT32 input_features;
    INT32 output_features;
    bool has_bias;
    xarray<T> weight;
    xarray<T> weight_grad;
    xarray<T> bias;
    xarray<T> bias_grad;

    using CalNode<T>::CalNode;

    Linear(INT32 id, UNSIGNED_INT32 input_features, UNSIGNED_INT32 output_features, bool has_bias) : CalNode<T>(id), input_features(input_features), output_features(output_features), has_bias(has_bias) {
        random::seed(time(NULL));

        FLOAT32 lim = 1 / sqrt((FLOAT32)this->input_features);
        this->weight = random::rand<T>({this->output_features, this->input_features}, -lim, lim);
        this->weight_grad = (T) 0;

        if(this->has_bias) {
            this->bias = random::rand<T>({this->output_features, 1}, -lim, lim);
            this->bias_grad = (T) 0;
        }
    } 

    void Forward() override {
        xarray<T> &X = this->inputs[0]->data;
        this->output->data = zeros<T>({(INT32)X.shape()[0], this->output_features, 1});

        for(UNSIGNED_INT32 i = 0; i < X.shape()[0]; i++) {
            xarray<T> X_slice = view(X, i);
            view(this->output->data, i) = MatMul<T>(this->weight, X_slice);
            if(this->has_bias) {
                view(this->output->data, i) += this->bias;
            }
        }
    }

    void Backward() override {
        xarray<T> &X = this->inputs[0]->data;
        
        xarray<T> input_grad = broadcast(expand_dims(this->weight, 0), {(INT32)X.shape()[0], this->output_features, this->input_features});
        this->inputs[0]->grad += MatMul<T>(this->output->grad, input_grad);

        xarray<T> weight_grad = zeros<T>({(INT32)X.shape()[0], this->output_features, this->output_features * this->input_features});
        for(UNSIGNED_INT32 i = 0; i < X.shape()[0]; i++) {
            xarray<T> X_slice = view(X, i);
            view(weight_grad, i) = linalg::kron(transpose(X_slice), eye(this->output_features));
        }
        this->weight_grad += MatMul<T>(this->output->grad, weight_grad);

        if(this->has_bias) {
            this->bias_grad += this->output->grad;
        }
    }

    void ZeroGrad() override {
        this->weight_grad = (T) 0;
        this->bias_grad = (T) 0;
    }

    void UpdateParams(FLOAT64 lr) override {
        this->weight -= lr * reshape_view(mean(this->weight_grad, 0), {this->output_features, this->input_features});

        if(this->has_bias) {
            this->bias -= lr * reshape_view(mean(this->bias_grad, 0), {this->output_features, 1});
        }
    }
};

#endif