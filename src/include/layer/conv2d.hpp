#ifndef CONV2D
#define CONV2D
#include <xtensor/xrandom.hpp>
#include <xtensor/xpad.hpp>
#include <utils/matmul.hpp>
#include <node/node.hpp>

using namespace xt;

template<typename T>
class Conv2d : public CalNode<T> {
public:
    UNSIGNED_INT32 input_channels;
    UNSIGNED_INT32 output_channels;
    bool has_bias;
    UNSIGNED_INT32 kernal_size;
    UNSIGNED_INT32 stride_size;
    UNSIGNED_INT32 padding_size;
    xarray<T> weight;
    xarray<T> weight_grad;
    xarray<T> bias;
    xarray<T> bias_grad;

    Conv2d(INT32 id, UNSIGNED_INT32 input_channels, UNSIGNED_INT32 output_channels, bool has_bias, 
            UNSIGNED_INT32 kernal_size, UNSIGNED_INT32 stride_size, UNSIGNED_INT32 padding_size) : 
            id(id), input_channels(input_channels), output_channels(output_channels), has_bias(has_bias), 
            kernal_size(kernal_size), stride_size(stride_size), padding_size(padding_size){
        random::seed(time(NULL));

        FLOAT32 lim = 1 / sqrt((FLOAT32)in_channels * kernel_size * kernel_size);
        this->weight = random::rand<T>({this->output_channels, this->input_channels, this->kernal_size, this->kernal_size}, -lim, lim);
        this->weight_grad = (T) 0;

        if(has_bias) {
            this->bias = random::rand<T>({this->output_channels});
            this->bias_grad = (T) 0; 
        }
    } 

    void Forward() override {
        xarray<T> X = this->inputs[0]->data;
        UNSIGNED_INT32 input_w = X.shape()[2];
        UNSIGNED_INT32 input_h = X.shape()[3];
        UNSIGNED_INT32 out_w = (input_w + 2 * this->padding_size - this->kernel_size) / this->stride_size + 1;
        UNSIGNED_INT32 out_h = (input_h + 2 * this->padding_size - this->kernel_size) / this->stride_size + 1;
        this->output->data = zeros<T>({X.shape()[0], this->output_channels, out_w, out_h});
        X = pad(X, this->padding_size);

        for(UNSIGNED_INT32 w = 0; w < out_w; w++) {
            for(UNSIGNED_INT32 h = 0; h < out_h; h++) {
                UNSIGNED_INT32 w_s = w * this->stride_size;
                UNSIGNED_INT32 w_e = w_s + this->kernal_size;
                UNSIGNED_INT32 h_s = h * this->stride_size;
                UNSIGNED_INT32 h_e = h_s + this->kernal_size;
                for(UNSIGNED_INT32 c = 0; c < this->output_channels; c++) {
                    view(this->output->data, all(), c, w, h) = 
                    sum(view(X, all(), all(), range(w_s, w_e), range(h_s, h_e)) * view(this->weight, newaxis(), c), {1, 2, 3});

                    if(this->has_bias) {
                        view(this->output->data, all(), c, w, h) += view(this->bias, newaxis(), c);
                    }
                }
            }
        }
    }

    void Backward() override {
        xarray<T> X = this->inputs[0]->data;
        xarray<T>::shape_type shape = this->inputs[0]->data.shape();
        X = pad(X, this->padding_size);
        xarray<T> X_grad = zeros<T>({shape[0], shape[1], shape[2] + 2 * this->padding_size, shape[3] + 2 * this->padding_size});
        UNSIGNED_INT32 out_w = (input_w + 2 * this->padding_size - this->kernel_size) / this->stride_size + 1;
        UNSIGNED_INT32 out_h = (input_h + 2 * this->padding_size - this->kernel_size) / this->stride_size + 1;

        for(UNSIGNED_INT32 w = 0; w < out_w; w++) {
            for(UNSIGNED_INT32 h = 0; h < out_h; h++) {
                UNSIGNED_INT32 w_s = w * this->stride_size;
                UNSIGNED_INT32 w_e = w_s + this->kernal_size;
                UNSIGNED_INT32 h_s = h * this->stride_size;
                UNSIGNED_INT32 h_e = h_s + this->kernal_size;
                for(UNSIGNED_INT32 c = 0; c < this->output_channels; c++) {
                    xarray<T> out = view(view(this->output->grad, all(), c, w, h), all(), newaxis(), newaxis(), newaxis());

                    view(X_grad, all(), all(), range(w_s, w_e), range(h_s, h_e)) += repeat(view(this->weight, newaxis(), c), shape[0], 0) * out;

                    view(this->weight_grad, all(), c) += view(X, all(), all(), range(w_s, w_e), range(h_s, h_e)) * out;

                    if(this->has_bias) {
                        view(this->bias_grad, all(), c) += view(this->output->grad, all(), c, w, h);
                    }
                }
            }
        }

        this->inputs[0]->grad += view(X_grad, all(), all(), range(this->padding_size, this->padding_size + shape[2]), range(this->padding_size, this->padding_size + shape[3]));
    }

    void ZeroGrad() override {
        this->weight_grad = (T) 0;
        this->bias_grad = (T) 0;
    }

    void void UpdateParams(FLOAT32 lr) override {
        this->weight -= lr * mean(this->weight_grad, 0);

        if(this->has_bias) {
            this->bias -= lr * mean(this->bias_grad, 0);
        }
    }
};

#endif