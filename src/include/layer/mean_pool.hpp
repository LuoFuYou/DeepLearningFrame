#ifndef MEANPOOL
#define MEANPOOL
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
    UNSIGNED_INT32 kernal_size;
    UNSIGNED_INT32 stride_size;
    UNSIGNED_INT32 padding_size;

    Conv2d(INT32 id, UNSIGNED_INT32 kernal_size, UNSIGNED_INT32 stride_size, UNSIGNED_INT32 padding_size) : 
            id(id), kernal_size(kernal_size), stride_size(stride_size), padding_size(padding_size) {} 

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
                view(this->output->data, all(), all(), w, h) = mean(view(X, all(), all(), range(w_s, w_e), range(h_s, h_e)), {2, 3});
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
                
                view(X_grad, all(), all(), range(w_s, w_e), range(h_s, h_e)) += view(view(this->output->grad, all(), all(), w, h) / (this->kernal_size * this->kernal_size), all(), all(), newaxis(), newaxis());
            }
        }

        this->inputs[0]->grad += view(X_grad, all(), all(), range(this->padding_size, this->padding_size + shape[2]), range(this->padding_size, this->padding_size + shape[3]));
    }
};

#endif