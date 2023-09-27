#pragma once
#include <datatype/datatype.h>
#include <vector>
#include <set>

using namespace xt;

template<typename T>
class CalNode;

template<typename T>
class DataNode {
public:
    DataNode() = default;

    DataNode(INT32 id) : id(id) {};

    DataNode(INT32 id, xarray<T> data) : id(id), data(data) {
    }

    void SetData(xarray<T> data) {
        this->data = data;
    }

    void SetProducer(CalNode<T> &producer) {
        this->producer = &producer;
    }

    void AddConsumer(CalNode<T> &consumer) {
        this->consumers.emplace_back(&consumer);
    }

    void ZeroGrad() {
        this->grad = (T) 0;
    }

    INT32 id;
    INT32 back_cnt = 0;
    xarray<T> data;
    xarray<T> grad;
    CalNode<T> *producer = nullptr;
    std::vector<CalNode<T> *> consumers;
};

template<typename T>
class CalNode {
public:
    CalNode() = default;

    CalNode(INT32 id) : id(id) {};

    void SetOutput(DataNode<T> &output) {
        this->output = &output;
    }

    void AddInput(DataNode<T> &input) {
        this->inputs.emplace_back(&input);
        this->pre_set.emplace(input.id);
    }

    virtual void Forward() = 0;
    virtual void Backward() = 0;

    virtual void ZeroGrad() {}
    virtual void UpdateParams(FLOAT64 lr) {}

    INT32 id;
    INT32 pre_cnt = 0;
    std::vector<DataNode<T> *> inputs;
    std::set<INT32> pre_set;
    DataNode<T> *output = nullptr;
};