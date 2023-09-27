#ifndef GRAPH
#define GRAPH
#include <node/node.hpp>
#include <queue>

using namespace xt;

template<typename T>
class Graph {
public:
    INT32 next_id = 0;
    std::vector<DataNode<T> *> begin_nodes;
    std::vector<DataNode<T> *> end_nodes;
    std::vector<DataNode<T> *> data_nodes;
    std::vector<CalNode<T> *> cal_nodes;

    void AddBeginNode(DataNode<T> &begin_node) {
        this->begin_nodes.emplace_back(&begin_node);
    }

    void AddDataNode(DataNode<T> &data_node) {
        this->data_nodes.emplace_back(&data_node);
    }

    void AddCalNode(CalNode<T> &cal_node) {
        this->cal_nodes.emplace_back(&cal_node);
    }

    INT32 NextId() {
        return this->next_id++;
    }

    void Forward() {
        std::queue<DataNode<T> *> q;
        for(auto itr = this->begin_nodes.begin(); itr != this->begin_nodes.end(); ++itr) {
            q.emplace(*itr);
        }

        while(!q.empty()) {
            DataNode<T> *data_node = q.front();
            q.pop();

            if (data_node->consumers.empty()) {
                this->end_nodes.emplace_back(data_node);
            }else {
                for (auto itr = data_node->consumers.begin(); itr != data_node->consumers.end(); ++itr) {
                    CalNode<T> *cal_node = *itr;
                    cal_node->pre_cnt++;
                    if(cal_node->pre_cnt == cal_node->pre_set.size()) {
                        cal_node->Forward();
                        cal_node->pre_cnt = 0;
                        q.emplace(cal_node->output);
                    }
                }
            }
        }
    }

    void Backward() {
        std::queue<DataNode<T> *> q;
        for(auto itr = this->end_nodes.begin(); itr != this->end_nodes.end(); ++itr) {
            (*itr)->grad = (T)1;
            q.emplace(*itr);
        }

        while(!q.empty()) {
            DataNode<T> *data_node = q.front();
            q.pop();

            if (data_node->producer) {
                data_node->producer->Backward();
                data_node->back_cnt = 0;
                for (auto itr = data_node->producer->inputs.begin(); itr != data_node->producer->inputs.end(); ++itr) {
                    DataNode<T> *pre_node = *itr;
                    pre_node->back_cnt++;
                    if(pre_node->back_cnt == pre_node->consumers.size()) {
                        q.emplace(pre_node);
                    }
                }
            }
        }
    }

    void ZeroGrad() {
        for(auto itr = this->data_nodes.begin(); itr != this->data_nodes.end(); itr++) {
            DataNode<T> *data_node = *itr;
            data_node->ZeroGrad();
        }

        for(auto itr = this->cal_nodes.begin(); itr != this->cal_nodes.end(); itr++) {
            CalNode<T> *cal_node = *itr;
            cal_node->ZeroGrad();
        }           
    }

    void UpdateParams(FLOAT64 lr) {
        for(auto itr = this->cal_nodes.begin(); itr != this->cal_nodes.end(); itr++) {
            CalNode<T> *cal_node = *itr;
            cal_node->UpdateParams(lr);
        }
    }
};

#endif