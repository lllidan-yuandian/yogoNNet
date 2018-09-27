//
// Created by yogo on 18-9-25.
//

#include "net.hpp"
#include <istream>
#include <fstream>
#include <sstream>
#include "utils.hpp"

yogoNNet::Net::Net() {

}

yogoNNet::Net::~Net(){

    for (int i = 0; i < cur_layer_size_; ++i) {
        if(layers_vec_[i]!= nullptr) {
            delete layers_vec_[i];
        }
    }

}

bool yogoNNet::Net::runNet(yogoNNet::Tensor input, yogoNNet::Tensor out) {

    for (int i = 0; i < cur_layer_size_; ++i) {
        layers_vec_[i]->forward(input,out);
        input = out;
    }
}

void yogoNNet::Net::addLayer(std::string name, LayerTypes type, TensorShape inputSize, TensorShape outputSize)
{
    if(cur_layer_size_ >= MAX_LAYER_COUNT)
    {
        return;
    }

    switch (type)
    {
        case LayerTypes::fc : {
            layers_vec_[cur_layer_size_] = new FCLayer(name, inputSize, outputSize);
            break;
        }
        case LayerTypes ::relu: {
            layers_vec_[cur_layer_size_] = new ReluLayer(name, inputSize, outputSize);
            break;
        }
        default:
            --cur_layer_size_;
    }

    ++cur_layer_size_;

}

void yogoNNet::Net::loadParameters(std::string path) {

    std::ifstream f;
    f.open(path.data(), std::ios::in);
    assert(f.is_open());

    float val;
    for (int i = 0; i < cur_layer_size_; ++i) {
        if(layers_vec_[i]->types() == 0)
        {
            FCLayer* fc_layer = (FCLayer*)layers_vec_[i];
            TensorShape i_size = fc_layer->inputSize();
            TensorShape o_size = fc_layer->outputSize();
            size_t weight_count = i_size[0]*i_size[1]*i_size[2]*o_size[0]*o_size[1]*o_size[2];
            size_t  bias_count = o_size[0]*o_size[1]*o_size[2];
            for (int j = 0; j < weight_count; ++j) {
                if (f.eof()==0){
                    f>>val;
                    fc_layer->weights()[j] = val;
                }
            }

            for (int k = 0; k < bias_count; ++k) {
                if (f.eof()==0){
                    f>>val;
                    fc_layer->bias()[k] = val;
                }
            }

        }
    }
    f.close();
}

