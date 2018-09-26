//
// Created by yogo on 18-9-25.
//

#include "net.h"
#include <Eigen/Dense>

yogoNNet::Net::Net() {

}

yogoNNet::Net::~Net(){

}

bool yogoNNet::Net::runNet(yogoNNet::Tensor input, yogoNNet::Tensor out) {
    if(input.shape[0] == 0 && input.shape[1] == 0 && input.shape[1] == 0)
    {
        return false;
    }

    for (int i = 0; i < layers_vec_.size(); ++i) {
        layers_vec_[i].foward(input,out);
        input = out;
    }
}

void yogoNNet::Net::addLayer(std::string name, LayerTypes type, TensorShape inputSize, TensorShape outputSize)
{
    switch (type)
    {
        case LayerTypes::fc :
            layers_vec_.emplace_back(FCLayer(name, inputSize, outputSize));
            break;
        case LayerTypes ::relu:
            layers_vec_.emplace_back(ReluLayer(name, inputSize, outputSize));
    }

}

void yogoNNet::Net::loadParameters(std::string path) {}

