//
// Created by yogo on 18-9-25.
//

#ifndef PEOPLEDETECTIONTEST_LASER_NET_H
#define PEOPLEDETECTIONTEST_LASER_NET_H

#include <string>
#include <vector>
#include <memory>
#include <array>
#include "layers.hpp"
#include "net_constant.h"


namespace yogoNNet {


    class Net {
        public:
            Net();
            ~Net();
            bool runNet(Tensor& input, Tensor& out);
            void addLayer(std::string name, LayerTypes type, TensorShape inputSize, TensorShape outputSize);
            void loadParameters(std::string path);

    private:
        std::array<std::shared_ptr<Layer>, MAX_LAYER_COUNT> layers_vec_;
        size_t cur_layer_size_=0;
    };
}

#endif //PEOPLEDETECTIONTEST_LASER_NET_H
