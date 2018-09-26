//
// Created by yogo on 18-9-25.
//

#ifndef PEOPLEDETECTIONTEST_LASER_NET_H
#define PEOPLEDETECTIONTEST_LASER_NET_H

#include <string>
#include <vector>
#include <array>
#include "layers.h"


namespace yogoNNet {

    typedef std::array<size_t, 3> TensorShape;

    struct Tensor
    {
        TensorShape shape{{0,0,0}};
        float* data;
    };


    class Net {
        public:
            Net();
            ~Net();
            bool runNet(Tensor input, Tensor out);
            void addLayer(std::string name, LayerTypes type, TensorShape inputSize, TensorShape outputSize);
            void loadParameters(std::string path);

    private:
        std::vector<Layer> layers_vec_;
    };
}

#endif //PEOPLEDETECTIONTEST_LASER_NET_H
