//
// Created by yogo on 18-9-27.
//

#include "yogoNNet.h"
using namespace yogoNNet;

int main(int code){

    Net laserNet;
    TensorShape i_shape = {256,1,1};
    TensorShape o_shape = {2,1,1};
    std::string model_path = "/home/yogo/workspace/laser-detection/laserNet";

    laserNet.addLayer("fc1", LayerTypes::fc, i_shape, TensorShape{16,1,1});
    laserNet.addLayer("relu1", LayerTypes::relu, TensorShape{16,1,1}, TensorShape{16,1,1});
    laserNet.addLayer("fc2", LayerTypes::fc, TensorShape{16,1,1}, TensorShape{8,1,1});
    laserNet.addLayer("relu2", LayerTypes::relu, TensorShape{8,1,1}, TensorShape{8,1,1});
    laserNet.addLayer("fc3", LayerTypes::fc, TensorShape{8,1,1}, o_shape);

    laserNet.loadParameters(model_path);


    Tensor input_tensor;
    input_tensor.data = (float*)malloc(sizeof(float)*i_shape[0]*i_shape[1]*i_shape[2]);
    float a = -5;
    float b = 5;
    for (int i = 0; i < i_shape[0] * i_shape[1] * i_shape[2]; ++i) {
        input_tensor.data[i] = a + (int)b * rand() / (RAND_MAX + 1);
    }

    Tensor result;
    laserNet.runNet(input_tensor, result);

    return 0;
}