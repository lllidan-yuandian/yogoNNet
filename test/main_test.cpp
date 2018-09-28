//
// Created by yogo on 18-9-27.
//

#include "yogoNNet.h"
using namespace yogoNNet;

#include <iostream>

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
    input_tensor.shape={256,1,1};
    input_tensor.data = new float[i_shape[0]*i_shape[1]*i_shape[2]];
    for (int i = 0; i < i_shape[0] * i_shape[1] * i_shape[2]; ++i) {
        input_tensor.data[i] = 0.1f;
    }

    Tensor result;
    clock_t  start_c, end_c;
    start_c = clock();
    for (int j = 0; j < 20; ++j) {
        laserNet.runNet(input_tensor, result);
    }
    end_c = clock();
    std::cout<<"elapse time:"<<(double)(end_c - start_c)/CLOCKS_PER_SEC<<std::endl;
    std::cout<<result.data[0]<<","<<result.data[1]<<std::endl;
    delete [] input_tensor.data;
    return 0;
}