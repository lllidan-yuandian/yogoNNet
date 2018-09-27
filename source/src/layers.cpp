//
// Created by yogo on 18-9-25.
//

#include "layers.hpp"
#include <algorithm>

#ifdef EIGEN

#include "Eigen/Dense"
#endif

yogoNNet::FCLayer::FCLayer(std::string name, yogoNNet::TensorShape inputSize, yogoNNet::TensorShape outputSize)
{
    name_ = name;
    input_size_ = inputSize;
    output_size_ = outputSize;
    weights_ = std::unique_ptr<float>(new float[inputSize[0]*inputSize[1]*inputSize[2]*outputSize[0]*outputSize[1]*outputSize[2]]);
    bias_ = std::unique_ptr<float>(new float[outputSize[0]*outputSize[1]*outputSize[2]]);
}


void yogoNNet::FCLayer::forward(yogoNNet::Tensor &input, yogoNNet::Tensor &output) {

#ifdef EIGEN
    Eigen::MatrixXf input_mat = Eigen::Map<Eigen::MatrixXf>(input.data, input.shape[0], input.shape[1]);
    Eigen::MatrixXf weight_mat = Eigen::Map<Eigen::MatrixXf>(weights(), input.shape[0], output.shape[1]);
    output_mat_ = input_mat*weight_mat;
    output.data = output_mat_.data();
#endif
}


yogoNNet::ReluLayer::ReluLayer(std::string name, yogoNNet::TensorShape inputSize, yogoNNet::TensorShape outputSize) {
    name_ = name;
    input_size_ = inputSize;
    output_size_ = outputSize;
}


void yogoNNet::ReluLayer::forward(yogoNNet::Tensor &input, yogoNNet::Tensor &output) {

    size_t size = input.shape[0]*input.shape[1]*input.shape[2];
#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        input.data[i] = std::max(input.data[i], .0f);
    }
    output.data=input.data;
}


