//
// Created by yogo on 18-9-25.
//

#ifndef PEOPLEDETECTIONTEST_LAYERS_H
#define PEOPLEDETECTIONTEST_LAYERS_H

#include <string>
#include <array>
#include <memory>
#include "net_constant.h"

#ifdef EIGEN
#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACKE
#define EIGEN_USE_LAPACKE_STRICT
#include "Eigen/Dense"
#endif


namespace yogoNNet {

    typedef std::array<size_t, 3> TensorShape;

    struct Tensor
    {
        TensorShape shape{{0,1,1}};
        size_t size(){return shape[0] * shape[1] * shape[2];}
        float* data= nullptr;
    };

    enum LayerTypes{
        fc=0,
        relu,
    };

    class Layer {
    public:
        virtual void forward(Tensor& input, Tensor& output)=0;
        virtual uint8_t types() = 0;
        TensorShape& inputSize(){return input_size_;}
        TensorShape& outputSize(){return output_size_;}

    protected:
        TensorShape input_size_;
        TensorShape output_size_;
        std::string name_;
    };

    class FCLayer : public Layer {
    public:
        FCLayer(){}
        FCLayer(std::string name, TensorShape inputSize, TensorShape outputSize);
        ~FCLayer(){}

        virtual void forward(Tensor& input, Tensor& output);
        virtual uint8_t types(){ return  int(LayerTypes::fc);};
        float* weights(){return weights_.get();}
        float* bias(){ return bias_.get();}

    protected:
        std::unique_ptr<float> weights_;
        std::unique_ptr<float> bias_;
#ifdef EIGEN
        Eigen::MatrixXf output_mat_;
#endif
    };

    class ReluLayer : public Layer {
    public:
        ReluLayer(){}
        ReluLayer(std::string name, TensorShape inputSize, TensorShape outputSize);
        ~ReluLayer() {}

        virtual void forward(Tensor& input, Tensor& output);
        virtual uint8_t types(){return int(LayerTypes::relu);}
    };
}

#endif //PEOPLEDETECTIONTEST_LAYERS_H
