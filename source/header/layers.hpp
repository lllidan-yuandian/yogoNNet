//
// Created by yogo on 18-9-25.
//

#ifndef PEOPLEDETECTIONTEST_LAYERS_H
#define PEOPLEDETECTIONTEST_LAYERS_H

#include <string>

namespace yogoNNet {

    enum LayerTypes{
        fc,
        relu,
    };

    class Layer {
    public:
        Layer(std::string name ,TensorShape inputSize, TensorShape outputSize) : name_(name), input_size_(inputSize), output_size_(outputSize) {
        };

        ~Layer();

        virtual void foward(Tensor& input, Tensor& output)=0;

    private:
        TensorShape input_size_;
        TensorShape output_size_;
        std::string name_;
    };

    class FCLayer : public Layer {
    public:
        FCLayer(std::string name, TensorShape inputSize, TensorShape outputSize):Layer(name, inputSize, outputSize) {
        }

        ~FCLayer() {};

        virtual void foward(Tensor& input, Tensor& output);
    };

    class ReluLayer : public Layer {
    public:
        ReluLayer(std::string name, TensorShape inputSize, TensorShape outputSize):Layer(name, inputSize, outputSize) {
        }

        ~ReluLayer() {};

        virtual void foward(Tensor& input, Tensor& output);
    };
}

#endif //PEOPLEDETECTIONTEST_LAYERS_H
