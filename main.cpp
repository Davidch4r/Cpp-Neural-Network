#include <iostream>
#include <vector>
#include "NeuralNetwork.h"

int main() {
    const std::vector<int> layers = {3, 10, 10, 8, 5, 3,  1};
    const std::vector<std::string> activations = {"input", "relu", "relu", "relu", "relu", "relu", "sigmoid"};
    const std::vector<std::vector<float>> inputs = {
        // XOR
        {0, 0, 0},
        {0, 0, 1},
        {0, 1, 0},
        {0, 1, 1},
        // AND
        {0.2f, 0, 0},
        {0.2f, 1, 0},
        {0.2f, 0, 1},
        {0.2f, 1, 1},
        // OR
        {0.4f, 0, 0},
        {0.4f, 1, 0},
        {0.4f, 0, 1},
        {0.4f, 1, 1},
        // NOR
        {0.6f, 0, 0},
        {0.6f, 1, 0},
        {0.6f, 0, 1},
        {0.6f, 1, 1},
        // NAND
        {0.8f, 0, 0},
        {0.8f, 1, 0},
        {0.8f, 0, 1},
        {0.8f, 1, 1},
        // XNOR
        {1, 0, 0},
        {1, 1, 0},
        {1, 0, 1},
        {1, 1, 1},

    };
    const std::vector<std::vector<float>> expectedOutputs = {
        // XOR
        {0},
        {1},
        {1},
        {0},
        // AND
        {0},
        {0},
        {0},
        {1},
        // OR
        {0},
        {1},
        {1},
        {1},
        // NOR
        {1},
        {0},
        {0},
        {0},
        // NAND
        {1},
        {1},
        {1},
        {0},
        // XNOR
        {1},
        {0},
        {0},
        {1},
    };
    NeuralNetwork net(layers, activations);
    net.Learn(inputs, expectedOutputs, 50000, 0.01f, 0.5f);
    for (auto &input : inputs) {
        std::cout << "Input: ";
        for (auto &inputVal : input) {
            std::cout << inputVal << ", ";
        }
        std::cout << "\nOutput: ";
        std::vector<float> output = net.getOutputs(input);
        for (auto &outputVal : output) {
            std::cout << outputVal << ", ";
        }
        std::cout << std::endl;
    }
    net.Test(inputs, expectedOutputs, 'A');
    return 0;
}
