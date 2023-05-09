//
// Created by David Pidugu on 5/9/2023.
//


#include <cmath>
#include <iostream>
#include<map>
#include <random>
#include <numeric>
#include <cmath>
#include <ctime>


#ifndef CPPNN_NEURALNETWORK_H
#define CPPNN_NEURALNETWORK_H

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<int> &layers, const std::vector<std::string> &activationStrings);
    void Learn(const std::vector<std::vector<float>> &inputData, const std::vector<std::vector<float>> &outputData, const int &epochs, const float &learningRate, const float batchSizePercent);
    void Test(const std::vector<std::vector<float>> &inputData, const std::vector<std::vector<float>> &outputData, char test);
    std::vector<float> getOutputs(const std::vector<float> &inputVals);

private:
    std::map<std::string, int> activationMap = {
            {"sigmoid", 0},
            {"tanh", 1},
            {"relu", 2},
            {"leakyRelu", 3},
            {"input", 4}
    };
    std::vector<int> activations;
    std::vector<std::vector<float>> network;
    std::vector<std::vector<float>> bias;
    std::vector<std::vector<std::vector<float>>> weights;

    void feedForward(const std::vector<float> &inputVals);
    void backPropagate(const std::vector<float> &expectedOutputs, float learningRate);
    static float activation(float x, int n);
    static float activationDerivative(float x, int n);
    static std::vector<std::vector<std::vector<float>>> getBatch(const std::vector<std::vector<float>> &inputData, const std::vector<std::vector<float>> &outputData, int batchSize);
};


/**
    * Reserves Sizes, Instantiates Network, Weights, Biases and Activations
    *
    * @param layers -> Vector in format of {InputAmount, HiddenAmount1, HiddenAmount2 ... , OutputAmount}
    * @param activationStrings -> Vectpr in format of {InputFunction, HiddenFunction1, HiddenFunction2 ... , OutputFunction}.
    *
    * Note: While you can customize Input Activation function, it is recommended to put "input" as the activation
 */
NeuralNetwork::NeuralNetwork(const std::vector<int> &layers, const std::vector<std::string> &activationStrings) {
    srand(time(nullptr));
    unsigned layerSize = layers.size();
    if (layerSize != activationStrings.size())
        throw std::invalid_argument("The layers and activation std::vectors must be constant");
    activations.reserve(layerSize);
    network.reserve(layerSize);
    bias.reserve(layerSize);
    weights.reserve(layerSize);
    for (int i = 0; i < layerSize; i++) {
        std::vector<float> newNetLayer(layers[i]);
        std::vector<float> newBiasLayer(layers[i]);
        std::vector<std::vector<float>> newWeightsLayer(layers[i]);
        for (int j = 0; j < layers[i]; j++) {
            std::vector<float> connections;
            if (i < layerSize - 1) {
                connections.resize(layers[i + 1]);
                for (int k = 0; k < layers[i + 1]; k++) {
                    connections[k] = static_cast<float>(std::rand()) / RAND_MAX * 2 - 1;
                }
            }
            newNetLayer[j] = 0;
            newBiasLayer[j] = static_cast<float>(std::rand()) / RAND_MAX * 2 - 1;
            newWeightsLayer[j] = connections;
        }
        activations.push_back(activationMap[activationStrings[i]]);
        network.push_back(newNetLayer);
        bias.push_back(newBiasLayer);
        weights.push_back(newWeightsLayer);
    }
}

/**
 * Learning algorithm of the Network through forward and backward propagation with batches
 *
 * @param inputData -> vector in a size equal to the inputNode size; Data for each output
 * @param outputData -> vector in a size equal to the outputNode size; Data for the output in respect to input index
 * @param epochs -> Epochs of the learning
 * @param learningRate -> Learning Rate of the learning
 * @param batchSizePercent -> How much percent of the data is in one batch
 */

void NeuralNetwork::Learn(const std::vector<std::vector<float>> &inputData, const std::vector<std::vector<float>> &outputData, const int &epochs, const float &learningRate, const float batchSizePercent) {
    if (inputData.size() != outputData.size())
        throw std::invalid_argument("InputData Size must equal OutputData size");
    if (batchSizePercent > 1)
        throw std::length_error("Batch Size Percentage cannot be about 1");
    std::cout << "Learning Started... " << std::endl;
    unsigned dataSize = inputData.size();
    int batchSize = static_cast<int>(dataSize * batchSizePercent);
    int divisible10 = epochs / 10;
    for (int i = 0; i < epochs; i++) {
        std::vector<std::vector<std::vector<float>>> batch = getBatch(inputData, outputData, batchSize);
        for (int j = 0; j < batchSize; j++) {
            feedForward(batch[j][0]);
            backPropagate(batch[j][1], learningRate);
        }
        if (i % divisible10 == 0) {
            float totalError = 0;
            for (int j = 0; j < batchSize; j++) {
                std::vector<float> outputs = getOutputs(batch[j][0]);
                std::vector<float> expectedOutputs = batch[j][1];
                for (unsigned k = 0; k < expectedOutputs.size(); k++) {
                    totalError += 0.5f * std::pow(expectedOutputs[k] - outputs[k], 2);
                }
            }
            float averageError = totalError / static_cast<float>(batchSize);
            std::cout<< "Average Error iter " << i << " -> " << averageError << std::endl;
        }
    }
    std::cout << "Learning Ended" << std::endl;

}
/**
 * Testing the accuracy of the Neural Network
 *
 * @param inputData -> vector in a size equal to the inputNode size; Data for each output
 * @param outputData -> vector in a size equal to the outputNode size; Data for the output in respect to input index
 * @param test -> the test used to calculate score. Test 'A' is rounding the outputs, Test 'B' is normalizing it, Test 'C' is soft-maxing the output
 *
 * Note: While useful for checking efficiency, not needed for Learning
 */
void NeuralNetwork::Test(const std::vector<std::vector<float>> &inputData, const std::vector<std::vector<float>> &outputData, char test) {
    if (inputData.size() != outputData.size())
        throw std::invalid_argument("InputData Size must equal OutputData size");
    std::cout << "Testing..." << std::endl;
    unsigned dataSize = inputData.size();
    unsigned outputDataSize = outputData[0].size();
    unsigned totalSize = dataSize * outputDataSize;
    int totalCorrect = 0;

    for (int i = 0; i < dataSize; i++) {
        std::vector<float> answers = getOutputs(inputData[i]);

        switch(test) {
            case 'A': {
                for (int j = 0; j < outputDataSize; j++) {
                    answers[j] = std::round(answers[j]);
                }
                break;
            }
            case 'B': {
                auto sum = accumulate(answers.begin(), answers.end(), 0.0f);
                for (int j = 0; j < outputDataSize; j++) {
                    answers[j] = answers[j] / sum;
                }

                break;
            }
            case 'C': {
                float expSum = 0;
                for (int j = 0; j < outputDataSize; j++) {
                    expSum += std::exp(answers[j]);
                }
                for (int j = 0; j < outputDataSize; j++) {
                    answers[j] = std::exp(answers[j]) / expSum;
                }

                break;
            }
            default: { }
        }

        for (int j = 0; j < outputDataSize; j++) {
            float expectedAnswer = outputData[i][j];
            float answer = answers[j];
            if (test == 'B' || test == 'C')
                answer = std::round(answer);
            if (answer == expectedAnswer)
                totalCorrect++;
        }

    }

    float percentage = 100 * static_cast<float>(totalCorrect) / static_cast<float>(totalSize);

    std::cout << "--------------\nPercentage Correct: " << percentage << "% \nTotal Correct: " << totalCorrect << "/" << totalSize << "\n--------------" << std::endl;

}
/**
 * gets output by feeding forward the network and measuring the output nodes
 *
 * @param inputVals -> vector in a size equal to the inputNode size; Get outputs for this vector
 * @return output vector from the inputs
 */
std::vector<float> NeuralNetwork::getOutputs(const std::vector<float> &inputVals) {
    if (inputVals.size() != network.begin()->size())
        throw std::invalid_argument("InputVals size must equal the the size of the network inputs");
    feedForward(inputVals);
    unsigned outputSize = network.back().size();
    std::vector<float> outputs(outputSize);
    for (int i = 0; i < outputSize; i++) {
        outputs[i] = network.back()[i];
    }
    return outputs;
}
/**
 * Feeding forward values
 *
 * @param inputVals -> Should be already checked before using feedforward
 *
 * Note: This does not return an output, rather it just feeds forward the network
 */
void NeuralNetwork::feedForward(const std::vector<float> &inputVals) {
    unsigned inputSize = inputVals.size();
    unsigned networkSize = network.size();
    for (int i = 0; i < inputSize; i++) {
        network[0][i] = inputVals[i];
    }
    for (int i = 1; i < networkSize; i++) {
        unsigned layerSize = network[i].size();
        unsigned prevLayerSize = network[i - 1].size();
        for (int j = 0; j < layerSize; j++) {
            float value = 0;
            for (int k = 0; k < prevLayerSize; k++) {
                value += network[i - 1][k] * weights[i - 1][k][j];
            }
            value += bias[i][j];
            network[i][j] = activation(value, activations[i]);
        }
    }
}
/**
 * Backpropogate the algorithm through goes through the slope of the gradiant
 *
 * @param expectedOutputs -> Expected outputs of the function, used to calculate cost
 * @param learningRate -> Rate of Change in going down the gradiant
 *
 * Note: This does not actually feed in any values. It is assumed the function is already fed forward
 */
void NeuralNetwork::backPropagate(const std::vector<float> &expectedOutputs, const float learningRate) {
    unsigned networkSize = network.size();
    std::vector<std::vector<float>> errors(networkSize);

    for (int i = 0; i < network.back().size(); i++) {
        float error = (expectedOutputs[i] - network.back()[i]);
        errors[networkSize - 1].push_back(error * activationDerivative(network.back()[i], activations.back()));
    }

    for (unsigned i = networkSize - 2; i > 0; i--) {
        for (int j = 0; j < network[i].size(); j++) {
            float sumError = 0;
            for (int k = 0; k < network[i + 1].size(); k++) {
                sumError += weights[i][j][k] * errors[i + 1][k];
            }
            errors[i].push_back(sumError * activationDerivative(network[i][j], activations[i]));
        }
    }

    for (unsigned i = networkSize - 1; i > 0; i--) {
        for (int j = 0; j < network[i].size(); j++) {
            bias[i][j] += learningRate * errors[i][j];

            for (int k = 0; k < network[i - 1].size(); k++) {
                weights[i - 1][k][j] += learningRate * network[i - 1][k] * errors[i][j];
            }
        }
    }
}
/**
 * Simple indexed Activation
 *
 * @param x -> value
 * @param n -> index
 * @return activated value
 */
float NeuralNetwork::activation(const float x, const int n) {
    switch(n) {
        case 0:
            return 1/(1 + std::exp(-x));
        case 1:
            return std::tanh(x);
        case 2:
            return x > 0 ? x : 0;
        case 3:
            return x > 0 ? x : 0.001f * x;
        default:
            return x;
    }
}
/**
 * Simple indexed derivative activation
 *
 * @param x -> value
 * @param n -> index
 * @return derived activated value
 */
float NeuralNetwork::activationDerivative(const float x, const int n) {
    switch (n) {
        case 0: {
            float sigmoid = activation(x, n);
            return sigmoid * (1 - sigmoid);
        }
        case 1: {
            float tanh = activation(x, n);
            return 1 - (tanh * tanh);
        }
        case 2: {
            return x > 0 ? 1 : 0;
        }
        case 3: {
            return x > 0 ? 1 : 0.001f;
        }
        default: {
            return 1;
        }
    }
}
/**
 * Algorithm to get batches from training data
 *
 * @param inputData -> Input training data
 * @param outputData -> Output training data
 * @param batchSize -> Size of each batch
 * @return a vector of a batch of input, output data combinations
 */
std::vector<std::vector<std::vector<float>>> NeuralNetwork::getBatch(const std::vector<std::vector<float>> &inputData, const std::vector<std::vector<float>> &outputData, const int batchSize) {
    unsigned dataSize = inputData.size();
    std::vector<std::vector<std::vector<float>>> batch;

    std::vector<unsigned> indices(dataSize);
    for (unsigned i = 0; i < dataSize; ++i) {
        indices[i] = i;
    }

    std::random_device rd;
    std::mt19937 g(rd());
    shuffle(indices.begin(), indices.end(), g);

    for (int i = 0; i < batchSize && i < dataSize; ++i) {
        std::vector<std::vector<float>> inputOutputPair = {inputData[indices[i]], outputData[indices[i]]};
        batch.push_back(inputOutputPair);
    }

    return batch;
}


#endif //CPPNN_NEURALNETWORK_H
