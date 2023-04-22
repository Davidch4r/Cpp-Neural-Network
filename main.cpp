#include <iostream>
#include <string>
#include <vector>
#include<map>
#include <random>
#include <numeric>
#include <cmath>
#include <ctime>

using namespace std;

class NeuralNetwork {
public:
    NeuralNetwork(const vector<int> &layers, const vector<string> &activationStrings) {
        srand(time(NULL));
        unsigned layerSize = layers.size();
        if (layerSize != activationStrings.size())
            throw invalid_argument("The layers and activation vectors must be constant");
        activations.reserve(layerSize);
        network.reserve(layerSize);
        bias.reserve(layerSize);
        weights.reserve(layerSize);
        for (int i = 0; i < layerSize; i++) {
            vector<float> newNetLayer(layers[i]);
            vector<float> newBiasLayer(layers[i]);
            vector<vector<float>> newWeightsLayer(layers[i]);
            for (int j = 0; j < layers[i]; j++) {
                vector<float> connections;
                if (i < layerSize - 1) {
                    connections.resize(layers[i + 1]);
                    for (int k = 0; k < layers[i + 1]; k++) {
                        connections[k] = static_cast<float>(rand()) / RAND_MAX * 2 - 1;
                    }
                }
                newNetLayer[j] = 0;
                newBiasLayer[j] = static_cast<float>(rand()) / RAND_MAX * 2 - 1;
                newWeightsLayer[j] = connections;
            }
            activations.push_back(activationMap[activationStrings[i]]);
            network.push_back(newNetLayer);
            bias.push_back(newBiasLayer);
            weights.push_back(newWeightsLayer);
        }
    }
    void Learn(const vector<vector<float>> &inputData, const vector<vector<float>> &outputData, const int &epochs, const float &learningRate, const float batchSizePercent) {
        if (inputData.size() != outputData.size())
            throw invalid_argument("InputData Size must equal OutputData size");
        if (batchSizePercent > 1)
            throw length_error("Batch Size Percentage cannot be about 1");
        cout << "Learning Started... " << endl;
        unsigned dataSize = inputData.size();
        int batchSize = static_cast<int>(dataSize * batchSizePercent);
        int divisible10 = epochs / 10;
        for (int i = 0; i < epochs; i++) {
            vector<vector<vector<float>>> batch = getBatch(inputData, outputData, batchSize);
            for (int j = 0; j < batchSize; j++) {
                feedForward(batch[j][0]);
                backPropagate(batch[j][1], learningRate);
            }
            if (i % divisible10 == 0) {
                float totalError = 0;
                for (int j = 0; j < batchSize; j++) {
                    vector<float> outputs = getOutputs(batch[j][0]);
                    vector<float> expectedOutputs = batch[j][1];
                    for (unsigned k = 0; k < expectedOutputs.size(); k++) {
                        totalError += 0.5 * pow(expectedOutputs[k] - outputs[k], 2);
                    }
                }
                float averageError = totalError / static_cast<float>(batchSize);
                cout<< "Average Error iter " << i << " -> " << averageError << endl;
            }
        }
        cout << "Learning Ended" << endl;
    }

    void Test(const vector<vector<float>> &inputData, const vector<vector<float>> &outputData, char test) {
        if (inputData.size() != outputData.size())
            throw invalid_argument("InputData Size must equal OutputData size");
        cout << "Testing..." << endl;
        unsigned dataSize = inputData.size();
        unsigned outputDataSize = outputData[0].size();
        int totalSize = dataSize * outputDataSize;
        int totalCorrect = 0;

        for (int i = 0; i < dataSize; i++) {
            vector<float> answers = getOutputs(inputData[i]);

            switch(test) {
                case 'A': {
                    for (int j = 0; j < outputDataSize; j++) {
                        answers[j] = round(answers[j]);
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
                        expSum += exp(answers[j]);
                    }
                    for (int j = 0; j < outputDataSize; j++) {
                        answers[j] = exp(answers[j]) / expSum;
                    }

                    break;
                }
                default: { }
            }

            for (int j = 0; j < outputDataSize; j++) {
                float expectedAnswer = outputData[i][j];
                float answer = answers[j];
                if (test == 'B' || test == 'C')
                    answer = round(answer);
                if (answer == expectedAnswer)
                    totalCorrect++;
            }

        }

        float percentage = 100 * static_cast<float>(totalCorrect) / static_cast<float>(totalSize);

        cout << "--------------\nPercentage Correct: " << percentage << "% \nTotal Correct: " << totalCorrect << "/" << totalSize << "\n--------------" << endl;
    }
    vector<float> getOutputs(const vector<float> &inputVals) {
        if (inputVals.size() != network.begin()->size())
            throw invalid_argument("InputVals size must equal the the size of the network inputs");
        feedForward(inputVals);
        unsigned outputSize = network.back().size();
        vector<float> outputs(outputSize);
        for (int i = 0; i < outputSize; i++) {
            outputs[i] = network.back()[i];
        }
        return outputs;
    }

private:
    map<string, int> activationMap = {
        {"sigmoid", 0},
        {"tanh", 1},
        {"relu", 2},
        {"leakyRelu", 3},
        {"input", 4}
    };
    vector<int> activations;
    vector<vector<float>> network;
    vector<vector<float>> bias;
    vector<vector<vector<float>>> weights;

    void feedForward(const vector<float> &inputVals) {
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
    void backPropagate(const vector<float> &expectedOutputs, const float learningRate) {
        unsigned networkSize = network.size();
        vector<vector<float>> errors(networkSize);

        for (int i = 0; i < network.back().size(); i++) {
            float error = (expectedOutputs[i] - network.back()[i]);
            errors[networkSize - 1].push_back(error * activationDerivative(network.back()[i], activations.back()));
        }

        for (int i = networkSize - 2; i > 0; i--) {
            for (int j = 0; j < network[i].size(); j++) {
                float sumError = 0;
                for (int k = 0; k < network[i + 1].size(); k++) {
                    sumError += weights[i][j][k] * errors[i + 1][k];
                }
                errors[i].push_back(sumError * activationDerivative(network[i][j], activations[i]));
            }
        }

        for (int i = networkSize - 1; i > 0; i--) {
            for (int j = 0; j < network[i].size(); j++) {
                bias[i][j] += learningRate * errors[i][j];

                for (int k = 0; k < network[i - 1].size(); k++) {
                    weights[i - 1][k][j] += learningRate * network[i - 1][k] * errors[i][j];
                }
            }
        }
    }
    static float activation(const float x, const int n) {
        switch(n) {
            case 0:
                return 1/(1 + exp(-x));
            case 1:
                return tanh(x);
            case 2:
                return x > 0 ? x : 0;
            case 3:
                return x > 0 ? x : 0.001f * x;
            default:
                return x;
        };
    }
    static float activationDerivative(const float x, const int n) {
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
        };
    }
    static vector<vector<vector<float>>> getBatch(const vector<vector<float>> &inputData, const vector<vector<float>> &outputData, const int batchSize) {
        unsigned dataSize = inputData.size();
        vector<vector<vector<float>>> batch;

        vector<unsigned> indices(dataSize);
        for (unsigned i = 0; i < dataSize; ++i) {
            indices[i] = i;
        }

        random_device rd;
        mt19937 g(rd());
        shuffle(indices.begin(), indices.end(), g);

        for (int i = 0; i < batchSize && i < dataSize; ++i) {
            vector<vector<float>> inputOutputPair = {inputData[indices[i]], outputData[indices[i]]};
            batch.push_back(inputOutputPair);
        }

        return batch;
    }



};

int main() {
    const vector<int> layers = {3, 10, 10, 8, 5, 3,  1};
    const vector<string> activations = {"input", "relu", "relu", "relu", "relu", "relu", "sigmoid"};
    const vector<vector<float>> inputs = {
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
    const vector<vector<float>> expectedOutputs = {
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
        cout << "Input: ";
        for (auto &inputVal : input) {
            cout << inputVal << ", ";
        }
        cout << "\nOutput: ";
        vector<float> output = net.getOutputs(input);
        for (auto &outputVal : output) {
            cout << outputVal << ", ";
        }
        cout << endl;
    }
    net.Test(inputs, expectedOutputs, 'A');
    return 0;
}
