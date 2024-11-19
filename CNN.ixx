#include <numeric>
#include <iomanip>
#include <thread>
#include <fstream>


export module CNN;

import ReshapeLayer;
import FullyConnectedLayer;
import ConvolutionalLayer;
import SigmoidFunction;
import Layer;
import <vector>;
import Cost;
import <iostream>;
import Matrix;
import <random>;
import <filesystem>;

import trainingData;

export typedef std::vector<std::vector<double>> d2Matrix;
export typedef std::vector<d2Matrix> d3Matrix;

export class Network 
{
private:
    std::vector<Layer*> layers;
    std::filesystem::path save_directory;

public:
    Network() = default;
    Network(std::filesystem::path directory)
        : save_directory(directory) {};
    ~Network();
    void train(const d3Matrix& X_train, const std::vector<int>& y_train, int epochs, double learning_rate);
    void saveWeightsAndBiases();
    void loadWeightsAndBiases();
    int testOnDataset(d3Matrix input, std::vector<int>& true_labels);
    int prediction(d2Matrix& input);
    void addLayer(Layer* layer);
};

void Network::addLayer(Layer* layer)
{
    layers.push_back(layer);
}

int Network::prediction(d2Matrix& input) 
{
    d3Matrix single_input = { input };
    for (auto& layer : layers)
        single_input = layer->forward(single_input);

    std::vector<double> pred_vector = Matrix::flatten(single_input);
    auto it = std::max_element(pred_vector.begin(), pred_vector.end());
    return std::distance(pred_vector.begin(), it);
}

Network::~Network() 
{
    for (Layer* layer : layers)
        delete layer;
}

void Network::train(const d3Matrix& X_train, const std::vector<int>& y_train, int epochs, double learning_rate)
{
    std::random_device rd;
    std::mt19937 g(rd());

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::vector<int> indices(X_train.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), g);

        double epoch_loss = 0.0;

        for (int i : indices) 
        {
            d3Matrix input = { X_train[i] };
            int digit = y_train[i];
            std::vector<int> true_label(10, 0); 
            true_label[digit] = 1;

            d3Matrix predicted = input;
            
            for (auto& layer : layers)
                predicted = layer->forward(predicted);

            double loss = CostFunction(true_label, Matrix::flatten(predicted));
            epoch_loss += loss;

            std::vector<double> loss_gradient = CostFunctionDerivative(true_label, Matrix::flatten(predicted));
            d3Matrix loss_gradient_3d = { {loss_gradient} };

            for (auto it = layers.rbegin(); it != layers.rend(); ++it)
                loss_gradient_3d = (*it)->backward(loss_gradient_3d, learning_rate);
            
        }
        std::cout << "Epoch " << epoch + 1 << " / " << epochs << " - Loss: " << epoch_loss / X_train.size() << std::endl;
    }
}

void Network::saveWeightsAndBiases() 
{
    std::filesystem::create_directories(save_directory);
    int convLayer_counter = 1;
    int denseLayer_counter = 1;

    for (Layer* layer : layers)
    {
        std::filesystem::path filepath;
        if (ConvolutionalLayer* Convolutional = dynamic_cast<ConvolutionalLayer*>(layer))
        {
            filepath = save_directory / ("convolutional" + std::to_string(convLayer_counter++) + ".txt");
            std::ofstream ofs(filepath, std::ios::trunc);
            if (ofs)
            {
                std::vector<d3Matrix> weights = Convolutional->getKernels();
                d3Matrix bias = Convolutional->getBias();
                for (int i = 0; i < weights.size(); ++i)
                    for (int j = 0; j < weights[i].size(); ++j)
                        for (int k = 0; k < weights[i][j].size(); ++k)
                            for (int l = 0; l < weights[i][j][k].size(); ++l)
                                ofs << std::to_string(weights[i][j][k][l]) << " ";

                for (int i = 0; i < bias.size(); ++i)
                    for (int j = 0; j < bias[i].size(); ++j)
                        for (int k = 0; k < bias[i][j].size(); ++k)
                            ofs << std::to_string(bias[i][j][k]) << " ";
            }

        }
        else if (Dense* dense = dynamic_cast<Dense*>(layer))
        {
            filepath = save_directory / ("dense" + std::to_string(denseLayer_counter++) + ".txt");
            std::ofstream ofs(filepath, std::ios::trunc);
            if (ofs) {
                d2Matrix weights = dense->getWeights();
                std::vector<double> biases = dense->getBiases();
                for (int i = 0; i < weights.size(); ++i)
                    for (int j = 0; j < weights[i].size(); ++j)
                        ofs << std::to_string(weights[i][j]) << " ";

                for (int i = 0; i < biases.size(); ++i)
                    ofs << std::to_string(biases[i]) << " ";
            }
        }
    }
}

void Network::loadWeightsAndBiases() 
{
    int convLayer_counter = 1;
    int denseLayer_counter = 1;
    for (Layer* layer : layers)
    {
        std::filesystem::path filepath;
        if (ConvolutionalLayer* convolutional = dynamic_cast<ConvolutionalLayer*>(layer))
        {
            filepath = save_directory / ("convolutional" + std::to_string(convLayer_counter++) + ".txt");
            std::ifstream ifs(filepath);
            if (ifs) 
            {
                std::vector<d3Matrix> weights = convolutional->getKernels();
                d3Matrix bias = convolutional->getBias();
                for (int i = 0; i < weights.size(); ++i)
                    for (int j = 0; j < weights[i].size(); ++j)
                        for (int k = 0; k < weights[i][j].size(); ++k)
                            for (int l = 0; l < weights[i][j][k].size(); ++l)
                                ifs >> weights[i][j][k][l];
                convolutional->setKernels(weights);

                for (int i = 0; i < bias.size(); ++i)
                    for (int j = 0; j < bias[i].size(); ++j)
                        for (int k = 0; k < bias[i][j].size(); ++k)
                            ifs >> bias[i][j][k];
                convolutional->setBias(bias);
            }
        }
        else if (Dense* dense = dynamic_cast<Dense*>(layer))
        {
            filepath = save_directory / ("dense" + std::to_string(denseLayer_counter++) + ".txt");
            std::ifstream ifs(filepath);
            if (ifs) {
                d2Matrix weights = dense->getWeights();
                std::vector<double> biases = dense->getBiases();
                for (int i = 0; i < weights.size(); ++i)
                    for (int j = 0; j < weights[i].size(); ++j)
                        ifs >> weights[i][j];
                dense->setWeights(weights);

                for (int i = 0; i < biases.size(); ++i)
                    ifs >> biases[i];
                dense->setBiases(biases);
            }
        }
    }
}

int Network::testOnDataset(d3Matrix input, std::vector<int>& true_labels)
{
    std::vector<int> pred_digits;
    int counter = 0;
    std::vector<int> indices(input.size());
    std::iota(indices.begin(), indices.end(), 0);

    for (int i : indices) 
    {
        d3Matrix single_input = { input[i] };
        int result = prediction(single_input[0]);
        pred_digits.emplace_back(result);
    }

    for (int i = 0; i < true_labels.size(); ++i)
        if (true_labels[i] == pred_digits[i])
            counter++;
    return counter;
}

