export module FullyConnectedLayer;

import Random;
import Matrix;
import <random>;
import Layer;
import <vector>;
import <thread>;
import <ranges>;

typedef std::vector<std::vector<std::vector<double>>> d3Matrix;
typedef std::vector<std::vector<double>> d2Matrix;

export class Dense : public Layer {
private:
    d2Matrix weights;
    std::vector<double> biases;

public:
    Dense(int input_size, int output_size);

    d3Matrix forward(const d3Matrix& input) override;

    d3Matrix backward(const d3Matrix& gradient, double learning_rate) override;

    d2Matrix getWeights() { return weights; };
    std::vector<double> getBiases() { return biases; };
    void setWeights(const d2Matrix& w) { weights = w; }
    void setBiases(const std::vector<double>& b) { biases = b; }
};

Dense::Dense(int input_size, int output_size)
    :weights(Random::generateRandomD2Matrix(output_size, input_size)), 
    biases(Random::generateRandomVector(output_size)) {}

d3Matrix Dense::forward(const d3Matrix& input){
    this->input = input;
    d3Matrix output(1, d2Matrix(1, biases));

    int num_threads = 10;
    int updates_per_thread = weights.size() / num_threads;
    std::vector<std::thread>threads;
    for (int t = 0; t < num_threads; ++t)
    {
        int start = t * updates_per_thread;
        int end = (t == num_threads - 1) ? weights.size() : (t + 1) * updates_per_thread;
        threads.emplace_back([&, start, end]()
        {
            for (int i = start; i < end; ++i)
                for (int j = 0; j < weights[i].size(); ++j)
                    output[0][0][i] += weights[i][j] * input[0][0][j];
        });
    }
    for (auto& t : threads)
        t.join();
    return output;
}

d3Matrix Dense::backward(const d3Matrix& output_gradient, double learning_rate)
{
    std::vector<double> input_gradient(weights[0].size(), 0);
    int num_threads = 10;
    int updates_per_thread = weights.size() / num_threads;
    std::vector<std::thread>threads;
    for (int t = 0; t < num_threads; ++t) 
    {
        int start = t * updates_per_thread;
        int end = (t == num_threads - 1) ? weights.size() : (t + 1) * updates_per_thread;
        threads.emplace_back([&, start, end]()
        {
            for (int i = start; i < end; ++i) 
            {
                for (int j = 0; j < weights[i].size(); ++j)
                {
                    input_gradient[j] += weights[i][j] * output_gradient[0][0][i];
                    weights[i][j] -= learning_rate * output_gradient[0][0][i] * input[0][0][j];
                }
                biases[i] -= learning_rate * output_gradient[0][0][i];
            }
        });
    }
    for (auto& t : threads)
        t.join();
    return d3Matrix(1, d2Matrix(1, input_gradient));
}
