export module SigmoidFunction;

import<cmath>;
import Layer;
import <vector>;
import <ranges>;

typedef std::vector<std::vector<std::vector<double>>> d3Matrix;
typedef std::vector<std::vector<double>> d2Matrix;

export class Sigmoid : public Layer {
private:
    inline double sigmoid(double x);
    inline double sigmoid_derivative(double x);
public:
    d3Matrix forward(const d3Matrix& input) override;

    d3Matrix backward(const d3Matrix& gradient, double) override;
};

double Sigmoid::sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double Sigmoid::sigmoid_derivative(double x) {
    double sig = sigmoid(x);
    return sig * (1.0 - sig);
}

d3Matrix Sigmoid::forward(const d3Matrix& input)
{
    this->input = input;
    output = d3Matrix(input.size(), d2Matrix(input[0].size(), std::vector<double>(input[0][0].size(), 0)));
    for (int batch = 0; batch < input.size(); ++batch)
        for (int i = 0; i < input[batch].size(); ++i) 
        {
            std::ranges::transform_view view = input[batch][i] | std::ranges::views::transform([&](double x) {return sigmoid(x); });
            std::ranges::copy(view, output[batch][i].begin());
        }
    return output;
}

d3Matrix Sigmoid::backward(const d3Matrix& gradient, double learning_rate)
{
    d3Matrix input_gradient = d3Matrix(gradient.size(), d2Matrix(gradient[0].size(), std::vector<double>(gradient[0][0].size(), 0)));
    for (int batch = 0; batch < gradient.size(); ++batch)
        for (int i = 0; i < gradient[0].size(); ++i)
        {
            auto gradient_it = gradient[batch][i].begin();
            auto input_it = this->input[batch][i].begin();
            auto input_grad_it = input_gradient[batch][i].begin();
            std::ranges::transform_view view = gradient[batch][i] | std::ranges::views::transform([&](double g) 
            {
                double inp = *input_it++;
                return g * sigmoid_derivative(inp);
            });
            std::ranges::copy(view, input_gradient[batch][i].begin());
        }
    return input_gradient;
}
