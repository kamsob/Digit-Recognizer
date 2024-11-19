export module ReshapeLayer;

import Matrix;
import Layer;
import <vector>;
import <iostream>;
typedef std::vector<std::vector<std::vector<double>>> d3Matrix;
typedef std::vector<std::vector<double>> d2Matrix;

export class Reshape : public Layer {
private:
    int output_depth, output_height, output_width;

public:
    Reshape(int new_depth, int new_height, int new_width) 
        : output_depth(new_depth), output_height(new_height), output_width(new_width){}

    d3Matrix forward(const d3Matrix& input) override;
    d3Matrix backward(const d3Matrix& gradient, double learning_rate) override;
};

d3Matrix Reshape::forward(const d3Matrix& input)
{
    this->input = input;
    output = d3Matrix(output_depth, d2Matrix(output_height, std::vector<double>(output_width, 0)));
    std::vector<double> flatten_vector = Matrix::flatten(input);
    
    int index = 0;
    for (int d = 0; d < output_depth; ++d)
        for (int h = 0; h < output_height; ++h)
            for (int w = 0; w < output_width; ++w)
                output[d][h][w] = flatten_vector[index++];
    return output;
}

d3Matrix Reshape::backward(const d3Matrix& gradient, double learning_rate){
    d3Matrix reshaped_gradient(input.size(), d2Matrix(input[0].size(), std::vector<double>(input[0][0].size(),0)));
    std::vector<double> flatten_vector = Matrix::flatten(gradient);
    int index = 0;
    for (int d = 0; d < reshaped_gradient.size(); ++d)
        for (int h = 0; h < reshaped_gradient[0].size(); ++h)
            for (int w = 0; w < reshaped_gradient[0][0].size(); ++w) 
            {
                reshaped_gradient[d][h][w] = flatten_vector[index];
                ++index;
            }
    return reshaped_gradient;
}