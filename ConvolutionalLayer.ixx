export module ConvolutionalLayer;

import Random;
import Layer;
import Matrix;
import <vector>;
import <thread>;

typedef std::vector<std::vector<std::vector<double>>> d3Matrix;
typedef std::vector<std::vector<double>> d2Matrix;

export class ConvolutionalLayer : public Layer {
private:
    std::vector<d3Matrix> kernels; // (input_depth, num_filters, kernel_size, kernel_size)
    d3Matrix bias;
    int input_depth, input_size, output_size, kernel_size, stride;

public:
    ConvolutionalLayer(int input_depth, int input_size, int kernel_size, int num_filters, int stride);

    d3Matrix forward(const d3Matrix& input);
    d3Matrix backward(const d3Matrix& gradient, double learning_rate);
    std::vector<d3Matrix> getKernels() { return kernels; };
    d3Matrix getBias() { return bias; }
    void setKernels(const std::vector<d3Matrix>& k) { kernels = k; }
    void setBias(const d3Matrix& b) { bias = b; }
};

ConvolutionalLayer::ConvolutionalLayer(int input_depth, int input_size, int kernel_size, int num_filters, int stride)
    : kernels(num_filters, Random::generateRandomD3Matrix(input_depth, kernel_size, kernel_size)),
    bias(Random::generateRandomD3Matrix(num_filters, (input_size-kernel_size)/stride+1, (input_size-kernel_size)/stride+1)), 
    input_size(input_size), output_size((input_size - kernel_size)/stride+1), kernel_size(kernel_size), input_depth(input_depth), stride(stride){}

d3Matrix ConvolutionalLayer::forward(const d3Matrix& input)
{
    this->input = input;
    int num_filters = kernels.size();

    d3Matrix output(num_filters, d2Matrix(output_size, std::vector<double>(output_size, 0)));

    std::vector<std::thread>threads;

    for (int f = 0; f < num_filters; ++f)
        threads.emplace_back([&, f]() {
        for (int d = 0; d < input_depth; ++d)
            output[f] = Matrix::CrossCorrelation(input[d], kernels[f][d], bias[f], stride);
            });
    for (std::thread& t : threads)
        t.join();
    return output;
}

d3Matrix ConvolutionalLayer::backward(const d3Matrix& output_gradient, double learning_rate)
{
    std::vector<d3Matrix> kernels_gradient(kernels.size(), d3Matrix(input_depth, d2Matrix(kernel_size, std::vector<double>(kernel_size, 0))));
    d3Matrix input_gradient(input.size(), d2Matrix(input_size, std::vector<double>(input_size, 0)));

    std::vector<std::thread>threads;

    for (int f = 0; f < kernels.size(); ++f) {
        threads.emplace_back([&, f]() {for (int d = 0; d < input_depth; ++d)
        {
            int output_gradient_size = input_size - output_gradient[f].size() + 1;
            kernels_gradient[f][d] = Matrix::CrossCorrelation(input[d], output_gradient[f], d2Matrix(output_gradient_size, std::vector<double>(output_gradient_size, 0)), 1);
            Matrix::Add(input_gradient[d], Matrix::FullConvolution(output_gradient[f], kernels[f][d], stride), true, 1);

            Matrix::Add(kernels[f][d], kernels_gradient[f][d], false, learning_rate);
        }
            Matrix::Add(bias[f], output_gradient[f], false, learning_rate);
        });
    }
    for (std::thread& t : threads)
        t.join();

    return input_gradient;
}