export module Matrix;

import <vector>;
import <algorithm>;
import <ranges>;

export typedef std::vector<std::vector<double>> d2Matrix;
export typedef std::vector<d2Matrix>d3Matrix;

export namespace Matrix
{
	d2Matrix CrossCorrelation(const d2Matrix& matrix, const d2Matrix& kernel, const d2Matrix& bias, int stride);
	d2Matrix FullConvolution(const d2Matrix& matrix, const d2Matrix& kernel, int stride);
	void Add(d2Matrix& m1, const d2Matrix& m2, bool sign, double multiplier);
	std::vector<double> flatten(const d3Matrix& matrix);
};

std::vector<double> Matrix::flatten(const d3Matrix& matrix) 
{
	std::vector<double> flat_vector;
	for (const auto& layer : matrix)
		for (const auto& row : layer)
			std::ranges::copy(row, std::back_inserter(flat_vector));
	return flat_vector;
}

void Matrix::Add(d2Matrix& m1, const d2Matrix& m2, bool sign, double multiplier)//sign - true for +, false for -| add/subtract 
{
	if (!sign)
		multiplier = multiplier * (-1);
	if (m1.size() == m2.size())
		for (int i = 0; i < m1.size(); ++i)
			std::ranges::views::transform([multiplier](double& a, double b) {
                a += b * multiplier;
            });
}

inline double kernelMultiplication(const d2Matrix& matrix, const d2Matrix& kernel, const int x, const int y)
{
	double output = 0;
	for (int i = 0; i < kernel.size(); ++i)
		for (int j = 0; j < kernel.size(); ++j)
			output += matrix[i + x][j + y] * kernel[i][j];
	return output;
}

d2Matrix Matrix::CrossCorrelation(const d2Matrix& matrix, const d2Matrix& kernel, const d2Matrix& bias, int stride)
{
	d2Matrix output = bias;
	int strides = (matrix.size() - kernel.size())/stride+1;
	for (int i = 0; i*stride < strides; ++i)
		for (int j = 0; j*stride < strides; ++j)
			output[i][j] += kernelMultiplication(matrix, kernel, i*stride, j*stride);
	return output;
}

inline d2Matrix kernelRotation(const d2Matrix& kernel) 
{
	int size = kernel.size();
	d2Matrix output(size, std::vector<double>(size,0));
	for (int i = 0; i < size; ++i)
		for (int j = 0; j < size; ++j)
			output[i][j] = kernel[size - 1 - i][size - 1 - j];
	return output;
}

d2Matrix Matrix::FullConvolution(const d2Matrix& matrix, const d2Matrix& kernel, int stride) {
	int padded_size = matrix.size() + 2 * (kernel.size() - 1);
	d2Matrix padded_matrix(padded_size, std::vector<double>(padded_size, 0));

	for (int i = 0; i < matrix.size(); ++i)
		for (int j = 0; j < matrix[0].size(); ++j)
			padded_matrix[i + kernel.size() - 1][j + kernel.size() - 1] = matrix[i][j];

	int output_size = (padded_size - kernel.size()) / stride + 1;
	d2Matrix output(output_size, std::vector<double>(output_size, 0));
	d2Matrix rotated_kernel = kernelRotation(kernel);

	for (int i = 0; i < output_size; ++i)
		for (int j = 0; j < output_size; ++j)
			output[i][j] = kernelMultiplication(padded_matrix, rotated_kernel, i * stride, j * stride);
	return output;
}