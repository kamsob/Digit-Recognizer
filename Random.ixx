export module Random;
import <random>;
import <vector>;

typedef std::vector<std::vector<double>> d2Matrix;
typedef std::vector<d2Matrix> d3Matrix;

export namespace Random
{
    inline double generateRandomNumber();
    std::vector<double> generateRandomVector(int width);
    d2Matrix generateRandomD2Matrix(int height, int width);
    d3Matrix generateRandomD3Matrix(int depth, int height, int width);
}

double Random::generateRandomNumber()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<> dis(0.0, 1.0);

    return dis(gen);
}

std::vector<double> Random::generateRandomVector(int width)
{
    std::vector<double> vector(width, 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    double stddev = std::sqrt(2.0 / width);
    std::normal_distribution<> dis(0, stddev);

    for (int w = 0; w < width; ++w)
        vector[w] = dis(gen);
    return vector;
}

d2Matrix Random::generateRandomD2Matrix(int height, int width) 
{
    d2Matrix matrix(height, std::vector<double>(width, 0));
    for (int h = 0; h < height; ++h)
        matrix[h] = generateRandomVector(width);
    return matrix;
}

d3Matrix Random::generateRandomD3Matrix(int depth, int height, int width) 
{
    d3Matrix matrix(depth, d2Matrix(height, std::vector<double>(width, 0)));
    for (int d = 0; d < depth; ++d)
        matrix[d] = generateRandomD2Matrix(height, width);
    return matrix;
}
