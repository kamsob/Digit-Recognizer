#include <vector>
#include <iostream>
#include <cmath>
export module Layer;

typedef std::vector<std::vector<std::vector<double>>> d3Matrix;
typedef std::vector<std::vector<double>> d2Matrix;

export class Layer 
{
protected:
	d3Matrix input;
	d3Matrix output;
	double learning_rate;
public:
	virtual d3Matrix forward(const d3Matrix &input) = 0;
	virtual d3Matrix backward(const d3Matrix &output, double learning_rate) = 0;
};