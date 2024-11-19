export module Cost;

import <vector>;
import <cmath>;

//binary cross entropy
export double CostFunction(const std::vector<int>& trueDigits, const std::vector<double>& calcDigits)
{
    double epsilon = 1e-15; // Small value to avoid log(0)
    double error = 0.0;
    for (int i = 0; i < trueDigits.size(); ++i)
        error += -trueDigits[i] * std::log(std::max(calcDigits[i], epsilon)) - (1 - trueDigits[i]) * std::log(std::max(1 - calcDigits[i], epsilon));
    return error / trueDigits.size();
}

export std::vector<double> CostFunctionDerivative(const std::vector<int>& trueDigits, const std::vector<double>& calcDigits)
{
    std::vector<double> gradient(trueDigits.size(), 0.0);
    double epsilon = 1e-15; // Small value to avoid division by zero
    for (int i = 0; i < trueDigits.size(); ++i)
        gradient[i] = ((1 - trueDigits[i]) / std::max(1 - calcDigits[i], epsilon) - trueDigits[i] / std::max(calcDigits[i], epsilon)) / trueDigits.size();
    return gradient;
}