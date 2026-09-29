/*
 * util.h
 *
 *  Created on: 06/03/2015
 *      Author: pedroa
 */

#ifndef UTIL_H_
#define UTIL_H_

#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <random>
#include <numeric>
#include <algorithm>

namespace util
{
    class Dataset
    {
    public:
        Dataset(int nOfInputs, int nOfOutputs, int nOfPatterns, std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> outputs)
        {
            this->nOfInputs = nOfInputs;
            this->nOfOutputs = nOfOutputs;
            this->nOfPatterns = nOfPatterns;
            this->inputs = inputs;
            this->outputs = outputs;
        }

        Dataset() = default;

        int nOfInputs;                            /* Number of inputs */
        int nOfOutputs;                           /* Number of outputs */
        int nOfPatterns;                          /* Number of patterns */
        std::vector<std::vector<double>> inputs;  /* Matrix with the inputs of the problem */
        std::vector<std::vector<double>> outputs; /* Matrix with the outputs of the problem */
    };

    // Obtain an integer random number in the range [low, high]
    int randomInt(int low, int high);

    // Obtain a real random number in the range [low, high]
    double randomDouble(double low, double high);

    // Read a dataset from a file name and return it
    Dataset readData(const std::string &fileName);

    // Print the dataset to the specified output stream
    void printDataset(const Dataset &dataset, int len = 0, std::ostream &os = std::cout);

    // Generate a vector of unique random integers in the range [min, max]
    inline std::vector<int> integerRandomVectorWithoutRepeating(int min, int max, int howMany)
    {
        if (min > max || howMany <= 0)
        {
            return {};
        }

        int total = max - min + 1;
        if (howMany > total)
        {
            howMany = total;
        }

        std::vector<int> numbers(total);
        std::iota(numbers.begin(), numbers.end(), min);

        static std::random_device rd;
        static std::mt19937 gen(rd());

        for (int i = 0; i < howMany; ++i)
        {
            std::uniform_int_distribution<int> dist(i, total - 1);
            int selectedIndex = dist(gen);
            std::swap(numbers[i], numbers[selectedIndex]);
        }

        numbers.resize(howMany);
        return numbers;
    }

    // Alias to retain compatibility with legacy typo in function name
    inline std::vector<int> integerRandomVectoWithoutRepeating(int min, int max, int howMany)
    {
        return integerRandomVectorWithoutRepeating(min, max, howMany);
    }

    // Transform a scalar x by scaling it to a given range [minAllowed, maxAllowed] considering the min
    // and max values of the feature in the dataset (minData and maxData).
    double minMaxScaler(double x, double minAllowed, double maxAllowed, double minData, double maxData)
    {
        if (maxData == minData)
        {
            return minAllowed;
        }
        return minAllowed + ((x - minData) * (maxAllowed - minAllowed)) / (maxData - minData);
    }

    // Scale the dataset inputs to a given range [minAllowed, maxAllowed] considering the min
    // and max values of the feature in the dataset (minData and maxData).
    void minMaxScalerDataSetInputs(Dataset &dataset, double minAllowed, double maxAllowed,
                                   const std::vector<double> &minData, const std::vector<double> &maxData);

    // Scale the dataset outputs to a given range [minAllowed, maxAllowed] considering the min
    // and max values of the output in the dataset (minData and maxData).
    void minMaxScalerDataSetOutputs(Dataset &dataset, double minAllowed, double maxAllowed,
                                    const std::vector<double> &minData, const std::vector<double> &maxData);

    // Extract maximum and minimum values from a matrix of doubles
    void obtainMaxMinValuesFromMatrix(const std::vector<std::vector<double>> &matrix,
                                      std::vector<double> &obtainedMins,
                                      std::vector<double> &obtainedMaxs);

    // Overload returning a pair of vectors: {minValues, maxValues}
    std::pair<std::vector<double>, std::vector<double>> obtainMaxMinValuesFromMatrix(
        const std::vector<std::vector<double>> &matrix);

} // namespace util

#endif /* UTIL_H_ */
