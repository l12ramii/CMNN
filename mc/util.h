/**
 * @file util.h
 * @brief Utility functions and data structures for dataset handling, scaling, and random number generation.
 * @author pedroa
 * @date 06/03/2015
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
#include <cmath>

/**
 * @namespace util
 * @brief Namespace containing helper utilities for neural network data processing, math scaling, and random number generation.
 */
namespace util
{
    /**
     * @class Dataset
     * @brief Represents a machine learning dataset containing input patterns and target output patterns.
     */
    class Dataset
    {
    public:
        /**
         * @brief Constructs a Dataset with specified dimensions and data matrices.
         *
         * @param[in] nOfInputs Number of input features per pattern.
         * @param[in] nOfOutputs Number of output targets per pattern.
         * @param[in] nOfPatterns Total number of patterns (samples) in the dataset.
         * @param[in] inputs Matrix of input features of size (nOfPatterns x nOfInputs).
         * @param[in] outputs Matrix of target outputs of size (nOfPatterns x nOfOutputs).
         */
        Dataset(int nOfInputs, int nOfOutputs, int nOfPatterns, std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> outputs)
        {
            this->nOfInputs = nOfInputs;
            this->nOfOutputs = nOfOutputs;
            this->nOfPatterns = nOfPatterns;
            this->inputs = inputs;
            this->outputs = outputs;
        }

        /**
         * @brief Default constructor for Dataset.
         */
        Dataset() = default;

        int nOfInputs{0};                         /**< Number of inputs per pattern */
        int nOfOutputs{0};                        /**< Number of outputs per pattern */
        int nOfPatterns{0};                       /**< Total number of patterns in the dataset */
        std::vector<std::vector<double>> inputs;  /**< Matrix with the inputs of the problem [pattern_index][input_index] */
        std::vector<std::vector<double>> outputs; /**< Matrix with the outputs of the problem [pattern_index][output_index] */
    };

    /**
     * @brief Sets the seed for the pseudo-random number generator.
     *
     * @param[in] seed Integer seed value for the random number generator.
     */
    void setSeed(int seed);

    /**
     * @brief Obtains a uniformly distributed random integer in the range [low, high].
     *
     * @param[in] low Minimum integer value (inclusive).
     * @param[in] high Maximum integer value (inclusive).
     * @return A random integer in the interval [low, high].
     */
    int randomInt(int low, int high);

    /**
     * @brief Obtains a uniformly distributed random real number in the range [low, high].
     *
     * @param[in] low Minimum real value (inclusive).
     * @param[in] high Maximum real value (inclusive).
     * @return A random double in the interval [low, high].
     */
    double randomDouble(double low, double high);

    /**
     * @brief Reads a dataset from a formatted text file.
     *
     * The file begins with a header line containing: <nOfInputs> <nOfOutputs> <nOfPatterns>,
     * followed by lines containing inputs and target outputs for each pattern.
     *
     * @param[in] fileName Path to the dataset file to be read.
     * @return The parsed Dataset object containing inputs and outputs.
     * @throw std::runtime_error If the file cannot be opened or if an error occurs while reading.
     */
    Dataset readData(const std::string &fileName);

    /**
     * @brief Prints the dataset to the specified output stream.
     *
     * @param[in] dataset The Dataset to print.
     * @param[in] len Number of patterns to print (default 0 means all patterns).
     * @param[in,out] os Output stream to write to (defaults to std::cout).
     */
    void printDataset(const Dataset &dataset, int len = 0, std::ostream &os = std::cout);

    /**
     * @brief Generates a vector of unique random integers in the range [min, max] without repetition.
     *
     * @param[in] min Minimum integer value (inclusive).
     * @param[in] max Maximum integer value (inclusive).
     * @param[in] howMany Number of unique integers to sample.
     * @return std::vector<int> containing the unique sampled integers.
     */
    std::vector<int> integerRandomVectorWithoutRepeating(int min, int max, int howMany);

    /**
     * @brief Alias to retain compatibility with legacy typo in function name.
     *
     * Calls integerRandomVectorWithoutRepeating(min, max, howMany).
     *
     * @param[in] min Minimum integer value (inclusive).
     * @param[in] max Maximum integer value (inclusive).
     * @param[in] howMany Number of unique integers to sample.
     * @return std::vector<int> containing the unique sampled integers.
     */
    inline std::vector<int> integerRandomVectoWithoutRepeating(int min, int max, int howMany)
    {
        return integerRandomVectorWithoutRepeating(min, max, howMany);
    }

    /**
     * @brief Transforms a scalar x by scaling it to a given range [minAllowed, maxAllowed].
     *
     * Considers the min and max values of the feature in the dataset (minData and maxData).
     * Formula: minAllowed + ((x - minData) * (maxAllowed - minAllowed)) / (maxData - minData)
     * If maxData equals minData, returns minAllowed.
     *
     * @param[in] x The scalar value to scale.
     * @param[in] minAllowed Lower bound of the target range.
     * @param[in] maxAllowed Upper bound of the target range.
     * @param[in] minData Minimum value of the feature in the dataset.
     * @param[in] maxData Maximum value of the feature in the dataset.
     * @return The scaled value in the range [minAllowed, maxAllowed].
     */
    inline double minMaxScaler(double x, double minAllowed, double maxAllowed, double minData, double maxData)
    {
        if (maxData == minData)
        {
            return minAllowed;
        }
        return minAllowed + ((x - minData) * (maxAllowed - minAllowed)) / (maxData - minData);
    }

    /**
     * @brief Scales the dataset inputs to a given range [minAllowed, maxAllowed] in-place.
     *
     * Considers the min and max values of each feature in the dataset (minData and maxData).
     *
     * @param[in,out] dataset Dataset whose inputs will be scaled in-place.
     * @param[in] minAllowed Lower bound of the target range.
     * @param[in] maxAllowed Upper bound of the target range.
     * @param[in] minData Vector containing the minimum value for each input feature.
     * @param[in] maxData Vector containing the maximum value for each input feature.
     */
    void minMaxScalerDataSetInputs(Dataset &dataset, double minAllowed, double maxAllowed,
                                   const std::vector<double> &minData, const std::vector<double> &maxData);

    /**
     * @brief Scales the dataset outputs to a given range [minAllowed, maxAllowed] in-place.
     *
     * Considers the min and max values of each output in the dataset (minData and maxData).
     *
     * @param[in,out] dataset Dataset whose outputs will be scaled in-place.
     * @param[in] minAllowed Lower bound of the target range.
     * @param[in] maxAllowed Upper bound of the target range.
     * @param[in] minData Vector containing the minimum value for each output target.
     * @param[in] maxData Vector containing the maximum value for each output target.
     */
    void minMaxScalerDataSetOutputs(Dataset &dataset, double minAllowed, double maxAllowed,
                                    const std::vector<double> &minData, const std::vector<double> &maxData);

    /**
     * @brief Extracts column-wise maximum and minimum values from a 2D matrix of doubles.
     *
     * @param[in] matrix 2D matrix of values (rows x columns).
     * @param[out] obtainedMins Vector filled with the minimum value found in each column.
     * @param[out] obtainedMaxs Vector filled with the maximum value found in each column.
     * @throw std::runtime_error If the matrix is empty or has empty rows.
     */
    void obtainMaxMinValuesFromMatrix(const std::vector<std::vector<double>> &matrix,
                                      std::vector<double> &obtainedMins,
                                      std::vector<double> &obtainedMaxs);

    /**
     * @brief Extracts column-wise maximum and minimum values from a 2D matrix of doubles.
     *
     * Overload returning a pair of vectors: {minValues, maxValues}.
     *
     * @param[in] matrix 2D matrix of values (rows x columns).
     * @return std::pair<std::vector<double>, std::vector<double>> Pair containing {minValues, maxValues} for each column.
     * @throw std::runtime_error If the matrix is empty or has empty rows.
     */
    std::pair<std::vector<double>, std::vector<double>> obtainMaxMinValuesFromMatrix(
        const std::vector<std::vector<double>> &matrix);

    /**
     * @brief Computes the standard logistic sigmoid activation function.
     *
     * Formula: \f$\sigma(x) = \frac{1}{1 + e^{-x}}\f$
     * Maps any real-valued net input into the range (0, 1).
     *
     * @param[in] x Net input value (activation potential).
     * @return Output activation value in the range (0, 1).
     */
    inline double sigmoid(double x)
    {
        return 1.0 / (1.0 + std::exp(-x));
    }
    
} // namespace util

#endif /* UTIL_H_ */
