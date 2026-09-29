/*********************************************************************
* File  : MultilayerPerceptron.h
* Date  : 2020
*********************************************************************/

#ifndef MULTILAYERPERCEPTRON_H_
#define MULTILAYERPERCEPTRON_H_

#include <vector>
#include <string>
#include <iostream>
#include <utility>

#include "util.h"

namespace mc
{

// Suggested structures
// ---------------------
struct Neuron
{
    double out{0.0};                     /* Output produced by the neuron (out_j^h) */
    double delta{0.0};                   /* Derivative of the output produced by the neuron (delta_j^h) */
    std::vector<double> w;               /* Input weight vector (w_{ji}^h) */
    std::vector<double> deltaW;          /* Change to be applied to every weight (\Delta_{ji}^h (t)) */
    std::vector<double> lastDeltaW;      /* Last change applied to every weight (\Delta_{ji}^h (t-1)) */
    std::vector<double> wCopy;           /* Copy of the input weights */
};

struct Layer
{
    int nOfNeurons{0};                   /* Number of neurons in the layer */
    std::vector<Neuron> neurons;         /* Vector with the neurons of the layer */
};

class MultilayerPerceptron
{
private:
    int nOfLayers{0};                    /* Total number of layers in the network */
    std::vector<Layer> layers;           /* Vector containing every layer */

    // Fill all the weights (w) with random numbers between -1 and +1
    void randomWeights();

    // Feed the input neurons of the network with a vector passed as an argument
    void feedInputs(const std::vector<double>& input);

    // Get the outputs predicted by the network (out vector of the output layer) and save them in the vector passed as an argument
    void getOutputs(std::vector<double>& output);

    // Make a copy of all the weights (copy w into wCopy)
    void copyWeights();

    // Restore a copy of all the weights (copy wCopy into w)
    void restoreWeights();

    // Calculate and propagate the outputs of the neurons, from the first layer until the last one -->-->
    void forwardPropagate();

    // Obtain the output error (MSE) of the out vector of the output layer wrt a target vector and return it
    double obtainError(const std::vector<double>& target);

    // Backpropagate the output error wrt a vector passed as an argument, from the last layer to the first one <--<--
    void backpropagateError(const std::vector<double>& target);

    // Accumulate the changes produced by one pattern and save them in deltaW
    void accumulateChange();

    // Update the network weights, from the first layer to the last one
    void weightAdjustment();

    // Print the network, i.e. all the weight matrices
    void printNetwork();

    // Perform an epoch: forward propagate the inputs, backpropagate the error and adjust the weights
    // input is the input vector of the pattern and target is the desired output vector of the pattern
    void performEpochOnline(const std::vector<double>& input, const std::vector<double>& target);

public:
    // Values of the parameters (they are public and can be updated from outside)
    double eta{0.1};                     // Learning rate
    double mu{0.9};                      // Momentum factor

    // Constructor: Default values for all the parameters
    MultilayerPerceptron();

    // Destructor (Rule of Zero applies; defaulted for clean interface)
    ~MultilayerPerceptron() = default;

    // Allocate memory / resize data structures
    // nl is the number of layers and npl is a vector containing the number of neurons in every layer
    int initialize(int nl, const std::vector<int>& npl);

    // Test the network with a dataset and return the MSE
    double test(const util::Dataset& dataset);

    // Obtain the predicted outputs for a dataset (prints in Kaggle format)
    void predict(const util::Dataset& testDataset);

    // Perform an online training for a specific dataset
    void trainOnline(const util::Dataset& trainDataset);

    // Run the training algorithm for a given number of epochs, using trainDataset.
    // Once finished, check the performance of the network on testDataset.
    // Both training and test MSEs are stored in errorTrain and errorTest.
    void runOnlineBackPropagation(const util::Dataset& trainDataset, const util::Dataset& testDataset,
                                  int maxiter, double& errorTrain, double& errorTest);

    // Optional Kaggle: Save the model weights in a text file
    bool saveWeights(const std::string& fileName);

    // Optional Kaggle: Load the model weights from a text file
    bool readWeights(const std::string& fileName);
};

} // namespace mc

#endif /* MULTILAYERPERCEPTRON_H_ */
