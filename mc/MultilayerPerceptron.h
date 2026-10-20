/**
 * @file MultilayerPerceptron.h
 * @brief Definition of the MultilayerPerceptron class, Layer class, and Neuron class for neural network models.
 * @date 2020
 */

#ifndef MULTILAYERPERCEPTRON_H_
#define MULTILAYERPERCEPTRON_H_

#include <vector>
#include <string>
#include <iostream>
#include <utility>
#include <functional>

#include "util.h"

/**
 * @namespace mc
 * @brief Namespace containing the Multilayer Perceptron neural network model components.
 */
namespace mc
{
    /**
     * @class Neuron
     * @brief Represents an individual neuron within a neural network layer.
     *
     * Holds activation output, local gradient (delta), synaptic weights, and change buffers for training.
     */
    class Neuron
    {
    public:
        /** Output produced by the neuron (\f$out_j^h\f$) */
        double out{0.0};
        /** Derivative / local error gradient produced by the neuron (\f$\delta_j^h\f$) */
        double delta{0.0};
        /** Input weight vector (\f$w_{ji}^h\f$), where index 0 is the bias weight */
        std::vector<double> w;
        /** Change to be applied to every weight in current step (\f$\Delta w_{ji}^h(t)\f$) */
        std::vector<double> deltaW;
        /** Last change applied to every weight (\f$\Delta w_{ji}^h(t-1)\f$) for momentum */
        std::vector<double> lastDeltaW;
        /** Copy of the input weights for model checkpointing / restoring */
        std::vector<double> wCopy;

        /**
         * @brief Default constructor for Neuron.
         */
        Neuron() = default;
    };

    /**
     * @class Layer
     * @brief Represents a layer of neurons in the multilayer perceptron.
     */
    class Layer
    {
    public:
        /** Number of neurons contained in this layer */
        int nOfNeurons;
        /** Vector containing the neurons of the layer */
        std::vector<Neuron> neurons;

        /**
         * @brief Default constructor for Layer.
         */
        Layer() = default;

        /**
         * @brief Constructs a Layer with a given number of neurons.
         *
         * @param[in] nOfNeurons Number of neurons to initialize in this layer.
         */
        Layer(int nOfNeurons)
        {
            this->nOfNeurons = nOfNeurons;
            this->neurons = std::vector<Neuron>(nOfNeurons);
        }
    };

    /**
     * @class MultilayerPerceptron
     * @brief Implementation of a Multilayer Perceptron (MLP) artificial neural network.
     *
     * Supports feedforward propagation, error backpropagation, online training, momentum,
     * model evaluation (MSE, Cross Entropy), and weight serialization.
     */
    class MultilayerPerceptron
    {
    public:
        /**
         * @brief Function pointer type for error calculation on a pattern.
         */
        using ErrorFunctionType = double (MultilayerPerceptron::*)(const std::vector<double> &);

    private:
        /** Total number of layers in the network (input + hidden + output) */
        int nOfLayers{0};
        /** Vector containing every layer in sequence from input to output */
        std::vector<Layer> layers;

        /**
         * @brief Initializes network weights uniformly using Xavier/Glorot initialization.
         *
         * Samples weights uniformly in \f$[-\frac{1}{\sqrt{n_{h-1}}}, \frac{1}{\sqrt{n_{h-1}}}]\f$,
         * where \f$n_{h-1}\f$ is the number of neurons in the previous layer.
         */
        void randomWeights();

        /**
         * @brief Feeds the input pattern into the neurons of the input layer.
         *
         * @param[in] input Vector of input features for a single pattern.
         */
        void feedInputs(const std::vector<double> &input);

        /**
         * @brief Extracts the outputs predicted by the network's output layer.
         *
         * @param[out] output Vector to store the output activations of the output layer.
         */
        void getOutputs(std::vector<double> &output);

        /**
         * @brief Saves a backup copy of all current synaptic weights into the wCopy vectors.
         */
        void copyWeights();

        /**
         * @brief Restores synaptic weights from the wCopy backup vectors back into the w vectors.
         */
        void restoreWeights();

        /**
         * @brief Computes and propagates outputs of neurons forward from the first layer to the last layer.
         *
         * Hidden layers use the Sigmoid activation function. The output layer uses either
         * Logistic Sigmoid or numerically stable Softmax depending on outSoftmax.
         */
        void forwardPropagate();

        /**
         * @brief Backpropagates the output error gradients from the output layer to the first hidden layer.
         *
         * Computes local error gradients (\f$\delta\f$) based on the active output activation
         * (Sigmoid or Softmax) and loss function (MSE or Cross-Entropy).
         *
         * @param[in] target Desired target output values for the current pattern.
         */
        void backpropagateError(const std::vector<double> &target);

        /**
         * @brief Accumulates the weight updates produced by one pattern and saves them in deltaW.
         */
        void accumulateChange();

        /**
         * @brief Updates network weights by applying the accumulated deltaW and momentum term.
         *
         * In offline/batch training, weight updates are averaged over the total number of patterns.
         *
         * @param[in] nOfPatterns Number of patterns over which gradients were accumulated (defaults to 1 for online).
         */
        void weightAdjustment(int nOfPatterns = 1);

        /**
         * @brief Prints the network architecture and all layer weight matrices to standard output.
         */
        void printNetwork();

        /**
         * @brief Performs a single online training epoch step for one input-target pattern pair.
         *
         * @param[in] input Input feature vector for the pattern.
         * @param[in] target Desired target output vector for the pattern.
         */
        void performEpochOnline(const std::vector<double> &input, const std::vector<double> &target);

        /**
         * @brief Performs a single offline training epoch step for one input-target pattern pair.
         *
         * @param[in] input Input feature vector for the pattern.
         * @param[in] target Desired target output vector for the pattern.
         */
        void performEpochOffline(const std::vector<double> &input, const std::vector<double> &target);

    public:
        /** Learning rate parameter (\f$\eta\f$) */
        double eta{0.1};
        /** Momentum factor parameter (\f$\mu\f$) */
        double mu{0.9};
        /** Output layer activation function (false: Sigmoid, true: Softmax) */
        bool outSoftmax{false};
        /** Error function type indicator (false: MSE, true: Cross-Entropy) */
        bool isCrossEntropy{false};

        /**
         * @brief Default constructor for MultilayerPerceptron.
         */
        MultilayerPerceptron() = default;

        /**
         * @brief Constructs an MLP network with a specified number of layers and layer sizes.
         *
         * @param[in] nl Number of layers (including input, hidden, and output layers).
         * @param[in] npl Vector specifying the number of neurons in each layer.
         */
        MultilayerPerceptron(int nl, const std::vector<int> &npl);

        /**
         * @brief Calculates the error using the default error metric (MSE) for backwards compatibility.
         *
         * @param[in] target Vector of target output values.
         * @return Error for the current pattern.
         */
        double obtainError(const std::vector<double> &target);

        /**
         * @brief Calculates the Mean Squared Error (MSE) of the output layer activations with respect to target values.
         *
         * @param[in] target Vector of target output values.
         * @return Mean Squared Error (MSE) for the current pattern.
         */
        double obtainErrorMSE(const std::vector<double> &target);

        /**
         * @brief Calculates the Cross Entropy error of the output layer activations with respect to target values.
         *
         * @param[in] target Vector of target output values.
         * @return Cross Entropy error for the current pattern.
         */
        double obtainErrorCrossEntropy(const std::vector<double> &target);

        /**
         * @brief Resolves an integer code to an ErrorFunctionType pointer without ifs.
         *
         * @param[in] code 0 for MSE, 1 for Cross Entropy.
         * @return Function pointer to the corresponding error method.
         */
        static ErrorFunctionType getErrorFunction(int code);

        /**
         * @brief Evaluates the network on a given dataset and returns the error using a function pointer.
         *
         * @param[in] dataset The dataset containing input patterns and expected outputs.
         * @param[in] errorFunc Pointer to error member function (&MultilayerPerceptron::obtainErrorMSE, etc.).
         * @return Error across all patterns in the dataset.
         */
        double test(const util::Dataset &dataset, ErrorFunctionType errorFunc);

        /**
         * @brief Tests the network with a dataset and returns the classification correctness rate (CCR).
         *
         * @param[in] dataset Dataset to evaluate.
         * @return CCR (percentage of correctly classified samples).
         */
        double testClassification(const util::Dataset &dataset);

        /**
         * @brief Obtains and prints the predicted outputs for a dataset in Kaggle CSV format (Id,Category).
         *
         * Performs feedforward inference and selects the winning class via argmax.
         *
         * @param[in] testDataset The dataset on which to perform inference.
         */
        void predict(const util::Dataset &testDataset);

        /**
         * @brief Trains the network for one epoch over the given training dataset.
         *
         * @param[in] trainDataset Training dataset.
         * @param[in] onlineTraining Flag indicating whether to use online (true) or offline (false) training.
         */
        void train(const util::Dataset &trainDataset, bool onlineTraining);

        /**
         * @brief Performs backpropagation training on the given datasets using a function pointer.
         *
         * Executes the training loop for up to maxIter epochs with early stopping,
         * tracking training error, test error, and classification correctness rates (CCR).
         * Restores best weights before final evaluation.
         *
         * @param[in] trainDataset Training dataset.
         * @param[in] testDataset Test dataset for evaluation.
         * @param[in] maxIter Maximum number of training iterations.
         * @param[out] errorTrain Training error achieved by best model weights.
         * @param[out] errorTest Test error evaluated on test set using best model weights.
         * @param[out] ccrTrain Classification correctness rate on training set (%).
         * @param[out] ccrTest Classification correctness rate on test set (%).
         * @param[in] eta Learning rate parameter (\f$\eta\f$).
         * @param[in] mu Momentum factor parameter (\f$\mu\f$).
         * @param[in] outSoftmax Whether to use Softmax on the output layer.
         * @param[in] onlineTraining Whether to train in online mode (true) or offline/batch mode (false).
         * @param[in] errorFunc Pointer to error member function (&MultilayerPerceptron::obtainErrorMSE, etc.).
         */
        void runBackPropagation(const util::Dataset &trainDataset,
                                const util::Dataset &testDataset,
                                std::size_t maxIter,
                                double &errorTrain,
                                double &errorTest,
                                double &ccrTrain,
                                double &ccrTest,
                                double eta,
                                double mu,
                                bool outSoftmax,
                                bool onlineTraining,
                                ErrorFunctionType errorFunc);

        /**
         * @brief Saves the network architecture and model weights to a text file.
         *
         * @param[in] fileName Path to the file where weights should be saved.
         * @return true if weights were successfully written, false otherwise.
         */
        bool saveWeights(const std::string &fileName);

        /**
         * @brief Loads the network architecture and model weights from a text file.
         *
         * @param[in] fileName Path to the file from which weights should be read.
         * @return true if weights were successfully read and loaded, false otherwise.
         */
        bool readWeights(const std::string &fileName);
    };

} // namespace mc

#endif /* MULTILAYERPERCEPTRON_H_ */
