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
        int nOfNeurons;              /**< Number of neurons contained in this layer */
        std::vector<Neuron> neurons; /**< Vector containing the neurons of the layer */

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
     * Supports feedforward propagation, error backpropagation, online and offline training, momentum,
     * model evaluation (MSE), and weight serialization.
     */
    class MultilayerPerceptron
    {
    private:
        /** Total number of layers irandomWeightsn the network (input + hidden + output) */
        int nOfLayers{0};
        /** Vector containing every layer in sequence from input to output */
        std::vector<Layer> layers;

        /**
         * @brief Initializes all network weights with random values uniformly sampled in [-1.0, 1.0].
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
         */
        void forwardPropagate();

        /**
         * @brief Calculates the Mean Squared Error (MSE) of the output layer activations with respect to target values.
         *
         * @param[in] target Vector of target output values.
         * @return Mean Squared Error (MSE) for the current pattern.
         */
        double obtainError(const std::vector<double> &target);

        /**
         * @brief Backpropagates the output error gradients from the output layer to the first hidden layer.
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
         */
        void weightAdjustment();

        /**
         * @brief Prints the network architecture and all layer weight matrices to standard output.
         */
        void printNetwork();

        /**
         * @brief Performs a single online training epoch step for one input-target pattern pair.
         *
         * Feeds inputs, computes forward propagation, backpropagates error, and adjusts weights.
         *
         * @param[in] input Input feature vector for the pattern.
         * @param[in] target Desired target output vector for the pattern.
         */
        void performEpochOnline(const std::vector<double> &input, const std::vector<double> &target);

    public:
        double eta{0.1}; /**< Learning rate parameter (\f$\eta\f$) */
        double mu{0.9};  /**< Momentum factor parameter (\f$\mu\f$) */

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
         * @brief Default destructor for MultilayerPerceptron.
         */
        ~MultilayerPerceptron() = default;

        /**
         * @brief Evaluates the network on a given dataset and returns the Mean Squared Error (MSE).
         *
         * @param[in] dataset The dataset containing input patterns and expected outputs.
         * @return Mean Squared Error (MSE) across all patterns in the dataset.
         */
        double test(const util::Dataset &dataset);

        /**
         * @brief Obtains and prints the predicted outputs for a dataset in Kaggle CSV format (Id,Predicted).
         *
         * @param[in] testDataset The dataset on which to perform inference.
         */
        void predict(const util::Dataset &testDataset);

        /**
         * @brief Performs one online training epoch over the entire training dataset.
         *
         * @param[in] trainDataset Dataset containing training patterns.
         */
        void trainOnline(const util::Dataset &trainDataset);

        /**
         * @brief Executes the online backpropagation training algorithm for up to maxiter iterations.
         *
         * Monitors training performance with early stopping, keeps track of the best weights,
         * and reports training and test errors.
         *
         * @param[in] trainDataset Training dataset.
         * @param[in] testDataset Test dataset used for performance evaluation.
         * @param[in] maxiter Maximum number of training epochs/iterations.
         * @param[out] errorTrain Reference to store the resulting minimum training MSE.
         * @param[out] errorTest Reference to store the resulting test MSE evaluated on testDataset.
         */
        void runOnlineBackPropagation(const util::Dataset &trainDataset, const util::Dataset &testDataset,
                                      int maxiter, double &errorTrain, double &errorTest);

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
