//============================================================================
// Computational models
// Name        : la1.cpp
// Author      : Pedro A. Gutiérrez
// Version     :
// Copyright   : Universidad de Córdoba
//============================================================================

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <numeric>
#include <limits>
#include <cstdlib>
#include <unistd.h>

#include "mc/MultilayerPerceptron.h"
#include "mc/util.h"

int main(int argc, char **argv)
{
    // Process arguments of the command line
    bool wflag = false, pflag = false, nflag = false;

    std::string tvalue;
    std::string Tvalue;
    std::string wvalue;

    int ivalue = 1000;
    int lvalue = 1;
    int hvalue = 5;
    double evalue = 0.1;
    double mvalue = 0.9;
    int c = 0;

    opterr = 0;

    // Command line argument flags:
    // -t: Training file
    // -T: Test file
    // -i: Iterations
    // -l: Hidden layers
    // -h: Neurons per hidden layer
    // -e: Learning rate (eta)
    // -m: Momentum factor (mu)
    // -n: Normalize dataset
    // -w: Weight file
    // -p: Kaggle prediction mode
    while ((c = getopt(argc, argv, "t:T:i:l:h:e:m:nw:p")) != -1)
    {
        switch (c)
        {
        case 't':
            if (optarg) tvalue = optarg;
            break;
        case 'T':
            if (optarg) Tvalue = optarg;
            break;
        case 'i':
            if (optarg) ivalue = std::stoi(optarg);
            break;
        case 'l':
            if (optarg) lvalue = std::stoi(optarg);
            break;
        case 'h':
            if (optarg) hvalue = std::stoi(optarg);
            break;
        case 'e':
            if (optarg) evalue = std::stod(optarg);
            break;
        case 'm':
            if (optarg) mvalue = std::stod(optarg);
            break;
        case 'n':
            nflag = true;
            break;
        case 'w':
            wflag = true;
            if (optarg) wvalue = optarg;
            break;
        case 'p':
            pflag = true;
            break;
        case '?':
            if (optopt == 't' || optopt == 'T' || optopt == 'i' || optopt == 'l' ||
                optopt == 'h' || optopt == 'e' || optopt == 'm' || optopt == 'w')
            {
                std::cerr << "The option -" << static_cast<char>(optopt) << " requires an argument.\n";
            }
            else if (isprint(optopt))
            {
                std::cerr << "Unknown option `-" << static_cast<char>(optopt) << "'.\n";
            }
            else
            {
                std::cerr << "Unknown character `\\x" << std::hex << optopt << "'.\n";
            }
            return EXIT_FAILURE;
        default:
            return EXIT_FAILURE;
        }
    }

    if (!pflag)
    {
        //////////////////////////////////
        // TRAINING AND EVALUATION MODE //
        //////////////////////////////////

        if (tvalue.empty() || Tvalue.empty())
        {
            std::cerr << "Error: Training (-t) and Test (-T) datasets must be specified.\n";
            return EXIT_FAILURE;
        }

        // Multilayer perceptron object
        mc::MultilayerPerceptron mlp;

        // Parameters of the mlp
        mlp.eta = evalue;
        mlp.mu = mvalue;

        int iterations = ivalue;

        // Read training and test data
        util::Dataset trainDataset = util::readData(tvalue);
        util::Dataset testDataset = util::readData(Tvalue);

        if (trainDataset.nOfPatterns == 0 || testDataset.nOfPatterns == 0)
        {
            std::cerr << "Error: Failed to load dataset files properly.\n";
            return EXIT_FAILURE;
        }

        // Scale dataset if requested
        if (nflag)
        {
            std::vector<double> minInputsTrain, maxInputsTrain;
            std::vector<double> minInputsTest, maxInputsTest;
            std::vector<double> minOutputsTrain, maxOutputsTrain;
            std::vector<double> minOutputsTest, maxOutputsTest;

            util::obtainMaxMinValuesFromMatrix(trainDataset.inputs, minInputsTrain, maxInputsTrain);
            util::obtainMaxMinValuesFromMatrix(testDataset.inputs, minInputsTest, maxInputsTest);
            util::obtainMaxMinValuesFromMatrix(trainDataset.outputs, minOutputsTrain, maxOutputsTrain);
            util::obtainMaxMinValuesFromMatrix(testDataset.outputs, minOutputsTest, maxOutputsTest);

            util::minMaxScalerDataSetInputs(trainDataset, -1.0, 1.0, minInputsTrain, maxInputsTrain);
            util::minMaxScalerDataSetInputs(testDataset, -1.0, 1.0, minInputsTest, maxInputsTest);
            util::minMaxScalerDataSetOutputs(trainDataset, 0.0, 1.0, minOutputsTrain, maxOutputsTrain);
            util::minMaxScalerDataSetOutputs(testDataset, 0.0, 1.0, minOutputsTest, maxOutputsTest);
        }

        // Initialize topology vector: input layer, hidden layers, output layer
        int hiddenLayers = lvalue;
        std::vector<int> topology(hiddenLayers + 2);
        topology.front() = trainDataset.nOfInputs;
        for (int i = 1; i <= hiddenLayers; ++i)
        {
            topology[i] = hvalue;
        }
        topology.back() = trainDataset.nOfOutputs;

        // Initialize the network using the topology vector
        mlp.initialize(hiddenLayers + 2, topology);

        // Seed for random numbers
        const std::vector<int> seeds = {1, 2, 3, 4, 5};
        std::vector<double> testErrors(seeds.size(), 0.0);
        std::vector<double> trainErrors(seeds.size(), 0.0);
        double bestTestError = std::numeric_limits<double>::max();

        for (size_t i = 0; i < seeds.size(); ++i)
        {
            std::cout << "**********\n";
            std::cout << "SEED " << seeds[i] << "\n";
            std::cout << "**********\n";
            std::srand(seeds[i]);
            mlp.runOnlineBackPropagation(trainDataset, testDataset, iterations, trainErrors[i], testErrors[i]);
            std::cout << "We end!! => Final test error: " << testErrors[i] << "\n";

            // Save the weights every time a better model is found
            if (wflag && testErrors[i] <= bestTestError)
            {
                mlp.saveWeights(wvalue);
                bestTestError = testErrors[i];
            }
        }

        std::cout << "WE HAVE FINISHED WITH ALL THE SEEDS\n";

        // Calculate training and test error statistics (Mean +- SD)
        double averageTrainError = std::accumulate(trainErrors.begin(), trainErrors.end(), 0.0) / trainErrors.size();
        double averageTestError = std::accumulate(testErrors.begin(), testErrors.end(), 0.0) / testErrors.size();

        double sumSqTrain = 0.0;
        double sumSqTest = 0.0;
        for (size_t i = 0; i < seeds.size(); ++i)
        {
            sumSqTrain += (trainErrors[i] - averageTrainError) * (trainErrors[i] - averageTrainError);
            sumSqTest += (testErrors[i] - averageTestError) * (testErrors[i] - averageTestError);
        }
        double stdTrainError = std::sqrt(sumSqTrain / trainErrors.size());
        double stdTestError = std::sqrt(sumSqTest / testErrors.size());

        std::cout << "\nFINAL REPORT\n";
        std::cout << "************\n";
        std::cout << "Train error (Mean +- SD): " << averageTrainError << " +- " << stdTrainError << "\n";
        std::cout << "Test error (Mean +- SD):          " << averageTestError << " +- " << stdTestError << "\n";

        return EXIT_SUCCESS;
    }
    else
    {
        //////////////////////////////
        // PREDICTION MODE (KAGGLE) //
        //////////////////////////////

        mc::MultilayerPerceptron mlp;

        // Initialize the network with weights from file
        if (!wflag || !mlp.readWeights(wvalue))
        {
            std::cerr << "Error while reading weights, we cannot continue\n";
            return EXIT_FAILURE;
        }

        // Reading test data
        util::Dataset testDataset = util::readData(Tvalue);
        if (testDataset.nOfPatterns == 0)
        {
            std::cerr << "The test file is not valid, we cannot continue\n";
            return EXIT_FAILURE;
        }

        mlp.predict(testDataset);

        return EXIT_SUCCESS;
    }
}
