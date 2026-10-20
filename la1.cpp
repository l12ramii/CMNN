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
#include <fstream>

#include "mc/MultilayerPerceptron.h"
#include "mc/util.h"

int main(int argc, char **argv)
{
    // Convert command-line arguments to std::vector<std::string>
    std::vector<std::string> args(argv, argv + argc);

    // Process arguments of the command line
    bool wflag = false, pflag = false, nflag = false, sflag = false, softmaxFlag = false, onlineFlag = false;

    std::string tvalue;
    std::string Tvalue;
    std::string wvalue;
    std::string svalue;

    int ivalue = 1000;
    int lvalue = 1;
    int hvalue = 4;
    int fValue = 0;
    double evalue = 0.7;
    double mvalue = 1.0;

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
    // -c / --csv: CSV output file for final metrics
    for (size_t i = 1; i < args.size(); ++i)
    {
        const std::string &arg = args[i];

        if (arg == "-t")
        {
            if (i + 1 < args.size())
            {
                tvalue = args[++i];
            }
            else
            {
                std::cerr << "The option -t requires an argument.\n";
                return EXIT_FAILURE;
            }
        }
        else if (arg == "-T")
        {
            if (i + 1 < args.size())
            {
                Tvalue = args[++i];
            }
            else
            {
                std::cerr << "The option -T requires an argument.\n";
                return EXIT_FAILURE;
            }
        }
        else if (arg == "-i")
        {
            if (i + 1 < args.size())
            {
                ivalue = std::stoi(args[++i]);
            }
            else
            {
                std::cerr << "The option -i requires an argument.\n";
                return EXIT_FAILURE;
            }
        }
        else if (arg == "-l")
        {
            if (i + 1 < args.size())
            {
                lvalue = std::stoi(args[++i]);
            }
            else
            {
                std::cerr << "The option -l requires an argument.\n";
                return EXIT_FAILURE;
            }
        }
        else if (arg == "-h")
        {
            if (i + 1 < args.size())
            {
                hvalue = std::stoi(args[++i]);
            }
            else
            {
                std::cerr << "The option -h requires an argument.\n";
                return EXIT_FAILURE;
            }
        }
        else if (arg == "-e")
        {
            if (i + 1 < args.size())
            {
                evalue = std::stod(args[++i]);
            }
            else
            {
                std::cerr << "The option -e requires an argument.\n";
                return EXIT_FAILURE;
            }
        }
        else if (arg == "-m")
        {
            if (i + 1 < args.size())
            {
                mvalue = std::stod(args[++i]);
            }
            else
            {
                std::cerr << "The option -m requires an argument.\n";
                return EXIT_FAILURE;
            }
        }
        else if (arg == "-n")
        {
            nflag = true;
        }
        else if (arg == "-w")
        {
            wflag = true;
            if (i + 1 < args.size())
            {
                wvalue = args[++i];
            }
            else
            {
                std::cerr << "The option -w requires an argument.\n";
                return EXIT_FAILURE;
            }
        }
        else if (arg == "-p")
        {
            pflag = true;
        }
        else if (arg == "-c" || arg == "--csv" || arg == "-csv")
        {
            sflag = true;
            if (i + 1 < args.size())
            {
                svalue = args[++i];
            }
            else
            {
                std::cerr << "The option " << arg << " requires an argument.\n";
                return EXIT_FAILURE;
            }
        }
        else if (arg == "-f")
        {
            if (i + 1 < args.size())
            {
                fValue = std::stoi(args[++i]);
                if (fValue != 0 and fValue != 1)
                {
                    std::cerr << "The option " << arg << " requires only 0 or 1. \n";
                    return EXIT_FAILURE;
                }
            }
            else
            {
                std::cerr << "The option " << arg << " requires an argument.\n";
                return EXIT_FAILURE;
            }
        }
        else if (arg == "-s")
        {
            softmaxFlag = true;
        }
        else if (arg == "-o")
        {
            onlineFlag = true;
        }
        else
        {
            std::cerr << "Unknown option `" << arg << "'.\n";
            return EXIT_FAILURE;
        }
    }

    if (!pflag)
    {
        //////////////////////////////////
        // TRAINING AND EVALUATION MODE //
        //////////////////////////////////

        if (tvalue.empty())
        {
            std::cerr << "Error: Training (-t) dataset must be specified.\n";
            return EXIT_FAILURE;
        }

        if (Tvalue.empty())
        {
            Tvalue = tvalue;
        }

        // Multilayer perceptron object
        mc::MultilayerPerceptron mlp;

        int iterations = ivalue;

        // Read training and test data
        util::Dataset trainDataset = util::readData(tvalue);
        util::Dataset testDataset = util::readData(Tvalue);

        if (trainDataset.nOfPatterns == 0 || testDataset.nOfPatterns == 0)
        {
            std::cerr << "Error: Failed to load dataset files properly.\n";
            return EXIT_FAILURE;
        }

        // Scale dataset if requested (inputs to [-1, 1], outputs are not normalized in classification)
        if (nflag)
        {
            std::vector<double> minInputsTrain, maxInputsTrain;

            util::obtainMaxMinValuesFromMatrix(trainDataset.inputs, minInputsTrain, maxInputsTrain);

            util::minMaxScalerDataSetInputs(trainDataset, -1.0, 1.0, minInputsTrain, maxInputsTrain);
            util::minMaxScalerDataSetInputs(testDataset, -1.0, 1.0, minInputsTrain, maxInputsTrain);
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
        mlp = mc::MultilayerPerceptron(hiddenLayers + 2, topology);

        // Parameters of the mlp
        mlp.eta = evalue;
        mlp.mu = mvalue;

        // Seed for random numbers
        const std::vector<int> seeds = {1, 2, 3, 4, 5};
        std::vector<double> testErrors(seeds.size(), 0.0);
        std::vector<double> trainErrors(seeds.size(), 0.0);
        std::vector<double> trainCCRs(seeds.size(), 0.0);
        std::vector<double> testCCRs(seeds.size(), 0.0);
        double bestTestError = std::numeric_limits<double>::max();

        mc::MultilayerPerceptron::ErrorFunctionType errorFunc = mc::MultilayerPerceptron::getErrorFunction(fValue);

        for (size_t i = 0; i < seeds.size(); ++i)
        {
            std::cout << "**********\n";
            std::cout << "SEED " << seeds[i] << "\n";
            std::cout << "**********\n";
            util::setSeed(seeds[i]);
            mlp.runBackPropagation(trainDataset,
                                   testDataset,
                                   iterations,
                                   trainErrors[i],
                                   testErrors[i],
                                   trainCCRs[i],
                                   testCCRs[i],
                                   evalue,
                                   mvalue,
                                   softmaxFlag,
                                   onlineFlag,
                                   errorFunc);
            std::cout << "We end!! => Final test CCR: " << testCCRs[i] << "\n";

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
        double averageTrainCCR = std::accumulate(trainCCRs.begin(), trainCCRs.end(), 0.0) / trainCCRs.size();
        double averageTestCCR = std::accumulate(testCCRs.begin(), testCCRs.end(), 0.0) / testCCRs.size();

        double sumSqTrain = 0.0;
        double sumSqTest = 0.0;
        double sumSqTrainCCR = 0.0;
        double sumSqTestCCR = 0.0;
        for (size_t i = 0; i < seeds.size(); ++i)
        {
            sumSqTrain += (trainErrors[i] - averageTrainError) * (trainErrors[i] - averageTrainError);
            sumSqTest += (testErrors[i] - averageTestError) * (testErrors[i] - averageTestError);
            sumSqTrainCCR += (trainCCRs[i] - averageTrainCCR) * (trainCCRs[i] - averageTrainCCR);
            sumSqTestCCR += (testCCRs[i] - averageTestCCR) * (testCCRs[i] - averageTestCCR);
        }
        double stdTrainError = std::sqrt(sumSqTrain / trainErrors.size());
        double stdTestError = std::sqrt(sumSqTest / testErrors.size());
        double stdTrainCCR = std::sqrt(sumSqTrainCCR / trainCCRs.size());
        double stdTestCCR = std::sqrt(sumSqTestCCR / testCCRs.size());

        std::cout << "\nFINAL REPORT\n";
        std::cout << "************\n";
        std::cout << "Train error (Mean +- SD): " << averageTrainError << " +- " << stdTrainError << "\n";
        std::cout << "Test error (Mean +- SD): " << averageTestError << " +- " << stdTestError << "\n";
        std::cout << "Train CCR (Mean +- SD): " << averageTrainCCR << " +- " << stdTrainCCR << "\n";
        std::cout << "Test CCR (Mean +- SD): " << averageTestCCR << " +- " << stdTestCCR << "\n";

        // Save final metrics to CSV if requested
        if (sflag)
        {
            bool fileExistsAndNotEmpty = false;
            bool needsNewline = false;
            {
                std::ifstream checkFile(svalue, std::ios::binary);
                if (checkFile.is_open())
                {
                    checkFile.seekg(0, std::ios::end);
                    std::streampos length = checkFile.tellg();
                    if (length > 0)
                    {
                        fileExistsAndNotEmpty = true;
                        checkFile.seekg(-1, std::ios::end);
                        char lastChar;
                        if (checkFile.get(lastChar) && lastChar != '\n')
                        {
                            needsNewline = true;
                        }
                    }
                }
            }

            std::ofstream csvFile(svalue, std::ios::app);
            if (!csvFile.is_open())
            {
                std::cerr << "Error: Could not open CSV file `" << svalue << "' for writing.\n";
                return EXIT_FAILURE;
            }

            if (!fileExistsAndNotEmpty)
            {
                csvFile << "nCapasOcultas,nNeuronas,mediaErrorTrain,stdErrorTrain,mediaErrorTest,stdErrorTest,mediaCCRTrain,stdCCRTrain,mediaCCRTest,stdCCRTest\n";
            }
            else if (needsNewline)
            {
                csvFile << "\n";
            }

            csvFile << lvalue << "," << hvalue << ","
                    << averageTrainError << "," << stdTrainError << ","
                    << averageTestError << "," << stdTestError << ","
                    << averageTrainCCR << "," << stdTrainCCR << ","
                    << averageTestCCR << "," << stdTestCCR << "\n";
        }

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

        mlp.outSoftmax = softmaxFlag;

        // Reading test data
        util::Dataset testDataset = util::readData(Tvalue);
        if (testDataset.nOfPatterns == 0)
        {
            std::cerr << "The test file is not valid, we cannot continue\n";
            return EXIT_FAILURE;
        }

        if (nflag)
        {
            std::vector<double> minInputsTest, maxInputsTest;
            util::obtainMaxMinValuesFromMatrix(testDataset.inputs, minInputsTest, maxInputsTest);
            util::minMaxScalerDataSetInputs(testDataset, -1.0, 1.0, minInputsTest, maxInputsTest);
        }

        mlp.predict(testDataset);

        return EXIT_SUCCESS;
    }
}
