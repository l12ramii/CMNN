//============================================================================
// Computational models
// Name        : la1.cpp
// Author      : Pedro A. Gutiérrez
// Version     :
// Copyright   : Universidad de Córdoba
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <ctime>   // To obtain current time time()
#include <cstdlib> // To establish the seed srand() and generate pseudorandom numbers rand()
#include <string.h>
#include <math.h>

#include "mc/MultilayerPerceptron.h"
#include "mc/util.h"
#include <cfloat>

using namespace mc;
using namespace std;
using namespace util;

int main(int argc, char **argv)
{
    // Process arguments of the command line
    bool Tflag = 0, wflag = 0, pflag = 0, tflag = 0, iflag = 0, lflag = 0, hflag = 0, eflag = 0, mflag = 0, nflag = 0;
    char *Tvalue = NULL, *wvalue = NULL, *tvalue = NULL;
    int ivalue = 1000, lvalue = 1, hvalue = 5;
    float evalue = 0.1, mvalue = 0.9;
    int c;

    opterr = 0;

    // a: Option that requires an argument
    // a:: The argument required is optional
    while ((c = getopt(argc, argv, "t:T::i::l:h::e::m::n::w:p")) != -1)
    {
        // The parameters needed for using the optional prediction mode of Kaggle have been included.
        // You should add the rest of parameters needed for the lab assignment.
        switch (c)
        {
        case 't':
            tflag = true;
            tvalue = optarg;
        case 'T':
            Tflag = true;
            Tvalue = optarg;
            break;
        case 'i':
            iflag = true;
            ivalue = atoi(optarg);
        case 'l':
            lflag = true;
            lvalue = atoi(optarg);
        case 'h':
            hflag = true;
            hvalue = atoi(optarg);
        case 'e':
            eflag = true;
            evalue = atof(optarg);
        case 'm':
            mflag = true;
            mvalue = atof(optarg);
        case 'n':
            nflag = true;
        case 'w':
            wflag = true;
            wvalue = optarg;
            break;
        case 'p':
            pflag = true;
            break;
        case '?':
            if (optopt == 't' or optopt == 'l' || optopt == 'w' || optopt == 'p')
                fprintf(stderr, "The option -%c requires an argument.\n", optopt);
            else if (isprint(optopt))
                fprintf(stderr, "Unknown option `-%c'.\n", optopt);
            else
                fprintf(stderr,
                        "Unknown character `\\x%x'.\n",
                        optopt);
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

        // Multilayer perceptron object
        MultilayerPerceptron mlp;

        // Parameters of the mlp. For example, mlp.eta = value;
        mlp.eta = evalue;
        mlp.mu = mvalue;

        int iterations = ivalue;

        // Read training and test data: call to util::readData(...)
        Dataset *trainDataset = util::readData(tvalue);
        Dataset *testDataset = util::readData(Tvalue);  

        // Scale dataset. Use the functions you have in util.h and util.cpp
        if (nflag)
        {
            //hay que hacer más cosas, esto no está acabado todavía

            double *minDataTrain, *minDataTest, *maxDataTrain, *maxDataTest;
            util::obtainMaxMinValuesFromMatrix(trainDataset->inputs, trainDataset->nOfPatterns, trainDataset->nOfInputs, minDataTrain, maxDataTrain);
            util::obtainMaxMinValuesFromMatrix(testDataset->inputs, testDataset->nOfPatterns, testDataset->nOfInputs, minDataTest, maxDataTest);

            util::minMaxScalerDataSetInputs(trainDataset, -1, 1, minDataTrain, maxDataTrain);
            util::minMaxScalerDataSetInputs(testDataset, -1, 1, minDataTest, maxDataTest);
        }

        // Initialize topology vector
        int layers = lvalue;
        int *topology = (int *)malloc(layers * sizeof(int));

        for (int i = 0; i < layers; i++)
        {
            topology[i] = hvalue;
        }

        // Initialize the network using the topology vector
        mlp.initialize(layers + 2, topology);

        // Seed for random numbers
        int seeds[] = {1, 2, 3, 4, 5};
        double *testErrors = new double[5];
        double *trainErrors = new double[5];
        double bestTestError = DBL_MAX;
        for (int i = 0; i < 5; i++)
        {
            cout << "**********" << endl;
            cout << "SEED " << seeds[i] << endl;
            cout << "**********" << endl;
            srand(seeds[i]);
            mlp.runOnlineBackPropagation(trainDataset, testDataset, iterations, &(trainErrors[i]), &(testErrors[i]));
            cout << "We end!! => Final test error: " << testErrors[i] << endl;

            // We save the weights every time we find a better model
            if (wflag && testErrors[i] <= bestTestError)
            {
                mlp.saveWeights(wvalue);
                bestTestError = testErrors[i];
            }
        }

        cout << "WE HAVE FINISHED WITH ALL THE SEEDS" << endl;

        double averageTestError = 0, stdTestError = 0;
        double averageTrainError = 0, stdTrainError = 0;

        // Obtain training and test averages and standard deviations

        cout << "FINAL REPORT" << endl;
        cout << "************" << endl;
        cout << "Train error (Mean +- SD): " << averageTrainError << " +- " << stdTrainError << endl;
        cout << "Test error (Mean +- SD):          " << averageTestError << " +- " << stdTestError << endl;
        return EXIT_SUCCESS;
    }
    else
    {

        //////////////////////////////
        // PREDICTION MODE (KAGGLE) //
        //////////////////////////////

        // Multilayer perceptron object
        MultilayerPerceptron mlp;

        // Initializing the network with the topology vector
        if (!wflag || !mlp.readWeights(wvalue))
        {
            cerr << "Error while reading weights, we can not continue" << endl;
            exit(-1);
        }

        // Reading training and test data: call to util::readData(...)
        Dataset *testDataset;
        testDataset = readData(Tvalue);
        if (testDataset == NULL)
        {
            cerr << "The test file is not valid, we can not continue" << endl;
            exit(-1);
        }

        mlp.predict(testDataset);

        return EXIT_SUCCESS;
    }
}
