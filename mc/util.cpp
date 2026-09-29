#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib> // To establish the seed srand() and generate pseudorandom numbers rand()

#include "MultilayerPerceptron.h"
#include "util.h"

using namespace mc;
using namespace std;
using namespace util;

// ------------------------------
// Obtain an integer random number in the range [Low,High]
int util::randomInt(int Low, int High)
{
    return rand() % (High - Low + 1) + Low;
}

// ------------------------------
// Obtain a real random number in the range [Low,High]
double util::randomDouble(double Low, double High)
{
    return ((double)rand() / RAND_MAX) * (High - Low) + Low;
}

// ------------------------------
// Read a dataset from a file name and return it
Dataset util::readData(const std::string &fileName)
{

    ifstream myFile(fileName); // Create an input stream

    if (!myFile.is_open())
    {
        throw std::runtime_error("ERROR: I cannot open the file " + fileName + "\n");
    }

    Dataset dataset = Dataset();

    string line;
    int i, j;

    if (myFile.good())
    {
        getline(myFile, line); // Read a line
        istringstream iss(line);
        iss >> dataset.nOfInputs;
        iss >> dataset.nOfOutputs;
        iss >> dataset.nOfPatterns;
    }
    dataset.inputs = std::vector<std::vector<double>>(dataset.nOfPatterns, std::vector<double>(dataset.nOfInputs, 0.0));
    dataset.outputs = std::vector<std::vector<double>>(dataset.nOfPatterns, std::vector<double>(dataset.nOfOutputs, 0.0));

    i = 0;
    while (myFile.good())
    {
        getline(myFile, line); // Read a line
        if (!line.empty())
        {
            istringstream iss(line);
            for (j = 0; j < dataset.nOfInputs; j++)
            {
                double value;
                iss >> value;
                if (!iss)
                    throw std::runtime_error("ERROR: I cannot open read the line \n");

                dataset.inputs[i][j] = value;
            }
            for (j = 0; j < dataset.nOfOutputs; j++)
            {
                double value;
                iss >> value;
                if (!iss)
                    throw std::runtime_error("ERROR: I cannot read the line \n");

                dataset.outputs[i][j] = value;
            }
            i++;
        }
    }

    myFile.close();

    return dataset;
}

// ------------------------------
// Transform an scalar x by scaling it to a given range [minAllowed, maxAllowed] considering the min
// and max values of the feature in the dataset (minData and maxData).
double util::minMaxScaler(double x, double minAllowed, double maxAllowed, double minData, double maxData)
{
    x = minAllowed + ((x - minData) * (maxAllowed - minAllowed)) / (maxData - minData);
    return x;
}

// ------------------------------
// Scale the dataset inputs to a given range [minAllowed, maxAllowed] considering the min
// and max values of the feature in the dataset (minData and maxData).
void util::minMaxScalerDataSetInputs(Dataset &dataset, double minAllowed, double maxAllowed,
                                     const std::vector<double> &minData, const std::vector<double> &maxData)
{
    for (size_t i = 0; i < dataset.nOfPatterns; i++)
    {
        for (size_t j = 0; j < dataset.nOfInputs; j++)
        {
            dataset.inputs[i][j] = util::minMaxScaler(dataset.inputs[i][j], minAllowed, maxAllowed, minData[j], maxData[j]);
        }
    }
}

// ------------------------------
// Scale the dataset outputs to a given range [minAllowed, maxAllowed] considering the min
// and max values of the output in the dataset (minData and maxData).
void util::minMaxScalerDataSetOutputs(Dataset &dataset, double minAllowed, double maxAllowed,
                                      const std::vector<double> &minData, const std::vector<double> &maxData)
{
    for (size_t i = 0; i < dataset.nOfPatterns; i++)
    {
        for (size_t j = 0; j < dataset.nOfOutputs; j++)
        {
            dataset.outputs[i][j] = util::minMaxScaler(dataset.outputs[i][j], minAllowed, maxAllowed, minData[j], maxData[j]);
        }
    }
}

// ------------------------------
// Extract maximum and minimum values from a matrix of doubles
void util::obtainMaxMinValuesFromMatrix(const std::vector<std::vector<double>> &matrix, std::vector<double> &obtainedMins, std::vector<double> &obtainedMaxs)
{

    if (matrix.empty() or matrix.at(0).empty())
    {
        throw std::runtime_error("ERROR: Matrix is empty. \n");
    }

    size_t numCol = matrix.at(0).size();
    size_t numFil = matrix.size();
    
    obtainedMaxs.resize(numCol);
    obtainedMins.resize(numCol);

    for (size_t c = 0; c < numCol; c++)
    {
        double min = matrix[0][c], max = matrix[0][c];
        for (size_t f = 1; f < numFil; f++)
        {
            if (matrix[f][c] > max)
            {
                max = matrix[f][c];
            }
            if (matrix[f][c] < min)
            {
                min = matrix[f][c];
            }
        }
        obtainedMins[c] = min;
        obtainedMaxs[c] = max;
    }
}

// ------------------------------
// Print the dataset
void util::printDataset(Dataset &dataset, int len)
{
    if (len == 0)
        len = dataset.nOfPatterns;

    for (int i = 0; i < len; i++)
    {
        cout << "P" << i << ":" << endl;
        for (int j = 0; j < dataset.nOfInputs; j++)
        {
            cout << dataset.inputs[i][j] << ",";
        }

        for (int j = 0; j < dataset.nOfOutputs; j++)
        {
            cout << dataset.outputs[i][j] << ",";
        }
        cout << endl;
    }
}
