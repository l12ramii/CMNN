/*********************************************************************
 * File  : MultilayerPerceptron.cpp
 * Date  : 2020
 *********************************************************************/

#include "MultilayerPerceptron.h"

#include "util.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <limits>
#include <cmath>

using namespace mc;
using namespace std;
using namespace util;

/*
// ------------------------------
// Constructor: Default values for all the parameters
MultilayerPerceptron::MultilayerPerceptron()
{
}
*/

MultilayerPerceptron::MultilayerPerceptron(int nl, const std::vector<int> &npl)
{
	// Asignar número de capas
	this->nOfLayers = nl;
	this->layers.resize(nl);

	// Para cada capa, llenarla de neuronas
	for (int i = 0; i < nl; i++)
	{
		this->layers[i].nOfNeurons = npl[i];
		this->layers[i].neurons.resize(npl[i]);
		// para la capa >1, el numero de inputs es el numero de neuronas de la capa anterior + el sesgo
		int nInputs = 0;
		if (i > 0)
		{
			nInputs = this->layers[i - 1].nOfNeurons + 1;
		}
		// para cada neurona, inicializar los pesos a cero
		for (int j = 0; j < npl[i]; j++)
		{
			this->layers[i].neurons[j].w.assign(nInputs, 0.0);
			this->layers[i].neurons[j].deltaW.assign(nInputs, 0.0);
			this->layers[i].neurons[j].lastDeltaW.assign(nInputs, 0.0);
			this->layers[i].neurons[j].wCopy.assign(nInputs, 0.0);
		}
	}
}

/*
// ------------------------------
// DESTRUCTOR: free memory
MultilayerPerceptron::~MultilayerPerceptron()
{
	// freeMemory();
}
*/

/*
// ------------------------------
// Free memory for the data structures
void MultilayerPerceptron::freeMemory() {

}
*/

void MultilayerPerceptron::randomWeights()
{
	for (auto &layer : this->layers)
	{
		for (auto &neuron : layer.neurons)
		{
			for (auto &weight : neuron.w)
			{
				weight = util::randomDouble(-1.0, 1.0);
			}
		}
	}
}

// ------------------------------

void MultilayerPerceptron::feedInputs(const std::vector<double> &input)
{
	for (size_t i = 0; i < input.size(); i++)
	{
		this->layers.front().neurons[i].out = input[i];
	}
}

// ------------------------------

void MultilayerPerceptron::getOutputs(std::vector<double> &output)
{
	for (size_t i = 0; i < output.size(); i++)
	{
		output[i] = this->layers.back().neurons[i].out;
	}
}

// ------------------------------

void MultilayerPerceptron::copyWeights()
{
	for (auto &layer : this->layers)
	{
		for (auto &neuron : layer.neurons)
		{
			neuron.wCopy = neuron.w;
		}
	}
}

// ------------------------------

void MultilayerPerceptron::restoreWeights()
{
	for (auto &layer : this->layers)
	{
		for (auto &neuron : layer.neurons)
		{
			neuron.w = neuron.wCopy;
		}
	}
}

// ------------------------------

void MultilayerPerceptron::forwardPropagate()
{
	// Para cada capa h a partir de la primera
	for (size_t h = 1; h < this->nOfLayers; h++)
	{
		// Para cada neurona j de la capa i
		for (size_t j = 1; j < this->layers[h].nOfNeurons; j++)
		{
			// Calcular el net_h_j
			double net_h_j = this->layers[h].neurons[j].w[0];
			for (size_t i = 1; i < this->layers[h - 1].nOfNeurons; i++)
			{
				net_h_j += this->layers[h].neurons[j].w[i] * this->layers[h - 1].neurons[i].out;
			}
			// Asignar la salida con sigmoide(net_h_j)
			this->layers[h].neurons[j].out = util::sigmoid(net_h_j);
		}
	}
}

// ------------------------------

double MultilayerPerceptron::obtainError(const std::vector<double> &target)
{
	double error = 0.0;
	for (size_t i = 0; i < this->layers.back().nOfNeurons; i++)
	{
		error += std::pow(target[i] - this->layers.back().neurons[i].out, 2);
	}

	return error;
}

// ------------------------------

void MultilayerPerceptron::backpropagateError(const std::vector<double> &target)
{
}

// ------------------------------
// Accumulate the changes produced by one pattern and save them in deltaW
void MultilayerPerceptron::accumulateChange()
{
	// Para cada capa h
	for (size_t h = 1; h < this->nOfLayers; h++)
	{
		// Para cada neurona de la capa h
		for (size_t j = 1; j < this->layers[h].nOfNeurons; j++)
		{
			// Para cada neurona de la capa h - 1
			for (size_t i = 1; i < this->layers[h - 1].nOfNeurons; i++)
			{
				this->layers[h].neurons[j].deltaW[i] = this->layers[h].neurons[j].deltaW[i] + this->layers[h].neurons[j].delta * this->layers[h - 1].neurons[i].out;
			}
			// Sesgo
			this->layers[h].neurons[j].deltaW[0] = this->layers[h].neurons[j].deltaW[0] + this->layers[h].neurons[j].delta;
		}
	}
}

// ------------------------------
// Update the network weights, from the first layer to the last one
void MultilayerPerceptron::weightAdjustment()
{
	// Para cada capa h
	for (size_t h = 1; h < this->nOfLayers; h++)
	{
		// Para cada neurona de la capa h
		for (size_t j = 1; j < this->layers[h].nOfNeurons; j++)
		{
			// Para cada neurona de la capa h - 1
			for (size_t i = 0; i < this->layers[h - 1].nOfNeurons; i++)
			{
				this->layers[h].neurons[j].w[i] = this->layers[h].neurons[j].w[i] - this->eta * this->layers[h].neurons[j].deltaW[i] - this->mu * this->eta * this->layers[h].neurons[j].lastDeltaW[i];
			}
			// Sesgo
			this->layers[h].neurons[j].w[0] = this->layers[h].neurons[j].w[0] - this->eta * this->layers[h].neurons[j].deltaW[0] - this->mu * this->eta * this->layers[h].neurons[j].lastDeltaW[0];
		}
	}
}

// ------------------------------
// Print the network, i.e. all the weight matrices
void MultilayerPerceptron::printNetwork()
{
}

// ------------------------------
// Perform an epoch: forward propagate the inputs, backpropagate the error and adjust the weights
// input is the input vector of the pattern and target is the desired output vector of the pattern
void MultilayerPerceptron::performEpochOnline(const std::vector<double> &input, const std::vector<double> &target)
{
}

// ------------------------------
// Perform an online training for a specific trainDataset
void MultilayerPerceptron::trainOnline(const util::Dataset &trainDataset)
{
	for (size_t i = 0; i < trainDataset.nOfPatterns; i++)
	{
		performEpochOnline(trainDataset.inputs[i], trainDataset.outputs[i]);
	}
}

// ------------------------------
// Test the network with a dataset and return the MSE
double MultilayerPerceptron::test(const util::Dataset &testDataset)
{
	return -1.0;
}

// Optional - KAGGLE
// Test the network with a dataset and return the MSE
// Your have to use the format from Kaggle: two columns (Id y predictied)
void MultilayerPerceptron::predict(const util::Dataset &pDatosTest)
{
	int numSalidas = layers[nOfLayers - 1].nOfNeurons;
	std::vector<double> obtained = std::vector<double>(numSalidas);

	cout << "Id,Predicted" << endl;

	for (size_t i = 0; i < pDatosTest.nOfPatterns; i++)
	{

		feedInputs(pDatosTest.inputs[i]);
		forwardPropagate();
		getOutputs(obtained);

		cout << i;

		for (size_t j = 0; j < numSalidas; j++)
			cout << "," << obtained[j];
		cout << endl;
	}
}

// ------------------------------
// Run the traning algorithm for a given number of epochs, using trainDataset
// Once finished, check the performance of the network in testDataset
// Both training and test MSEs should be obtained and stored in errorTrain and errorTest
void MultilayerPerceptron::runOnlineBackPropagation(const util::Dataset &trainDataset, const util::Dataset &pDatosTest, int maxiter, double &errorTrain, double &errorTest)
{
	int countTrain = 0;

	// Random assignment of weights (starting point)
	randomWeights();

	double minTrainError = 0;
	int iterWithoutImproving = 0;
	double testError = 0;

	// Learning
	do
	{

		trainOnline(trainDataset);
		double trainError = test(trainDataset);
		if (countTrain == 0 || trainError < minTrainError)
		{
			if ((minTrainError - trainError) > 0.00001)
				iterWithoutImproving = 0;
			else
				iterWithoutImproving++;
			minTrainError = trainError;
			copyWeights();
		}
		else
			iterWithoutImproving++;

		if (iterWithoutImproving == 50)
		{
			cout << "We exit because the training is not improving!!" << endl;
			restoreWeights();
			countTrain = maxiter;
		}

		countTrain++;

		cout << "Iteration " << countTrain << "\t Training error: " << trainError << endl;

	} while (countTrain < maxiter);

	cout << "NETWORK WEIGHTS" << endl;
	cout << "===============" << endl;
	printNetwork();

	cout << "Desired output Vs Obtained output (test)" << endl;
	cout << "=========================================" << endl;
	for (int i = 0; i < pDatosTest.nOfPatterns; i++)
	{
		std::vector<double> prediction = std::vector<double>(pDatosTest.nOfOutputs);

		// Feed the inputs and propagate the values
		feedInputs(pDatosTest.inputs[i]);
		forwardPropagate();
		getOutputs(prediction);
		for (int j = 0; j < pDatosTest.nOfOutputs; j++)
			cout << pDatosTest.outputs[i][j] << " -- " << prediction[j] << " ";
		cout << endl;
	}

	testError = test(pDatosTest);
	errorTest = testError;
	errorTrain = minTrainError;
}

// Optional Kaggle: Save the model weights in a textfile
bool MultilayerPerceptron::saveWeights(const std::string &archivo)
{
	// Object for writing the file
	ofstream f(archivo);

	if (!f.is_open())
		return false;

	// Write the number of layers and the number of layers in every layer
	f << nOfLayers;

	for (int i = 0; i < nOfLayers; i++)
		f << " " << layers[i].nOfNeurons;
	f << endl;

	// Write the weight matrix of every layer
	for (int i = 1; i < nOfLayers; i++)
		for (int j = 0; j < layers[i].nOfNeurons; j++)
			for (int k = 0; k < layers[i - 1].nOfNeurons + 1; k++)
				f << layers[i].neurons[j].w[k] << " ";

	f.close();

	return true;
}

// Optional Kaggle: Load the model weights from a textfile
bool MultilayerPerceptron::readWeights(const std::string &archivo)
{
	// Object for reading a file
	ifstream f(archivo);

	if (!f.is_open())
		return false;

	// Number of layers and number of neurons in every layer
	int nl;
	std::vector<int> npl;

	// Read number of layers
	f >> nl;

	npl = std::vector<int>(nl);

	// Read number of neurons in every layer
	for (int i = 0; i < nl; i++)
		f >> npl[i];

	// Initialize vectors and data structures
	*this = MultilayerPerceptron(nl, npl);

	// Read weights
	for (int i = 1; i < nOfLayers; i++)
		for (int j = 0; j < layers[i].nOfNeurons; j++)
			for (int k = 0; k < layers[i - 1].nOfNeurons + 1; k++)
				f >> layers[i].neurons[j].w[k];

	f.close();

	return true;
}
