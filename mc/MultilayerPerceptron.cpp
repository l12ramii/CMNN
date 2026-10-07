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
		for (size_t j = 0; j < this->layers[h].nOfNeurons; j++)
		{
			mc::Neuron &neuron = this->layers[h].neurons[j];
			// Calcular el net
			double net = neuron.w[0];
			for (size_t i = 0; i < this->layers[h - 1].nOfNeurons; i++)
			{
				net += neuron.w[i + 1] * this->layers[h - 1].neurons[i].out;
			}
			// Asignar la salida con sigmoide(net_h_j)
			neuron.out = util::sigmoid(net);
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

	error = error / target.size();

	return error;
}

// ------------------------------

void MultilayerPerceptron::backpropagateError(const std::vector<double> &target)
{
	// Para cada neurona de la capa salida, calcular delta
	for (int j = 0; j < this->layers.back().nOfNeurons; j++)
	{
		mc::Neuron &neuron = this->layers.back().neurons[j];
		neuron.delta = -(target[j] - neuron.out) * neuron.out * (1.0 - neuron.out);
	}

	// Para cada capa desde el final hasta el principio
	for (auto h = this->nOfLayers - 2; h > 0; h--)
	{
		// Para cada neurona de la capa h
		for (int j = 0; j < this->layers[h].nOfNeurons; j++)
		{
			mc::Neuron &neuron = this->layers[h].neurons[j];
			double aux = 0.0;
			// Para cada neurona de la capa h + 1 conectada con j
			for (int i = 0; i < this->layers[h + 1].nOfNeurons; i++)
			{
				aux += this->layers[h + 1].neurons[i].w[j + 1] * this->layers[h + 1].neurons[i].delta;
			}
			neuron.delta = aux * neuron.out * (1 - neuron.out);
		}
	}
}

// ------------------------------

void MultilayerPerceptron::accumulateChange()
{
	// Para cada capa h
	for (size_t h = 1; h < this->nOfLayers; h++)
	{
		// Para cada neurona de la capa h
		for (size_t j = 0; j < this->layers[h].nOfNeurons; j++)
		{
			mc::Neuron &neuron = this->layers[h].neurons[j];
			// Para cada neurona de la capa h - 1
			for (size_t i = 0; i < this->layers[h - 1].nOfNeurons; i++)
			{
				neuron.deltaW[i + 1] += neuron.delta * this->layers[h - 1].neurons[i].out;
			}
			// Sesgo
			neuron.deltaW[0] += neuron.delta;
		}
	}
}

// ------------------------------

void MultilayerPerceptron::weightAdjustment()
{
	// Para cada capa h
	for (size_t h = 1; h < this->nOfLayers; h++)
	{
		// Para cada neurona de la capa h
		for (size_t j = 0; j < this->layers[h].nOfNeurons; j++)
		{
			mc::Neuron &neuron = this->layers[h].neurons[j];
			// Para cada neurona de la capa anterior (h - 1)
			for (size_t i = 0; i < this->layers[h - 1].nOfNeurons; i++)
			{
				neuron.w[i + 1] -= this->eta * neuron.deltaW[i + 1] + this->mu * this->eta * neuron.lastDeltaW[i + 1];
				// Guardar deltaW en t-1 (instante de tiempo anterior)
				neuron.lastDeltaW[i + 1] = neuron.deltaW[i + 1];
			}
			// Sesgo
			neuron.w[0] -= this->eta * neuron.deltaW[0] + this->mu * this->eta * neuron.lastDeltaW[0];
			neuron.lastDeltaW[0] = neuron.deltaW[0];
		}
	}
}

// ------------------------------

void MultilayerPerceptron::printNetwork()
{
	for (size_t h = 1; h < this->nOfLayers; h++)
	{
		std::cout << "Matriz de pesos Capa " << h << ":" << std::endl;
		for (size_t j = 0; j < this->layers[h].nOfNeurons; j++)
		{
			mc::Neuron &neuron = this->layers[h].neurons[j];
			for (size_t i = 0; i < neuron.w.size(); i++)
			{
				std::cout << neuron.w[i] << " ";
			}
			std::cout << std::endl;
		}
	}
}

// ------------------------------

void MultilayerPerceptron::performEpochOnline(const std::vector<double> &input, const std::vector<double> &target)
{
	// Para cada capa
	for (int h = 1; h < this->nOfLayers; h++)
	{
		// Para cada neurona
		for (int j = 0; j < this->layers[h].nOfNeurons; j++)
		{
			// Poner deltaW a cero
			auto &deltaW = this->layers[h].neurons[j].deltaW;
			std::fill(deltaW.begin(), deltaW.end(), 0.0);
		}
	}

	this->feedInputs(input);
	this->forwardPropagate();
	this->backpropagateError(target);
	this->accumulateChange();
	this->weightAdjustment();
}

// ------------------------------

void MultilayerPerceptron::trainOnline(const util::Dataset &trainDataset)
{
	for (size_t i = 0; i < trainDataset.nOfPatterns; i++)
	{
		performEpochOnline(trainDataset.inputs[i], trainDataset.outputs[i]);
	}
}

// ------------------------------

double MultilayerPerceptron::test(const util::Dataset &testDataset)
{

	int numSalidas = this->layers.back().nOfNeurons;
	std::vector<double> obtained = std::vector<double>(numSalidas);

	double mse = 0.0;
	for (size_t i = 0; i < testDataset.nOfPatterns; i++)
	{

		this->feedInputs(testDataset.inputs[i]);
		this->forwardPropagate();
		this->getOutputs(obtained);
		mse += this->obtainError(testDataset.outputs[i]);
	}
	return mse / testDataset.nOfPatterns;
}

// Optional - KAGGLE
// Test the network with a dataset and return the MSE
// Your have to use the format from Kaggle: two columns (Id y predictied)
void MultilayerPerceptron::predict(const util::Dataset &pDatosTest)
{
	int numSalidas = this->layers.back().nOfNeurons;
	std::vector<double> obtained = std::vector<double>(numSalidas);

	cout << "Id,Predicted" << endl;

	for (size_t i = 0; i < pDatosTest.nOfPatterns; i++)
	{

		this->feedInputs(pDatosTest.inputs[i]);
		this->forwardPropagate();
		this->getOutputs(obtained);

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
	this->randomWeights();

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
			{
				iterWithoutImproving = 0;
			}
			else
			{
				iterWithoutImproving++;
			}
			minTrainError = trainError;
			copyWeights();
		}
		else
			iterWithoutImproving++;

		if (iterWithoutImproving == 50)
		{
			std::cout << "We exit because the training is not improving!!" << std::endl;
			this->restoreWeights();
			countTrain = maxiter;
		}

		countTrain++;

		std::cout << "Iteration " << countTrain << "\t Training error: " << trainError << std::endl;

	} while (countTrain < maxiter);

	std::cout << "NETWORK WEIGHTS" << std::endl;
	std::cout << "===============" << std::endl;
	printNetwork();

	std::cout << "Desired output Vs Obtained output (test)" << std::endl;
	std::cout << "=========================================" << std::endl;
	for (int i = 0; i < pDatosTest.nOfPatterns; i++)
	{
		std::vector<double> prediction = std::vector<double>(pDatosTest.nOfOutputs);

		// Feed the inputs and propagate the values
		this->feedInputs(pDatosTest.inputs[i]);
		this->forwardPropagate();
		this->getOutputs(prediction);
		for (int j = 0; j < pDatosTest.nOfOutputs; j++)
		{
			std::cout << pDatosTest.outputs[i][j] << " -- " << prediction[j] << " ";
		}
		std::cout << std::endl;
	}

	testError = test(pDatosTest);
	errorTest = testError;
	errorTrain = minTrainError;
}

// Optional Kaggle: Save the model weights in a textfile
bool MultilayerPerceptron::saveWeights(const std::string &archivo)
{
	// Object for writing the file
	std::ofstream f(archivo);

	if (!f.is_open())
	{
		return false;
	}

	// Write the number of layers and the number of layers in every layer
	f << nOfLayers;

	for (int i = 0; i < nOfLayers; i++)
	{
		f << " " << layers[i].nOfNeurons;
	}
	f << endl;

	// Write the weight matrix of every layer
	for (int i = 1; i < nOfLayers; i++)
	{
		for (int j = 0; j < layers[i].nOfNeurons; j++)
		{
			for (int k = 0; k < layers[i - 1].nOfNeurons + 1; k++)
			{
				f << layers[i].neurons[j].w[k] << " ";
			}
		}
	}
	f.close();

	return true;
}

// Optional Kaggle: Load the model weights from a textfile
bool MultilayerPerceptron::readWeights(const std::string &archivo)
{
	// Object for reading a file
	std::ifstream f(archivo);

	if (!f.is_open())
	{
		return false;
	}

	// Number of layers and number of neurons in every layer
	int nl;
	std::vector<int> npl;

	// Read number of layers
	f >> nl;

	npl = std::vector<int>(nl);

	// Read number of neurons in every layer
	for (int i = 0; i < nl; i++)
	{
		f >> npl[i];
	}

	// Initialize vectors and data structures
	*this = mc::MultilayerPerceptron(nl, npl);

	// Read weights
	for (int i = 1; i < nOfLayers; i++)
	{
		for (int j = 0; j < layers[i].nOfNeurons; j++)
		{
			for (int k = 0; k < layers[i - 1].nOfNeurons + 1; k++)
			{
				f >> layers[i].neurons[j].w[k];
			}
		}
	}

	f.close();

	return true;
}
