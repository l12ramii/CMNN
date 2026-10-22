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
#include <algorithm>
#include <iterator>

using namespace mc;
using namespace std;
using namespace util;

MultilayerPerceptron::ErrorFunctionType MultilayerPerceptron::getErrorFunction(int code)
{
	static const ErrorFunctionType errorFunctions[] = {
		&MultilayerPerceptron::obtainErrorMSE,
		&MultilayerPerceptron::obtainErrorCrossEntropy};

	return errorFunctions[code];
}

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

void MultilayerPerceptron::randomWeights()
{
	for (size_t h = 1; h < this->nOfLayers; h++)
	{
		for (size_t j = 0; j < this->layers[h].nOfNeurons; j++)
		{
			mc::Neuron &neuron = this->layers[h].neurons[j];
			for (size_t i = 0; i < neuron.w.size(); i++)
			{
				neuron.w[i] = util::randomDouble(-(1.0 / std::sqrt(this->layers[h - 1].nOfNeurons)), (1 / std::sqrt(this->layers[h - 1].nOfNeurons)));
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
	// Para cada capa h a partir de la primera excepto la última
	for (size_t h = 1; h < this->nOfLayers - 1; h++)
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

	// Índice de la penúltima capa
	int prevOutLayer = this->nOfLayers - 2;
	if (this->outSoftmax == false)
	{
		// Si no usa softmax, calcular la salida de la última capa con sigmoide
		for (size_t j = 0; j < this->layers.back().nOfNeurons; j++)
		{
			mc::Neuron &neuron = this->layers.back().neurons[j];
			double net = neuron.w[0];

			for (size_t i = 0; i < this->layers[prevOutLayer].nOfNeurons; i++)
			{
				net += neuron.w[i + 1] * this->layers[prevOutLayer].neurons[i].out;
			}
			neuron.out = util::sigmoid(net);
		}
	}
	else
	{
		// Si sí se usa softmax
		std::vector<double> netVector = std::vector<double>(this->layers.back().nOfNeurons);
		for (size_t j = 0; j < this->layers.back().nOfNeurons; j++)
		{
			mc::Neuron &neuron = this->layers.back().neurons[j];
			double net = neuron.w[0];

			for (size_t i = 0; i < this->layers[prevOutLayer].nOfNeurons; i++)
			{
				net += neuron.w[i + 1] * this->layers[prevOutLayer].neurons[i].out;
			}
			netVector[j] = net;
		}

		std::vector<double> outVector = std::vector<double>(this->layers.back().nOfNeurons);
		outVector = util::softmax(netVector);

		for (size_t j = 0; j < this->layers.back().nOfNeurons; j++)
		{
			mc::Neuron &neuron = this->layers.back().neurons[j];
			neuron.out = outVector[j];
		}
	}
}

// ------------------------------

double MultilayerPerceptron::obtainError(const std::vector<double> &target)
{
	return obtainErrorMSE(target);
}

// ------------------------------

double MultilayerPerceptron::obtainErrorMSE(const std::vector<double> &target)
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

double MultilayerPerceptron::obtainErrorCrossEntropy(const std::vector<double> &target)
{
	double error = 0.0;
	size_t k = this->layers.back().nOfNeurons;
	for (size_t i = 0; i < k; i++)
	{
		double out = this->layers.back().neurons[i].out;
		if (out < 1e-10)
		{
			out = 1e-10;
		}
		error += target[i] * std::log(out);
	}

	return -error / k;
}

// ------------------------------

void MultilayerPerceptron::backpropagateError(const std::vector<double> &target)
{
	size_t nH = this->layers.back().nOfNeurons;

	if (!this->outSoftmax)
	{
		// Capa de salida con activación Sigmoide
		for (size_t j = 0; j < nH; j++)
		{
			mc::Neuron &neuron = this->layers.back().neurons[j];
			if (!this->isCrossEntropy)
			{
				// MSE + Sigmoide
				neuron.delta = -(target[j] - neuron.out) * neuron.out * (1.0 - neuron.out);
			}
			else
			{
				// Entropía Cruzada + Sigmoide
				neuron.delta = -target[j] * (1.0 - neuron.out);
			}
		}
	}
	else
	{
		// Capa de salida con activación Softmax
		if (this->isCrossEntropy)
		{
			// Entropía Cruzada + Softmax
			for (size_t j = 0; j < nH; j++)
			{
				mc::Neuron &neuron = this->layers.back().neurons[j];
				neuron.delta = neuron.out - target[j];
			}
		}
		else
		{
			// MSE + Softmax
			double sumTerm = 0.0;
			for (size_t i = 0; i < nH; i++)
			{
				sumTerm += (target[i] - this->layers.back().neurons[i].out) * this->layers.back().neurons[i].out;
			}
			for (size_t j = 0; j < nH; j++)
			{
				mc::Neuron &neuron = this->layers.back().neurons[j];
				neuron.delta = neuron.out * (sumTerm - (target[j] - neuron.out));
			}
		}
	}

	// Para cada capa desde el final hasta el principio
	for (size_t h = this->nOfLayers - 2; h > 0; h--)
	{
		// Para cada neurona de la capa h
		for (size_t j = 0; j < this->layers[h].nOfNeurons; j++)
		{
			mc::Neuron &neuron = this->layers[h].neurons[j];
			double aux = 0.0;
			// Para cada neurona de la capa h + 1 conectada con j
			for (size_t i = 0; i < this->layers[h + 1].nOfNeurons; i++)
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

void MultilayerPerceptron::weightAdjustment(int nOfPatterns)
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
				neuron.w[i + 1] -= (this->eta * neuron.deltaW[i + 1]) / nOfPatterns + (this->mu * this->eta * neuron.lastDeltaW[i + 1]) / nOfPatterns;
				// Guardar deltaW en t-1 (instante de tiempo anterior)
				neuron.lastDeltaW[i + 1] = neuron.deltaW[i + 1];
			}
			// Sesgo
			neuron.w[0] -= (this->eta * neuron.deltaW[0]) / nOfPatterns + (this->mu * this->eta * neuron.lastDeltaW[0]) / nOfPatterns;
			neuron.lastDeltaW[0] = neuron.deltaW[0];
		}
	}
}

// ------------------------------

void MultilayerPerceptron::printNetwork()
{
	for (size_t h = 1; h < this->nOfLayers; h++)
	{
		std::cout << "Layer " << h << std::endl;
		std::cout << "------" << std::endl;
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

void MultilayerPerceptron::performEpochOffline(const std::vector<double> &input, const std::vector<double> &target)
{
	this->feedInputs(input);
	this->forwardPropagate();
	this->backpropagateError(target);
	this->accumulateChange();
}

// ------------------------------

void MultilayerPerceptron::train(const util::Dataset &trainDataset, bool onlineTraining)
{
	if (onlineTraining)
	{
		for (size_t i = 0; i < trainDataset.nOfPatterns; i++)
		{
			this->performEpochOnline(trainDataset.inputs[i], trainDataset.outputs[i]);
		}
	}
	else
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
		for (size_t i = 0; i < trainDataset.nOfPatterns; i++)
		{
			this->performEpochOffline(trainDataset.inputs[i], trainDataset.outputs[i]);
		}
		this->weightAdjustment(trainDataset.nOfPatterns);
	}
}

// ------------------------------

double MultilayerPerceptron::test(const util::Dataset &testDataset, ErrorFunctionType errorFunc)
{
	int numSalidas = this->layers.back().nOfNeurons;
	std::vector<double> obtained = std::vector<double>(numSalidas);

	double totalError = 0.0;
	for (size_t i = 0; i < testDataset.nOfPatterns; i++)
	{
		this->feedInputs(testDataset.inputs[i]);
		this->forwardPropagate();
		this->getOutputs(obtained);
		totalError += (this->*errorFunc)(testDataset.outputs[i]);
	}
	return totalError / testDataset.nOfPatterns;
}

// ------------------------------

double MultilayerPerceptron::testClassification(const util::Dataset &dataset)
{
	int CCR = 0;

	if (dataset.nOfPatterns == 0)
	{
		return 0.0;
	}

	for (size_t i = 0; i < dataset.nOfPatterns; i++)
	{
		this->feedInputs(dataset.inputs[i]);
		this->forwardPropagate();

		std::vector<double> outputs(this->layers.back().nOfNeurons);
		this->getOutputs(outputs);

		// Source - https://stackoverflow.com/a/65163939
		// Posted by Kostas, modified by community. See post 'Timeline' for change history
		// Retrieved 2026-10-19, License - CC BY-SA 4.0
		auto predIt = std::max_element(outputs.begin(), outputs.end());
		int predictedClass = std::distance(outputs.begin(), predIt);

		auto targetIt = std::max_element(dataset.outputs[i].begin(), dataset.outputs[i].end());
		int trueClass = std::distance(dataset.outputs[i].begin(), targetIt);

		if (predictedClass == trueClass)
		{
			CCR++;
		}
	}

	return (static_cast<double>(CCR) / dataset.nOfPatterns) * 100.0;
}

// Optional - KAGGLE
// Test the network with a dataset and return the MSE
// Your have to use the format from Kaggle: two columns (Id y predictied)
void MultilayerPerceptron::predict(const util::Dataset &pDatosTest)
{
	int numSalidas = this->layers.back().nOfNeurons;
	std::vector<double> obtained = std::vector<double>(numSalidas);

	std::cout << "Id,Category" << std::endl;

	for (size_t i = 0; i < pDatosTest.nOfPatterns; i++)
	{
		this->feedInputs(pDatosTest.inputs[i]);
		this->forwardPropagate();
		this->getOutputs(obtained);

		auto predIt = std::max_element(obtained.begin(), obtained.end());
		int predictedClass = std::distance(obtained.begin(), predIt);

		std::cout << i << "," << predictedClass << std::endl;
	}
}

// ------------------------------
// Run the traning algorithm for a given number of epochs, using trainDataset
// Once finished, check the performance of the network in testDataset
// Both training and test errors should be obtained and stored in errorTrain and errorTest
void MultilayerPerceptron::runBackPropagation(const util::Dataset &trainDataset,
											  const util::Dataset &pDatosTest,
											  std::size_t maxIter,
											  double &errorTrain,
											  double &errorTest,
											  double &ccrTrain,
											  double &ccrTest,
											  double eta,
											  double mu,
											  bool outSoftmax,
											  bool onlineTraining,
											  int errorFunction)
{
	int countTrain = 0;

	// Random assignment of weights (starting point)
	this->randomWeights();
	this->eta = eta;
	this->mu = mu;
	this->outSoftmax = outSoftmax;
	this->isCrossEntropy = (errorFunction == 1);
	mc::MultilayerPerceptron::ErrorFunctionType errorFunc = mc::MultilayerPerceptron::getErrorFunction(errorFunction);
	double minTrainError = 0;
	int iterWithoutImproving = 0;
	double testError = 0;

	// Learning
	do
	{

		train(trainDataset, onlineTraining);
		double trainError = test(trainDataset, errorFunc);
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
			countTrain = maxIter;
		}

		countTrain++;

		std::cout << "Iteration " << countTrain << "	 Training error: " << trainError << std::endl;

	} while (countTrain < maxIter);

	this->restoreWeights();

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

	testError = test(pDatosTest, errorFunc);
	ccrTrain = testClassification(trainDataset);
	ccrTest = testClassification(pDatosTest);
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
