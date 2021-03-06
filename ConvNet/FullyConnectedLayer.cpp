#include "FullyConnectedLayer.h"
#include <random>

FullyConnectedLayer::FullyConnectedLayer(FullyConnectedLayer::FullyConnectedLayerParams* fclp) {
	this->numberOfInputs = fclp->numberOfInputs;
	this->numberOfOutputs = fclp->numberOfOutputs;
	this->activationFunctionPtr = fclp->activationFunctionPtr;

	InitializeBias();
	InitializeWeights();
}

FullyConnectedLayer::~FullyConnectedLayer() {
	delete[] weights;
	delete[] bias;
	delete[] result;
	delete[] dEdW;
	delete[] dEdX;
}

void FullyConnectedLayer::InitializeBias() {
	if (!bias)
		bias = new float[numberOfOutputs];

	for (int i = 0; i < numberOfOutputs; i++) {
		bias[i] = 0.0f;
	}
}

void FullyConnectedLayer::InitializeWeights() {
	if (!weights)
		weights = new float[numberOfInputs * numberOfOutputs];

	for (int i = 0; i < numberOfInputs; i++) {
		for (int j = 0; j < numberOfOutputs; j++) {
			weights[i*numberOfOutputs + j] = ((float)rand()) / (float)RAND_MAX;
		}
	}
}


float* FullyConnectedLayer::Forward(float* X) {
	ptrX = X;

	if (!result)
		result = new float[numberOfOutputs];

	MatrixMultiplication(X, weights, result, 1, numberOfInputs, numberOfOutputs);
	for (int i = 0; i < numberOfOutputs; i++) {
		result[i] = activationFunctionPtr(result[i] + bias[i]);
	}

	return result;
}

float* FullyConnectedLayer::Backward_XGrad(float* dEdY) {
	if (!dEdX)
		dEdX = new float[numberOfOutputs * numberOfInputs];

	float* weightsTranposed = new float[numberOfInputs * numberOfOutputs];
	for (int i = 0; i < numberOfOutputs; i++) {
		for (int j = 0; j < numberOfInputs; j++) {
			weightsTranposed[i*numberOfInputs + j] = weights[j*numberOfOutputs + i];
		}
	}

	MatrixMultiplication(dEdY, weightsTranposed, dEdX, 1, numberOfOutputs, numberOfInputs);
	delete[] weightsTranposed;
	return dEdX;
}

void FullyConnectedLayer::Backward_WGrad(float* dEdY) {
	if (!dEdW)
		dEdW = new float[numberOfInputs * numberOfOutputs];

	// Recall that an 1 x n matrix is just a single dimension vector, so transposing has
	// the same representation in memory

	MatrixMultiplication(ptrX, dEdY, dEdW, numberOfInputs, 1, numberOfOutputs);
}

float* FullyConnectedLayer::Backward(float* dEdY) {
	Backward_WGrad(dEdY);
	Backward_XGrad(dEdY);
	Learn();
	return dEdX;
}

void FullyConnectedLayer::Learn() {
	for (int i = 0; i < numberOfInputs; i++) {
		for (int j = 0; j < numberOfOutputs; j++) {
			weights[i * numberOfOutputs + j] -= learningRate * dEdW[i * numberOfOutputs + j];
		}
	}
}

/* A = Pointer to a [m x n] float matrix
   B = Pointer [n x l] float matrix
   C = Pointer [m x l] float matrix
*/
void FullyConnectedLayer::MatrixMultiplication(float* A, float* B, float* C, int m, int n, int l) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < l; j++) {
			float sum = 0.0f;
			for (int k = 0; k < n; k++) {
				sum += A[i*n + k] * B[k*l + j];
			}
			C[i*l + j] = sum;
		}
	}
}