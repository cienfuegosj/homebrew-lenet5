#include "FullyConnectedLayer.h"
#include <random>
#include <iostream>

#ifdef __NVCC__
#include <cublas_v2.h>
#endif

FullyConnectedLayer::FullyConnectedLayer(FullyConnectedLayer::FullyConnectedLayerParams* fclp) {
	this->numberOfInputs = fclp->numberOfInputs;
	this->numberOfOutputs = fclp->numberOfOutputs;
	this->activationFunctionPtr = fclp->activationFunctionPtr;

	InitializeBias();
	InitializeWeights();

	// Initialize the CUBLAS handle if we are using Nvidia CUDA
#ifdef __NVCC__
	cublasCreate(&this->hnd_Cublas);
#endif
}

FullyConnectedLayer::~FullyConnectedLayer() {
	delete[] weights;
	delete[] bias;
	delete[] result;
	delete[] dEdW;
	delete[] dEdX;

#ifdef __NVCC__
	if (this->hnd_Cublas)
		cublasDestroy(this->hnd_Cublas);
#endif
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
#ifdef __CUDACC__
	// Perform traditional host to device and device to host memory handling
	float *d_X, *d_weights, *d_result;
	cudaMalloc((float **)&d_X, sizeof(float) * numberOfInputs);
	cudaMalloc((float **)&d_weights, sizeof(float) * numberOfInputs * numberOfOutputs);
	cudaMalloc((float **)&d_result, sizeof(float) * numberOfOutputs);
	
	cublasSetVector(numberOfInputs, sizeof(float), X, 1, d_X, 1);
	cublasSetMatrix(numberOfInputs, numberOfOutputs, sizeof(float), weights, numberOfOutputs, d_weights, numberOfOutputs);
	
	float const alpha = 1.0f;
	float const beta = 0.0f;
	
	// X * W [1 x numberOfInputs][numberOfInputs x numberOfOutputs]	
	cublasSgemm(hnd_Cublas, CUBLAS_OP_N, CUBLAS_OP_N, numberOfOutputs, 1, numberOfInputs, &alpha, d_weights, numberOfOutputs, d_X, numberOfInputs, &beta, d_result, numberOfOutputs);
	cublasGetVector(numberOfOutputs, sizeof(float), d_result, 1, result, 1);	

	cudaFree(d_X);
	cudaFree(d_weights);
	cudaFree(d_result);
#else
	MatrixMultiplication(X, weights, result, 1, numberOfInputs, numberOfOutputs);
	for (int i = 0; i < numberOfOutputs; i++) {
		result[i] = activationFunctionPtr(result[i] + bias[i]);
	}
#endif
	return result;
}

float* FullyConnectedLayer::Backward_XGrad(float* dEdY) {
	if (!dEdX)
		dEdX = new float[numberOfOutputs * numberOfInputs];
#ifdef __CUDACC__
	// Perform the transposition using cuBLAS
	float const beta = 0.0f;
	float const alpha = 1.0f;
	float* weightsTransposed = new float[numberOfOutputs * numberOfInputs];

	float *d_weights, *d_weightsT;
	cudaMalloc((float **)&d_weights, sizeof(float) * numberOfInputs * numberOfOutputs);
	cudaMalloc((float **)&d_weightsT, sizeof(float) * numberOfOutputs * numberOfInputs);
	cublasSetMatrix(numberOfInputs, numberOfOutputs, sizeof(float), weights, numberOfOutputs, d_weights, numberOfOutputs);
	cublasSgeam(hnd_Cublas, CUBLAS_OP_T, CUBLAS_OP_T, numberOfInputs, numberOfOutputs, &alpha, d_weights, numberOfOutputs, &beta, d_weights, numberOfOutputs, d_weightsT, numberOfInputs);
	cublasGetMatrix(numberOfOutputs, numberOfInputs, sizeof(float), d_weightsT, numberOfInputs, weightsTransposed, numberOfInputs);
	cudaFree(d_weights);
	cudaFree(d_weightsT);

	float *device_dEdY, *device_WTransposed, *device_product;
	cudaMalloc((float **)&device_dEdY, sizeof(float) * numberOfOutputs);
	cudaMalloc((float **)&device_WTransposed, sizeof(float) * numberOfOutputs * numberOfInputs);
	cudaMalloc((float **)&device_product, sizeof(float) * numberOfInputs);
	cublasSetVector(numberOfOutputs, sizeof(float), dEdY, 1, device_dEdY, 1);
	cublasSetMatrix(numberOfOutputs, numberOfInputs, sizeof(float), weightsTransposed, numberOfInputs, device_WTransposed, numberOfInputs); 
	// dEdY * wT [1 x numberOfOutputs][numberOfOutputs x numberOfInputs]
	cublasSgemm(hnd_Cublas, CUBLAS_OP_N, CUBLAS_OP_N, numberOfInputs, 1, numberOfOutputs, &alpha, device_WTransposed, numberOfInputs, device_dEdY, numberOfOutputs, &beta, device_product, numberOfInputs);
	cublasGetVector(numberOfInputs, sizeof(float), device_product, 1, dEdX, 1);
	cudaFree(device_product);
	cudaFree(device_dEdY);
	cudaFree(device_WTransposed);
	delete[] weightsTransposed;
#else
	float *weightsTransposed = new float[numberOfOutputs * numberOfInputs];
	for (int i = 0; i < numberOfOutputs; i++) {
		for (int j = 0; j < numberOfInputs; j++) {
			weightsTranposed[i*numberOfInputs + j] = weights[j*numberOfOutputs + i];
		}
	}

	MatrixMultiplication(dEdY, weightsTranposed, dEdX, 1, numberOfOutputs, numberOfInputs);
	delete[] weightsTransposed;
#endif
	return dEdX;
}

void FullyConnectedLayer::Backward_WGrad(float* dEdY) {
	if (!dEdW)
		dEdW = new float[numberOfInputs * numberOfOutputs];

	// Recall that an 1 x n matrix is just a single dimension vector, so transposing has
	// the same representation in memory
#ifdef __CUDACC__
	float const alpha = 1.0f;
	float const beta = 0.0f;
	float *device_X, *device_dEdY, *device_dEdW;
	cudaMalloc((float **)&device_X, sizeof(float) * numberOfInputs);
	cudaMalloc((float **)&device_dEdY, sizeof(float) * numberOfOutputs);
	cudaMalloc((float **)&device_dEdW, sizeof(float) * numberOfInputs * numberOfOutputs);
	cublasSetVector(numberOfInputs, sizeof(float), ptrX, 1, device_X, 1);
	cublasSetVector(numberOfOutputs, sizeof(float), dEdY, 1, device_dEdY, 1);
	// X * dEdY [numberOfInputs x 1][1 x numberOfOutputs]
	cublasSgemm(hnd_Cublas, CUBLAS_OP_N, CUBLAS_OP_N, numberOfOutputs, numberOfInputs, 1, &alpha, device_dEdY, numberOfOutputs, device_X, 1, &beta, device_dEdW, numberOfOutputs);
	cublasGetMatrix(numberOfInputs, numberOfOutputs, sizeof(float), device_dEdW, numberOfOutputs, dEdW, numberOfInputs);
	cudaFree(device_X);
	cudaFree(device_dEdY);
	cudaFree(device_dEdW);
#else
	MatrixMultiplication(ptrX, dEdY, dEdW, numberOfInputs, 1, numberOfOutputs);
#endif
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
