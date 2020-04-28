#pragma once

#if __NVCC__
#include <cublas_v2.h>
#endif

class FullyConnectedLayer
{
public:
	struct FullyConnectedLayerParams {
		int numberOfInputs;
		int numberOfOutputs;
		float(*activationFunctionPtr)(float);
	};

	FullyConnectedLayer(FullyConnectedLayerParams* fclp);
	~FullyConnectedLayer();

	float* Forward(float* X);
	float* Backward(float* dEdy);
private:
	int numberOfInputs = 0;
	int numberOfOutputs = 0;
	float learningRate = 0.1f;
	float(*activationFunctionPtr)(float) = nullptr;
	float* weights = nullptr;
	float* bias = nullptr;
	float* result = nullptr;
	float* ptrX = nullptr;
	float* dEdW = nullptr;
	float* dEdX = nullptr;

	void InitializeBias();
	void InitializeWeights();
	void MatrixMultiplication(float* A, float* B, float* C, int m, int n, int k);

	float* Backward_XGrad(float* dEdy);
	void Backward_WGrad(float* dEdy);
	void Learn();

#ifdef __NVCC__
	cublasHandle_t hnd_Cublas;
#endif
};

