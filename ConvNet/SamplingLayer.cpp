#include "SamplingLayer.h"
#ifdef __NVCC__
#include <cuda_runtime.h>
#endif

SamplingLayer::SamplingLayer(SamplingLayer::SamplingLayerParams* slp) {
	activationFunctionPtr = slp->activationFunctionPtr;
	numberOfInputsFeatures = slp->numberOfInputFeatures;
	heightOfInputFeature = slp->heightOfInputFeatures;
	widthOfInputFeature = slp->widthOfInputsFeatures;
	neighborhoodDimension = slp->neighborhoodDimension;

	bias = new float[numberOfInputsFeatures];
	int size = numberOfInputsFeatures * (heightOfInputFeature / neighborhoodDimension) * (widthOfInputFeature * neighborhoodDimension);
	outputFeatureMaps = new float[size];

	InitializeBias();
}

SamplingLayer::~SamplingLayer() {
	delete[]bias;
	delete[]outputFeatureMaps;
	delete[]dEdX;
}

void SamplingLayer::InitializeBias() {
	for (int i = 0; i < numberOfInputsFeatures; i++) {
		bias[i] = 0.0f;
	}
}

#ifdef __CUDACC__
__global__ void DeviceForward(float* X, float* Y, int nd, int Hin, int Win) {
	int yIndex = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

	float sum = 0.0f;
	for (int p = 0; p < nd; p++) {
		for (int q = 0; q < nd; q++) {
			int xIndex = blockIdx.x * Hin * Win + (nd * threadIdx.y + p) * Win + (nd * threadIdx.x + q);
			sum += X[xIndex];		
		}
	}

	sum /= float(nd * nd);

	Y[yIndex] = sum;
}
#endif

float* SamplingLayer::Forward(float* X) {
#ifdef __CUDACC__
	int Wout = widthOfInputFeature / neighborhoodDimension;
	int Hout = heightOfInputFeature / neighborhoodDimension;
	dim3 grid(numberOfInputsFeatures);
	dim3 block(Wout, Hout);
	
	float *deviceX, *deviceY;
	cudaMalloc((float**)&deviceX, sizeof(float) * numberOfInputsFeatures * heightOfInputFeature * widthOfInputFeature);
	cudaMalloc((float**)&deviceY, sizeof(float) * numberOfInputsFeatures * Hout * Wout);
	cudaMemcpy(deviceX, X, sizeof(float) * numberOfInputsFeatures * heightOfInputFeature * widthOfInputFeature, cudaMemcpyHostToDevice);
	
	DeviceForward<<< grid, block >>>(deviceX, deviceY, neighborhoodDimension, heightOfInputFeature, widthOfInputFeature);
       	cudaMemcpy(outputFeatureMaps, deviceY, sizeof(float) * numberOfInputsFeatures * Hout * Wout, cudaMemcpyDeviceToHost);	
	cudaFree(deviceX);
	cudaFree(deviceY);
#else
	int m, h, w, p, q;
	for (m = 0; m < numberOfInputsFeatures; m++) {
		for (h = 0; h < heightOfInputFeature / neighborhoodDimension; h++) {
			for (w = 0; w < widthOfInputFeature / neighborhoodDimension; w++) {
				int index = m * (heightOfInputFeature / neighborhoodDimension) * (widthOfInputFeature / neighborhoodDimension) + h * (widthOfInputFeature / neighborhoodDimension) + w;
				outputFeatureMaps[index] = 0;
				for (p = 0; p < neighborhoodDimension; p++) {
					for (q = 0; q < neighborhoodDimension; q++) {
						outputFeatureMaps[index] += X[m * heightOfInputFeature * widthOfInputFeature + (neighborhoodDimension * h + p) * widthOfInputFeature + (neighborhoodDimension * w + q)] / float(neighborhoodDimension * neighborhoodDimension);
					}
				}
				outputFeatureMaps[index] = activationFunctionPtr(outputFeatureMaps[index] + bias[m]);
			}
		}
	}
#endif
	return outputFeatureMaps;
}

#ifdef __NVCC__
__global__ void DeviceBackward(float* dY, float* dX, int nd, int Hin, int Win) {
	int dYIndex = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

	int dXRowIdx = 0;
	int dXColIdx = 0;

	for (int p = 0; p < nd; p++) {
		dXRowIdx = nd * threadIdx.y + p;
		for (int q = 0; q < nd; q++) {
			dXColIdx = nd * threadIdx.x + q;
			dX[dXRowIdx, dXColIdx] += dY[dYIndex] / float(nd*nd);	
		}
	}

}
#endif

float* SamplingLayer::Backward(float* dEdY) {
	if (!dEdX)
		dEdX = new float[numberOfInputsFeatures * heightOfInputFeature * widthOfInputFeature];

	for (int c = 0; c < numberOfInputsFeatures; c++) {
		for (int h = 0; h < heightOfInputFeature; h++) {
			for (int w = 0; w < widthOfInputFeature; w++) {
				dEdX[c * heightOfInputFeature * widthOfInputFeature + h * widthOfInputFeature + w] = 0.0f;
			}
		}
	}

#ifdef __CUDACC__

	int Wout = widthOfInputFeature / neighborhoodDimension;
        int Hout = heightOfInputFeature / neighborhoodDimension;
	
	dim3 block(Hout, Wout);
	dim3 grid(numberOfInputsFeatures);
	
	float *device_dEdY, *device_dEdX;
	cudaMalloc((float**)&device_dEdX, sizeof(float) * numberOfInputsFeatures * heightOfInputFeature * widthOfInputFeature);
        cudaMalloc((float**)&device_dEdY, sizeof(float) * numberOfInputsFeatures * Hout * Wout);
        cudaMemcpy(device_dEdX, dEdX, sizeof(float) * numberOfInputsFeatures * heightOfInputFeature * widthOfInputFeature, cudaMemcpyHostToDevice);
	cudaMemcpy(device_dEdY, dEdY, sizeof(float) * numberOfInputsFeatures * Hout * Wout, cudaMemcpyHostToDevice);

	DeviceBackward<<< grid, block >>>(dEdX, dEdY, neighborhoodDimension, Hout, Wout);
	cudaMemcpy(dEdX, device_dEdX, sizeof(float) * numberOfInputsFeatures * heightOfInputFeature * widthOfInputFeature, cudaMemcpyDeviceToHost);
	cudaFree(device_dEdX);
	cudaFree(device_dEdY);
#else	
	for (int m = 0; m < numberOfInputsFeatures; m++) {
		for (int rowPtr = 0; rowPtr < heightOfInputFeature - neighborhoodDimension + 1; rowPtr++) {
			for (int colPtr = 0; colPtr < widthOfInputFeature - neighborhoodDimension + 1; colPtr++) {
				for (int i = 0; i < neighborhoodDimension; i++) {
					for (int j = 0; j < neighborhoodDimension; j++) {
						dEdX[m * heightOfInputFeature * widthOfInputFeature + (rowPtr + i) * widthOfInputFeature + (colPtr + j)] += dEdY[m * (heightOfInputFeature / neighborhoodDimension) * (widthOfInputFeature / neighborhoodDimension) + i * (widthOfInputFeature / neighborhoodDimension) + j] / float(neighborhoodDimension * neighborhoodDimension);
					}
				}
			}
		}
	}
#endif
	return dEdX;
}
