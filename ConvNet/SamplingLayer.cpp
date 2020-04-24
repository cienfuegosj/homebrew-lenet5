#include "SamplingLayer.h"

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

float* SamplingLayer::Forward(float* X) {
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

	return outputFeatureMaps;
}

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
	
	return dEdX;
}