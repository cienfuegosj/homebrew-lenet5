#include "OutputLayer.h"
#include <iostream>
#include <cmath>
#include <climits>

OutputLayer::OutputLayer(int numberOfLabels, bool printEstimates) {
	this->numberOfLabels = numberOfLabels;
	this->printEstimates = printEstimates;
	this->dEdY = new float[numberOfLabels];
	numberOfTrials = 0;
	numberOfTrialsCorrect = 0;
}

float* OutputLayer::Learn(float* X, float* labels) {
	numberOfTrials++;

	// Create a new array of floats so that we don't have to modify the input data.
	estimates = new float[numberOfLabels];
	for (int i = 0; i < numberOfLabels; i++) {
		estimates[i] = X[i];
	}

	ApplySoftmaxActivation();
	
	int correctLabel;
	for (int i = 0; i < numberOfLabels; i++) {
		dEdY[i] = labels[i] == 1.0 ? estimates[i] - 1.0 : estimates[i];
		if (labels[i] == 1.0) correctLabel = i;
	}

	if (correctLabel == Max(estimates, numberOfLabels)) {
		numberOfTrialsCorrect++;
	}

	delete[] estimates;

	return dEdY;
}

void OutputLayer::ApplySoftmaxActivation() {
	float distributionSum = 0.0f;
	for (int j = 0; j < numberOfLabels; j++) {
		distributionSum += (float)exp(estimates[j]);
	}

	for (int i = 0; i < numberOfLabels; i++) {
		estimates[i] = (float)exp(estimates[i])/distributionSum;
	}
}

// Returns the position of the maximum value in a float array
int OutputLayer::Max(float* arr, int n) {
	float currentMax = (float)INT_MIN;
	int currentMaxPos = 0;

	for (int i = 0; i < n; i++) {
		if (arr[i] > currentMax) {
			currentMax = arr[i];
			currentMaxPos = i;
		}
	}

	return currentMaxPos;
}

OutputLayer::~OutputLayer() {
	delete[] dEdY;
}