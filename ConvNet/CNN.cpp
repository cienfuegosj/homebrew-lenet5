#include "CNN.h"
#include "ConvLayer.h"
#include "SamplingLayer.h"
#include "FullyConnectedLayer.h"
#include "OutputLayer.h"
#include <cmath>
#include <iostream>

float Sigmoid(float x) {
	return float(1.0 / (1.0 + (float)exp(-1.0*x)));
}

CNN::CNN(CNN::CNNParams* cp) {
	this->floatImageData = cp->floatImageData;
	this->floatLabelData = cp->floatLabelData;
	this->rawImageData = cp->rawImageData;
	this->imageCount = cp->imageCount;
	this->imageWidth = cp->imageWidth;
	this->imageHeight = cp->imageHeight;
	this->numberOfLabels = cp->numberOfLabels;
	this->numberOfClasses = cp->numberOfClasses;
	CreateLayers();
}

void CNN::CreateLayers() {

	// Setup C1 convolutional layer
	ConvLayer::ConvLayerParams clp1;
	clp1.stride = 1;
	clp1.widthOfFilterBank = 5;
	clp1.numberOfInputsFeatures = 1;
	clp1.heightOfInputFeature = imageHeight;
	clp1.widthOfInputFeature = imageWidth;
	clp1.numberOfOutputFeatures = 6;
	C1 = new ConvLayer(&clp1);

	// Setup S2 Sampling Layer
	SamplingLayer::SamplingLayerParams slp2;
	slp2.activationFunctionPtr = &Sigmoid;
	slp2.neighborhoodDimension = 2;
	slp2.heightOfInputFeatures = 28;
	slp2.widthOfInputsFeatures = 28;
	slp2.numberOfInputFeatures = 6;
	S2 = new SamplingLayer(&slp2);

	// Setup C3 Convolutional Layer
	ConvLayer::ConvLayerParams clp3;
	clp3.stride = 1;
	clp3.widthOfFilterBank = 5;
	clp3.numberOfInputsFeatures = 6;
	clp3.heightOfInputFeature = 14;
	clp3.widthOfInputFeature = 14;
	clp3.numberOfOutputFeatures = 16;
	C3 = new ConvLayer(&clp3);

	// Setup S4 Sampling Layer
	SamplingLayer::SamplingLayerParams slp4;
	slp4.activationFunctionPtr = &Sigmoid;
	slp4.neighborhoodDimension = 2;
	slp4.heightOfInputFeatures = 10;
	slp4.widthOfInputsFeatures = 10;
	slp4.numberOfInputFeatures = 16;
	S4 = new SamplingLayer(&slp4);

	// Setup C5 Convolutional Layer
	ConvLayer::ConvLayerParams clp5;
	clp5.stride = 1;
	clp5.widthOfFilterBank = 5;
	clp5.numberOfInputsFeatures = 16;
	clp5.heightOfInputFeature = 5;
	clp5.widthOfInputFeature = 5;
	clp5.numberOfOutputFeatures = 120;
	C5 = new ConvLayer(&clp5);

	// Setup F6 Full-Connection Layer
	FullyConnectedLayer::FullyConnectedLayerParams fclp6;
	fclp6.numberOfInputs = 120;
	fclp6.numberOfOutputs = 84;
	fclp6.activationFunctionPtr = &Sigmoid;
	F6 = new FullyConnectedLayer(&fclp6);

	// Setup F7 Full-Connection Layer (Hidden Layer)
	FullyConnectedLayer::FullyConnectedLayerParams fclp7;
	fclp7.numberOfInputs = 84;
	fclp7.numberOfOutputs = 10;
	fclp7.activationFunctionPtr = &Sigmoid;
	F7 = new FullyConnectedLayer(&fclp7);

	// Setup O Output Layer
	O = new OutputLayer(numberOfClasses, true);
}

void CNN::TrainNetwork(int batchSize, int epochs) {
	if (floatImageData == nullptr) return;
	float* accepted = new float[numberOfClasses];

	int numberCorrect = 0;
	for (int i = 0; i < imageCount; i++) {
		float* image = &floatImageData[i * imageHeight * imageWidth];
		float label = floatLabelData[i];
		PopulateClassification(accepted, label);

		float* Y1 = C1->Forward(image);
		float* Y2 = S2->Forward(Y1);
		float* Y3 = C3->Forward(Y2);
		float* Y4 = S4->Forward(Y3);
		float* Y5 = C5->Forward(Y4);
		float* Y6 = F6->Forward(Y5);
		float* Y7 = F7->Forward(Y6);
		float* dEdY7 = O->Learn(Y7, accepted);
		float* dEdY6 = F7->Backward(dEdY7);
		float* dEdY5 = F6->Backward(dEdY6);
		float* dEdY4 = C5->Backward(dEdY5);
		float* dEdY3 = S4->Backward(dEdY4);
		float* dEdY2 = C3->Backward(dEdY3);
		float* dEdY1 = S2->Backward(dEdY2);
		float* dEdY0 = C1->Backward(dEdY1);

		std::cout << "Success rate:\t" << O->numberOfTrialsCorrect << "/" << O->numberOfTrials << "\t" << 100 * ((float)O->numberOfTrialsCorrect / (float)O->numberOfTrials) << "%" << std::endl;
	}

	delete[] accepted;
}

CNN::~CNN() {
	delete C1;
	delete S2;
	delete C3;
	delete S4;
	delete C5;
	delete F6;
	delete F7;
}

void CNN::TranformRawToFloat() {
	if (floatImageData != nullptr) return;
	if (rawImageData == nullptr) return;

	floatImageData = new float[imageCount * imageWidth * imageHeight];

	int c, i, j;
	for (c = 0; c < imageCount; c++) {
		const unsigned char* image = rawImageData[c];
		for (i = 0; i < imageHeight; i++) {
			for (j = 0; j < imageWidth; j++) {
				floatImageData[c*imageWidth*imageHeight + i * imageWidth + j] = (float)image[i*imageHeight + j];
			}
		}
	}
}

void CNN::PopulateClassification(float* X, float selected) {
	for (int i = 0; i < numberOfClasses; i++) {
		X[i] = ((float)i == selected) ? 1.0f : 0.0f;
	}
}
