#include "ConvLayer.h"
#include "SamplingLayer.h"
#include "FullyConnectedLayer.h"
#include "OutputLayer.h"

#pragma once
class CNN
{
public:
	struct CNNParams {
		unsigned char** rawImageData;
		int imageCount;
		int imageWidth;
		int imageHeight;
		int numberOfLabels;
		int numberOfClasses;
		float* floatImageData;
		float* floatLabelData;
	};

	CNN(CNNParams* cp);
	~CNN();

	void TranformRawToFloat();
	void TrainNetwork(int batchSize, int epochs);
	
	float* weights;
private:
	unsigned char** rawImageData = nullptr;
	float* imageData = nullptr;
	float* floatImageData = nullptr;
	float* floatLabelData = nullptr;
	int imageCount = 0;
	int imageWidth = 0;
	int imageHeight = 0;
	int numberOfLabels = 0;
	int numberOfClasses = 0;

	void CreateLayers();
	void PopulateClassification(float* X, float selected);

	// These layers are created to be specific for LeNet5 implementation, but we would like
	// to have an object factory pattern where we can create layers dynamically and attach it to
	// the model.
	ConvLayer* C1 = nullptr;
	SamplingLayer* S2 = nullptr;
	ConvLayer* C3 = nullptr;
	SamplingLayer* S4 = nullptr;
	ConvLayer*C5 = nullptr;
	FullyConnectedLayer* F6 = nullptr;
	FullyConnectedLayer* F7 = nullptr;
	OutputLayer* O = nullptr;
};

