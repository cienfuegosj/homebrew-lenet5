#include "ConvLayer.h"
#include <random>

ConvLayer::ConvLayer(ConvLayer::ConvLayerParams* clp) {
	this->numberOfOutputFeatures = clp->numberOfOutputFeatures;
	this->numberOfInputsFeatures = clp->numberOfInputsFeatures;
	this->heightOfInputFeature = clp->heightOfInputFeature;
	this->widthOfInputFeature = clp->widthOfInputFeature;
	this->widthOfFilterBank = clp->widthOfFilterBank;
	this->stride = clp->stride;

	InitializeWeights();
}

/* Initialize the weights to random float values
*/
void ConvLayer::InitializeWeights() {

	if (!weights)
		weights = new float[numberOfInputsFeatures * numberOfOutputFeatures * widthOfFilterBank * widthOfFilterBank];

	int m, c, k, j;
	for (m = 0; m < numberOfOutputFeatures; m++) {
		for (c = 0; c < numberOfInputsFeatures; c++) {
			for (k = 0; k < widthOfFilterBank; k++) {
				for (j = 0; j < widthOfFilterBank; j++) {
					this->weights[m * numberOfInputsFeatures * widthOfFilterBank * widthOfFilterBank + c * widthOfFilterBank * widthOfFilterBank + k * widthOfFilterBank + j] = ((float)rand()) / (float)RAND_MAX;
				}
			}
		}
	}
}

/* Performs the forward convolution by performing convolutions
on the input feature maps with the associated filter banks and
storing the information in the output feature maps.
*/
float* ConvLayer::Forward(float* X) {
	inputFeatureMaps = X;

	int m, c, h, w, p, q;
	int Hout = heightOfInputFeature - widthOfFilterBank + 1;
	int Wout = widthOfInputFeature - widthOfFilterBank + 1;

	if (!outputFeatureMaps)
		outputFeatureMaps = new float[numberOfOutputFeatures * Hout * Wout];

	for (m = 0; m < numberOfOutputFeatures; m++) {
		for (h = 0; h < Hout; h++) {
			for (w = 0; w < Wout; w++) {
				outputFeatureMaps[m * Hout * Wout + h*Wout + w] = 0;
				for (c = 0; c < numberOfInputsFeatures; c++) {
					for (p = 0; p < widthOfFilterBank; p++) {
						for (q = 0; q < widthOfFilterBank; q++) {
							int inputFeatureIndex = c * heightOfInputFeature * widthOfInputFeature + (h + p) * widthOfInputFeature + (w + q);
							int weightIndex = m * numberOfInputsFeatures * widthOfFilterBank * widthOfFilterBank + c * widthOfFilterBank * widthOfFilterBank + p * widthOfFilterBank + q;
							outputFeatureMaps[m * Hout * Wout + h * Wout + w] += inputFeatureMaps[inputFeatureIndex] * weights[weightIndex];
						}
					}
				}
			}
		}
	}

	return outputFeatureMaps;
}

/* Calculates dE/dx when given a dE/dy so that it can
send it back to the previous layer.
*/
float* ConvLayer::Backward_XGrad(float* dEdy) {
	int m, c, h, w, p, q;
	int Hout = heightOfInputFeature - widthOfFilterBank + 1;
	int Wout = widthOfInputFeature - widthOfFilterBank + 1;
	
	if (!dEdx)
		dEdx = new float[numberOfInputsFeatures * heightOfInputFeature * widthOfInputFeature];

	for (c = 0; c < numberOfInputsFeatures; c++) {
		for (h = 0; h < heightOfInputFeature; h++) {
			for (w = 0; w < widthOfInputFeature; w++) {
				dEdx[c * heightOfInputFeature * widthOfInputFeature + h * widthOfInputFeature + w] = 0.0f;
			}
		}
	}

	for (m = 0; m < numberOfOutputFeatures; m++) {
		for (h = 0; h < Hout; h++) {
			for (w = 0; w < Wout; w++) {
				for (c = 0; c < numberOfInputsFeatures; c++) {
					for (p = 0; p < widthOfFilterBank; p++) {
						for (q = 0; q < widthOfFilterBank; q++) {
							dEdx[c * heightOfInputFeature * widthOfInputFeature + (h + p) * widthOfInputFeature + (w + q)] += dEdy[m * Hout * Wout + h * Wout + w] * weights[m * numberOfInputsFeatures * widthOfFilterBank * widthOfFilterBank + c * widthOfFilterBank * widthOfFilterBank + p * widthOfFilterBank + q];
						}
					}
				}
			}
		}
	}

	return dEdx;
}


/* Calculate dE/dW when given a dE/dy so that it can
we can reduce W(t+1) to W(t) - k*(dE/dW). Save the pointer to this
data so that we can perform the weight manipulation iteratively.
*/
void ConvLayer::Backward_WGrad(float* dEdy) {
	int m, c, h, w, p, q;
	int Hout = heightOfInputFeature - widthOfFilterBank + 1;
	int Wout = widthOfInputFeature - widthOfFilterBank + 1;

	if (!dEdW)
		dEdW = new float[numberOfOutputFeatures * numberOfInputsFeatures * widthOfFilterBank * widthOfFilterBank];

	for (m = 0; m < numberOfOutputFeatures; m++) {
		for (c = 0; c < numberOfInputsFeatures; c++) {
			for (p = 0; p < widthOfFilterBank; p++) {
				for (q = 0; q < widthOfFilterBank; q++) {
					dEdW[m * numberOfInputsFeatures * widthOfFilterBank * widthOfFilterBank + c * widthOfFilterBank * widthOfFilterBank + p * widthOfFilterBank + q] = 0.0f;
				}
			}
		}
	}

	for (m = 0; m < numberOfOutputFeatures; m++) {
		for (h = 0; h < Hout; h++) {
			for (w = 0; w < Wout; w++) {
				for (c = 0; c < numberOfInputsFeatures; c++) {
					for (p = 0; p < widthOfFilterBank; p++) {
						for (q = 0; q < widthOfFilterBank; q++) {
							dEdW[m * numberOfInputsFeatures * widthOfFilterBank * widthOfFilterBank + c * widthOfFilterBank * widthOfFilterBank + p * widthOfFilterBank + q] += inputFeatureMaps[c * heightOfInputFeature * widthOfInputFeature + (h + p) * widthOfInputFeature + (w + q)] * dEdy[m * Hout * Wout + h * Wout + w];
						}
					}
				}
			}
		}
	}
}

float* ConvLayer::Backward(float* dEdY) {
	Backward_WGrad(dEdY);
	Backward_XGrad(dEdY);
	Learn();
	return dEdx;
}

/* Applies the learning rate by reducing the weights by
W(t + 1) = W(t) - k*(dE/dW)
*/
void ConvLayer::Learn() {
	for (int m = 0; m < numberOfOutputFeatures; m++) {
		for (int c = 0; c < numberOfInputsFeatures; c++) {
			for (int p = 0; p < widthOfFilterBank; p++) {
				for (int q = 0; q < widthOfFilterBank; q++) {
					int index = m * numberOfInputsFeatures * widthOfFilterBank * widthOfFilterBank + c * widthOfFilterBank * widthOfFilterBank + p * widthOfFilterBank + q;
					weights[index] -= learningRate * dEdW[index];
				}
			}
		}
	}
}

/* Delete the allocated error rates, output feature maps, and weights.*/
ConvLayer::~ConvLayer() {
	delete[]dEdx;
	delete[]outputFeatureMaps;
	delete[]weights;
}