#pragma once

class ConvLayer {
public:
	struct ConvLayerParams {
		int numberOfOutputFeatures;
		int numberOfInputsFeatures;
		int heightOfInputFeature;
		int widthOfInputFeature;
		int widthOfFilterBank;
		int stride;
	};

	ConvLayer() { }
	ConvLayer(ConvLayerParams* clp);
	float* Forward(float* X);
	float* Backward(float* dEdy);
	~ConvLayer();


private:
	int numberOfOutputFeatures = 0;
	int numberOfInputsFeatures = 0;
	int heightOfInputFeature = 0;
	int widthOfInputFeature = 0;
	int widthOfFilterBank = 0;
	float learningRate = 0.1f;
	float* inputFeatureMaps = nullptr;
	float* weights = nullptr;
	float* outputFeatureMaps = nullptr;
	float* dEdW = nullptr;
	float* dEdx = nullptr;
	int stride = 0;
	
	void InitializeWeights();
	float* Backward_XGrad(float* dEdy);
	void Backward_WGrad(float* dEdy);
	void Learn();
};

