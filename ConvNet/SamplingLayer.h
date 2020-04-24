#pragma once
class SamplingLayer
{
public:
	struct SamplingLayerParams {
		int numberOfInputFeatures;
		int heightOfInputFeatures;
		int widthOfInputsFeatures;
		int neighborhoodDimension;
		float(*activationFunctionPtr)(float);
	};

	SamplingLayer(SamplingLayerParams* slp);
	float* Forward(float* inputFeatureMaps);
	float* Backward(float* dEdY);
	~SamplingLayer();

private:
	float* bias = nullptr;
	float* outputFeatureMaps = nullptr;
	float* dEdX = nullptr;
	float(*activationFunctionPtr)(float) = nullptr;
	int numberOfInputsFeatures = 0;
	int heightOfInputFeature = 0;
	int widthOfInputFeature = 0;
	int neighborhoodDimension = 0;

	void InitializeBias();
};

