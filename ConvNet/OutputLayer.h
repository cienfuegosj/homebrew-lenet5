#pragma once
class OutputLayer
{
public:
	int numberOfTrials;
	int numberOfTrialsCorrect;

	OutputLayer(int numberOfLabels, bool printEstimates);
	float* Learn(float* X, float* labels);
	~OutputLayer();
private:
	int numberOfLabels = 0;
	float* estimates = nullptr;
	float* dEdY = nullptr;
	bool printEstimates = false;
	void ApplySoftmaxActivation();
	int Max(float* arr, int n);
};


