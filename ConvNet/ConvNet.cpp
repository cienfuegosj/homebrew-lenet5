// ConvNet.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include "CNN.h"

int reverseInt(int i) {
	unsigned char c1, c2, c3, c4;
	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;

	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

int main(int argc, char* argv[])
{
	// Read in the training images
	const char* pathTrainingImages = "../MNIST_Database/training/train-images.idx3-ubyte";
	unsigned char** trainingImagesBinary = nullptr;
	
	std::cout << "Reading in training image data..." << std::endl;
	std::ifstream trainingImagesInput(pathTrainingImages, std::ios::binary);

	int NUMBER_OF_IMAGES = 0;
	int NUMBER_OF_ROWS = 0;
	int NUMBER_OF_COLS = 0;

	if (trainingImagesInput.is_open()) {
		int MAGIC_NUMBER = 0;

		// Read in the magic number
		trainingImagesInput.read((char*)&MAGIC_NUMBER, sizeof(MAGIC_NUMBER));
		MAGIC_NUMBER = reverseInt(MAGIC_NUMBER);

		// Read in the number of images
		trainingImagesInput.read((char*)&NUMBER_OF_IMAGES, sizeof(NUMBER_OF_IMAGES));
		NUMBER_OF_IMAGES = reverseInt(NUMBER_OF_IMAGES);

		// Read in the number of rows
		trainingImagesInput.read((char*)&NUMBER_OF_ROWS, sizeof(NUMBER_OF_ROWS));
		NUMBER_OF_ROWS = reverseInt(NUMBER_OF_ROWS);

		// Read in the number of cols
		trainingImagesInput.read((char*)&NUMBER_OF_COLS, sizeof(NUMBER_OF_COLS));
		NUMBER_OF_COLS = reverseInt(NUMBER_OF_COLS);

		int IMAGE_SIZE = NUMBER_OF_ROWS * NUMBER_OF_COLS;
		unsigned char** dataset = new unsigned char*[NUMBER_OF_IMAGES];

		for (int i = 0; i < NUMBER_OF_IMAGES; i++) {
			dataset[i] = new unsigned char[IMAGE_SIZE];
			trainingImagesInput.read((char*)dataset[i], IMAGE_SIZE);
		}

		trainingImagesBinary = dataset;
		std::cout << "Training image read COMPLETE" << std::endl;
	}

	// Read in the training label vector
	const char* pathTrainingLabels = "../MNIST_Database/training/train-labels.idx1-ubyte";

	unsigned char* trainingLabelsBinary = nullptr;

	std::cout << "Reading in training label data..." << std::endl;
	std::ifstream trainingLabelInput(pathTrainingLabels, std::ios::binary);
	int NUMBER_OF_LABELS = 0;
	if (trainingLabelInput.is_open()) {
		int MAGIC_NUMBER = 0;

		// Read in the magic number
		trainingLabelInput.read((char*)&MAGIC_NUMBER, sizeof(MAGIC_NUMBER));
		MAGIC_NUMBER = reverseInt(MAGIC_NUMBER);

		// Read in the number of labels
		trainingLabelInput.read((char*)&NUMBER_OF_LABELS, sizeof(NUMBER_OF_LABELS));
		NUMBER_OF_LABELS = reverseInt(NUMBER_OF_LABELS);

		unsigned char* dataset = new unsigned char[NUMBER_OF_LABELS];
		for (int i = 0; i < NUMBER_OF_LABELS; i++) {
			trainingLabelInput.read((char*)&dataset[i], 1);
		}

		trainingLabelsBinary = dataset;
		std::cout << "Training label read COMPLETE" << std::endl;
	}

	// Converting binary image data into float data making sure to add 2-extra padding on each side to make the MNIST images 32x32
	std::cout << "Converting binary image data into float data with two column paddings on left, right, top, and bottom..." << std::endl;
	NUMBER_OF_ROWS += 4; NUMBER_OF_COLS += 4;
	float* trainingImagesFloat = new float[NUMBER_OF_IMAGES*NUMBER_OF_ROWS*NUMBER_OF_COLS];
	for (int i = 0; i < NUMBER_OF_IMAGES; i++) {
		unsigned char* binaryMatrix = trainingImagesBinary[i];
		for (int j = 0; j < NUMBER_OF_ROWS; j++) {
			for (int k = 0; k < NUMBER_OF_COLS; k++) {
				if (j < 2 || j > NUMBER_OF_ROWS - 3 || k < 2 || k > NUMBER_OF_COLS - 3) {
					trainingImagesFloat[i*NUMBER_OF_ROWS*NUMBER_OF_COLS + j * NUMBER_OF_ROWS + k] = 0.0;
					continue;
				}
				trainingImagesFloat[i*NUMBER_OF_ROWS*NUMBER_OF_COLS + j*NUMBER_OF_ROWS + k] = (float)binaryMatrix[(j - 3) * NUMBER_OF_COLS + (k - 3)];
			}
		}
	}
	std::cout << "Converting COMPLETE" << std::endl;

	// Converting binary label data into float data
	std::cout << "Converting binary label data into float data" << std::endl;
	float* trainingLabelFloat = new float[NUMBER_OF_LABELS];
	for (int i = 0; i < NUMBER_OF_LABELS; i++) {
		trainingLabelFloat[i] = (float)trainingLabelsBinary[i];
	}
	std::cout << "Converting COMPLETE" << std::endl;

	// Create the convolutional neural network
	std::cout << "Creating convolutional neural network..." << std::endl;
	CNN::CNNParams lenet5_params;

	lenet5_params.floatImageData = trainingImagesFloat;
	lenet5_params.floatLabelData = trainingLabelFloat;
	lenet5_params.imageCount = NUMBER_OF_IMAGES;
	lenet5_params.imageHeight = NUMBER_OF_ROWS;
	lenet5_params.imageWidth = NUMBER_OF_COLS;
	lenet5_params.numberOfLabels = NUMBER_OF_LABELS;
	lenet5_params.numberOfClasses = 10;

	CNN* lenet5 = new CNN(&lenet5_params);

	std::cout << "Convolutional neural network creation COMPLETE" << std::endl;

	// Begin training the network

	std::cout << "Starting the training of the CNN..." << std::endl;
	lenet5->TrainNetwork(-1, 0); // Operating in non-batch mode
	std::cout << "Training of the CNN COMPLETE..." << std::endl;

	// Deallocate the training binary images
	for (int i = 0; i < NUMBER_OF_IMAGES; i++) {
		delete trainingImagesBinary[i];
	}
	delete trainingImagesBinary;

	// Deallocate training float images
	delete trainingImagesFloat;

	// Deallocate the training binary labels
	delete trainingLabelsBinary;

	// Deallocate the training float labels
	delete trainingLabelFloat;

	delete lenet5;

	return 0;

}
