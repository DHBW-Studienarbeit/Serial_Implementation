//============================================================================
// Name        : CNN_seriell.cpp
// Author      : Josua Benz, Benjamin Riedle, Florian Schmidt
// Version     :
// Copyright   : Open Source - Take what you can get
// Description : CNN as a serial implementation
//============================================================================

#include <iostream>
#include <string>
#include "./testfile.h"
#include "matrix.hpp"
#include "Network.hpp"

#define PICTURE_SIZE_D			28
#define CONV_SIZE_1_D			5
#define CONV_SIZE_2_D			5
#define CONV_STEP_1_D 			1
#define CONV_STEP_2_D			1
#define CONV_FEATURES_1_D		6
#define CONV_FEATURES_2_D		8
#define MAX_POOL_SIZE_1_D		2
#define MAX_POOL_SIZE_2_D		2
#define FULLY_CONNECTED_SIZE_D	10

#define BATCH_SIZE_D	1000
#define NO_ITERATIONS_D 1

using namespace std;

int main(int argc, char **argv) {
	bool correct_net = false;
	bool train_success = false;
	float accuracy = 0.0f;

	Network* network = new Network();

	network->add_Layer((Layer*) new Input_Layer(PICTURE_SIZE_D, PICTURE_SIZE_D));
	network->add_Layer((Layer*) new Conv_Layer(CONV_SIZE_1_D, CONV_SIZE_1_D, CONV_STEP_1_D, CONV_FEATURES_1_D));
	network->add_Layer((Layer*) new MaxPooling_Layer(MAX_POOL_SIZE_1_D, MAX_POOL_SIZE_1_D, CONV_FEATURES_1_D));
	network->add_Layer((Layer*) new Conv_Layer(CONV_SIZE_2_D, CONV_SIZE_2_D, CONV_STEP_2_D, CONV_FEATURES_2_D));
	network->add_Layer((Layer*) new MaxPooling_Layer(MAX_POOL_SIZE_2_D, MAX_POOL_SIZE_2_D, CONV_FEATURES_2_D));
	network->add_Layer((Layer*) new FullyConnected_Layer(FULLY_CONNECTED_SIZE_D));

	std::cout << "Generate CNN..." << std::endl;

	correct_net = network->generate_network();
	if(correct_net)
	{
		std::cout << "Generation was successful!" << std::endl;
		train_success = network->train(BATCH_SIZE_D, NO_ITERATIONS_D);
	}
	else
	{
		std::cout<< "Bad network configuration! Program ends!" << std::endl;
	}

	if(train_success)
	{
		std::cout << "Training finished! Start Test..." << std::endl;
		accuracy = network->test();
		std::cout << "Accuracy: " << accuracy << std::endl;
	}
	else
	{
		std::cout << "Training failed!" << std::endl;
	}


	return 0;
}
