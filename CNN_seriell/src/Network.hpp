/*
 * network.hpp
 *
 *  Created on: 23.11.2017
 *      Author: Benjamin Riedle
 *
 *  This file defines a class Network, which will determine the final
 *  CNN with all used nodes
 *
 */

#ifndef NETWORK_HPP_
#define NETWORK_HPP_

#include <vector>
#include "Layer.hpp"
#include "matrix.hpp"
#include "InputLayer.hpp"
#include "ConvLayer.hpp"
#include "DropoutLayer.hpp"
#include "FullyConnectedLayer.hpp"
#include "MaxPoolingLayer.hpp"
#include "mathematics.h"
#include "PictureContainer.h"


#define NO_DATA_D	55000
#define NO_TEST_FILES_D	 10
#define NO_PICS_PER_FILE_D	1000

using namespace std;

class Network {

private:
	vector<Layer*>* layer_list;
	vector<Matrix*>* node_list;
	vector<Matrix*>* weight_list;
	vector<Matrix*>* bias_list;

	PictureContainer* train_picture_container;
	PictureContainer* test_picture_container;


public:
	Network();
	~Network();

	void add_Layer(Layer* layer);
	bool generate_network(); /* returns success of function */
	bool train(int batch_size, int no_iterations); /* returns success of function */
	float test();

private:
	bool backpropagate(float cost);
	float forward(float* labels);

};

namespace mathematisches {
	void sigmoid(float* in, float* out, int size);
}

#endif /* NETWORK_HPP_ */
