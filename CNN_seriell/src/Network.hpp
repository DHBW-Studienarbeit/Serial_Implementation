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


#define NO_DATA_D	100
#define NO_TEST_FILES_D	 1
#define NO_PICS_PER_FILE_D	1000
#define LEARNING_RATE 0.5f

using namespace std;

class Network {

private:
	vector<Layer*>* layer_list;
	vector<Matrix*>* node_list;
	vector<Matrix*>* weight_list;
	vector<Matrix*>* bias_list;
	vector<Matrix*>* node_deriv_list;
	vector<Matrix*>* weight_deriv_list;
	vector<Matrix*>* bias_deriv_list;

	PictureContainer* train_picture_container;
	PictureContainer* test_picture_container;

	void reset_backprop_state(void);

public:
	Network();
	~Network();

	void add_Layer(Layer* layer);
	bool generate_network(); /* returns success of function */
	bool train(int batch_size, int no_iterations); /* returns success of function */
	float test();

private:
	bool backpropagate(float* labels);
	float forward(float* labels);
	void gradient_descent(float cost);

};


#endif /* NETWORK_HPP_ */
