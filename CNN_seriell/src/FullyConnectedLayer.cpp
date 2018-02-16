/*
 * FullyConnectedLayer.cpp
 *
 *  Created on: 29.11.2017
 *      Author: Benjamin Riedle
 */

#include "FullyConnectedLayer.hpp"

FullyConnected_Layer::FullyConnected_Layer(int size) : Layer(size, FULLY_CONNECTED_LAYER) {

}

FullyConnected_Layer::~FullyConnected_Layer() {
	// TODO Auto-generated destructor stub
}

void FullyConnected_Layer::backpropagate( Matrix* inputs,
							Matrix* activations,
							Matrix* input_derivations,
							Matrix* activation_derivations,
							Matrix* weights,
							Matrix* biases,
							Matrix* weight_derivations,
							Matrix* bias_derivations )
{

}
