/*
 * DropoutLayer.cpp
 *
 *  Created on: 29.11.2017
 *      Author: Benjamin Riedle
 */

#include "DropoutLayer.hpp"

Dropout_Layer::Dropout_Layer() : Layer(DROPOUT_LAYER) {
	// TODO Auto-generated constructor stub

}

Dropout_Layer::~Dropout_Layer() {
	// TODO Auto-generated destructor stub
}

void Dropout_Layer::backpropagate( Matrix* inputs,
							Matrix* activations,
							Matrix* input_derivations,
							Matrix* activation_derivations,
							Matrix* weights,
							Matrix* biases,
							Matrix* weight_derivations,
							Matrix* bias_derivations )
{

}
