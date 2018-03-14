/*
 * FullyConnectedLayer.cpp
 *
 *  Created on: 29.11.2017
 *      Author: Benjamin Riedle
 */

#include "FullyConnectedLayer.hpp"
#include "mathematics.h"

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
	int outputsize = activations->getHeight();
	int inputsize = weights->getLength();
	Matrix *y_deriv_z = new Matrix(outputsize, 1);
	Matrix *y_deriv_x = new Matrix(outputsize, inputsize);
	// make it easier to treat input as vector
	float *inputdata = inputs->get();
	float *inputdata_deriv = input_derivations->get();
	// calc y_deriv_z
	mathematics::sigmoid_backward_derivated(activations->get(), y_deriv_z->get(), outputsize);
	// calc input derivations
	for(int i=0; i<inputsize; i++)
	{
		inputdata_deriv[i]=0.0;
		for(int o=0;o<outputsize; o++)
		{
			y_deriv_x->set(o,i, y_deriv_z->get(o,0) * weights->get(o,i) );
			input_derivations->set(i, 0, input_derivations->get(i,0) + activation_derivations->get(o,0) * y_deriv_x->get(o,i) );
		}
	}
	// calc weight- and bias-derivations
	for(int o=0;o<outputsize; o++)
	{
		for(int i=0; i<inputsize; i++)
		{
			weight_derivations->set(o,i, activation_derivations->get(o,0) * y_deriv_z->get(o,0) * inputdata[i] + weight_derivations->get(o,i) );
		}
		bias_derivations->set(o,0, activation_derivations->get(o,0) * y_deriv_z->get(o,0) + bias_derivations->get(o,0) );
	}
	// clean up locals
	delete y_deriv_z;
	delete y_deriv_x;
}
