/**
 * ConvLayer.cpp
 *
 *  Created on: 29.11.2017
 *      Author: Benjamin Riedle
 */

#include "ConvLayer.hpp"
#include "mathematics.h"

/**
 * The constructor of a convolutional layer needs the specification of
 * the local receptive fields and the step size to generate an output
 *
 * <param> int x_receptive - size of receptive field in x-direction </param>
 * <param> int y_receptive - size of receptive field in y-direction </param>
 * <param> int step_size - step size to move receptive field </param>
 */
Conv_Layer::Conv_Layer(int x_receptive, int y_receptive, int step_size, int no_feature_maps) : Layer(CONV_LAYER){
	this->step_size = step_size;
	this->x_receptive = x_receptive;
	this->y_receptive = y_receptive;
	this->no_feature_maps = no_feature_maps;
}

Conv_Layer::~Conv_Layer() {
}

int Conv_Layer::getNoFeatureMaps()
{
	return no_feature_maps;
}

int Conv_Layer::getXSize()
{
	return x_size;
}

int Conv_Layer::getYSize()
{
	return y_size;
}

void Conv_Layer::setXSize(int size)
{
	this->x_size = size;
}

void Conv_Layer::setYSize(int size)
{
	this->y_size = size;
}

int Conv_Layer::getXReceptive()
{
	return x_receptive;
}

int Conv_Layer::getYReceptive()
{
	return y_receptive;
}

int Conv_Layer::getStepSize()
{
	return step_size;
}


void Conv_Layer::backpropagate( Matrix* inputs,
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
