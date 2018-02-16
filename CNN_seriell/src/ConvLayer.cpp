/**
 * ConvLayer.cpp
 *
 *  Created on: 29.11.2017
 *      Author: Benjamin Riedle
 */

#include "ConvLayer.hpp"

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

}
