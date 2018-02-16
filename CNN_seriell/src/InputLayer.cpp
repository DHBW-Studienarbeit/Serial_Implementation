/*
 * InputLayer.cpp
 *
 *  Created on: 29.11.2017
 *      Author: Benjamin Riedle
 */

#include "InputLayer.hpp"

/*
 * The default constructor of Conv_Layer uses the
 * constructor of the base class Layer to set number
 * of layers nodes
 *
 * <params> int size - number of layers nodes </params>
 *
 */
Input_Layer::Input_Layer(int rows, int cols) : Layer(rows*cols,INPUT_LAYER) {

	this->rows = rows;
	this->cols = cols;
}

Input_Layer::~Input_Layer() {

}

int Input_Layer::getCols()
{
	return cols;
}

int Input_Layer::getRows()
{
	return rows;
}


void Input_Layer::backpropagate( Matrix* inputs,
							Matrix* activations,
							Matrix* input_derivations,
							Matrix* activation_derivations,
							Matrix* weights,
							Matrix* biases,
							Matrix* weight_derivations,
							Matrix* bias_derivations )
{
	
}
