/*
 * Layer.cpp
 *
 *  Created on: 23.11.2017
 *      Author: Benjamin Riedle
 */


#include "Layer.hpp"

/*
 * The default constructor for class Layer sets its size
 *
 * <param>int size - no_nodes </param>
 * <param> LAYER_TYPE layer_type </param>
 *
 */
Layer::Layer(int size, LAYER_TYPE layer_type)
{
	no_nodes = size;
	type = layer_type;
}

/*
 * This constructor variant just initializes the layer type
 * The number of nodes has to be set later on during building the Network
 * It is used for convolutional layers because they don't know anything about their input.
 *
 * <param> LAYER_TYPE layer_type - the type of the layer </param>
 *
 */

Layer::Layer(LAYER_TYPE layer_type)
{
	no_nodes = -1;
	type = layer_type;
}

/*
 * The default destructor does nothing atm
 */
Layer::~Layer()
{

}

LAYER_TYPE Layer::getLayerType()
{
	return type;
}

int Layer::getSize()
{
	return no_nodes;
}

void Layer::setSize(int new_size)
{
	this->no_nodes = new_size;
}
