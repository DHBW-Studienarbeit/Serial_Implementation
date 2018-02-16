/*
 * Layer.hpp
 *
 *  Created on: 23.11.2017
 *      Author: Benjamin Riedle
 */

#ifndef LAYER_HPP_
#define LAYER_HPP_

#include "matrix.hpp"

typedef enum {
	INPUT_LAYER, CONV_LAYER, POOLING_LAYER, FULLY_CONNECTED_LAYER,
	DROPOUT_LAYER
} LAYER_TYPE;

class Layer
{
private:
	LAYER_TYPE type; /* type of this layer */
	int no_nodes; /* combined number of nodes for this layer */

public:
	Layer(int size, LAYER_TYPE layer_type);
	Layer(LAYER_TYPE layer_type);
	~Layer();

	LAYER_TYPE getLayerType();
	int getSize();
	void setSize(int new_size);

	/**
	 * Method for backpropagation in this layer
	 * must be overrided by each concrete layer type
	 *  the last 4 parameters are ignored for layers which do not provide wheights or biases
	 *  it is recommended to set these to NULL for backpropagation on these types of layers
	 *
	 * inputs: input vector which was used during the last forward process
	 * activations: activation vector which was calculated the last forward process
	 * input_derivations: the method will *OUTPUT* derivations of the cost function from each input here
	 * activation_derivations: derivations of the cost function from each activation
	 * weights: current state of the weights in this layer
	 * biases: current state of the biases in this layer
	 * weight_derivations: the method will *ADD* derivations of the cost function from each weight here
	 * bias_derivations: the method will *ADD* derivations of the cost function from each weight here
	 */
	virtual void backpropagate( Matrix* inputs,
								Matrix* activations,
								Matrix* input_derivations,
								Matrix* activation_derivations,
								Matrix* weights,
								Matrix* biases,
								Matrix* weight_derivations,
								Matrix* bias_derivations ) = 0;
};



#endif /* LAYER_HPP_ */
