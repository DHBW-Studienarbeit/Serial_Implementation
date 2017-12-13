/*
 * Layer.hpp
 *
 *  Created on: 23.11.2017
 *      Author: Benjamin Riedle
 */

#ifndef LAYER_HPP_
#define LAYER_HPP_

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
};



#endif /* LAYER_HPP_ */
