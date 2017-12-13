/*
 * FullyConnectedLayer.hpp
 *
 *  Created on: 29.11.2017
 *      Author: Benjamin Riedle
 */

#ifndef FULLYCONNECTEDLAYER_HPP_
#define FULLYCONNECTEDLAYER_HPP_

#include "Layer.hpp"

class FullyConnected_Layer: public Layer {
public:
	FullyConnected_Layer(int size);
	~FullyConnected_Layer();
};

#endif /* FULLYCONNECTEDLAYER_HPP_ */
