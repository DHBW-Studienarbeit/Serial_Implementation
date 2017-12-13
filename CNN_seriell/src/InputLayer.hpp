/*
 * InputLayer.hpp
 *
 *  Created on: 29.11.2017
 *      Author: Benjamin Riedle
 */

#ifndef INPUTLAYER_HPP_
#define INPUTLAYER_HPP_

#include "Layer.hpp"

class Input_Layer: public Layer {
private:
	int rows;
	int cols;

public:
	Input_Layer(int rows, int cols);
	~Input_Layer();
	int getRows();
	int getCols();
};

#endif /* INPUTLAYER_HPP_ */
