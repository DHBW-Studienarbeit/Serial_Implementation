/*
 * MaxPoolingLayer.hpp
 *
 *  Created on: 29.11.2017
 *      Author: Benjamin Riedle
 */

#ifndef MAXPOOLINGLAYER_HPP_
#define MAXPOOLINGLAYER_HPP_

#include "Layer.hpp"

class MaxPooling_Layer: public Layer {
private:
	int x_size;
	int y_size;
	int x_receptive;
	int y_receptive;
	int no_features;
public:
	MaxPooling_Layer(int x_receptive, int y_receptive, int no_features);
	virtual ~MaxPooling_Layer();

	int  getNoFeatures();
	int  getXSize();
	int  getYSize();
	void setXSize(int size);
	void setYSize(int size);
	int  getXReceptive();
	int  getYReceptive();

	virtual void backpropagate( Matrix* inputs,
								Matrix* activations,
								Matrix* input_derivations,
								Matrix* activation_derivations,
								Matrix* weights,
								Matrix* biases,
								Matrix* weight_derivations,
								Matrix* bias_derivations ) override;

};

#endif /* MAXPOOLINGLAYER_HPP_ */
